from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser, MultifieldParser
from transformers import pipeline
from pymongo import MongoClient
import os
import spacy
nlp = spacy.load("en_core_web_sm")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://nghednh:123@cluster0.llhn1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client['movie_db']  # Database name
collection = db['movies']  # Collection name

# Initialize the NLP model (BERT for query analysis)
nlp_model = pipeline('feature-extraction', model='bert-base-uncased', framework='pt')

# Whoosh Schema for movie data
from whoosh.fields import Schema, TEXT, ID

# Whoosh Schema for movie data
schema = Schema(
    movie_id=ID(stored=True, unique=True),  # Unique identifier for the movie
    name=TEXT(stored=True),  # Higher weight for the name field
    overview=TEXT(stored=True),  # Lower weight for overview
    subtitle=TEXT(stored=True)  # Optional subtitle field
)


# Index directory
index_dir = "movie_text_index"

# Function to check if index exists, if not create it
def get_index():
    if os.path.exists(index_dir):
        try:
            return open_dir(index_dir)  # Attempt to open existing index
        except:
            print("Index not found, creating a new one.")  # If open fails, we'll create a new index
            return create_in(index_dir, schema)  # Create the index if it doesn't exist
    else:
        print("Index directory not found, creating a new one.")  # Directory doesn't exist
        os.makedirs(index_dir)  # Create the directory if it doesn't exist
        return create_in(index_dir, schema)  # Create the index

# Function to update index without reindexing everything
def update_index():
    print("Updating index...")  # Debugging statement
    ix = get_index()  # Open or create the index
    writer = ix.writer()

    # Loop through each movie in the database
    for movie in collection.find():  # Adjust limit as necessary
        # Use update_document with movie_id to ensure it updates the existing document
        writer.update_document(
            movie_id=str(movie['movie_id']),  # Ensure it's treated as the unique identifier
            name=movie['name'],
            overview=movie['overview'],
            subtitle=movie['subtitle']
        )

    writer.commit()
    print("Index update complete.")  # Debugging statement


# Process the user's query (NLP model analysis)
def process_query(query):
    print(f"Processing query: {query}")  # Debugging statement
    query_embedding = nlp_model(query)
    return query_embedding


def get_synonyms(text):
    pass


def expand_query(query):
    doc = nlp(query)
    expanded_query = query
    for token in doc:
        # Example: If token has synonyms, add them to the query
        if token.has_vector:
            synonyms = get_synonyms(token.text)  # You can define a function for this
            if synonyms and isinstance(synonyms, (list, set)):  # Ensure synonyms is an iterable
                expanded_query += " " + " ".join(synonyms)  # Join synonyms as space-separated string
    return expanded_query

# Search movies based on user input

def search_movies(query, top_k=10):
    expanded_query = expand_query(query)  # Expand the query
    query_embedding = process_query(expanded_query)  # Use expanded query for NLP processing
    ix = get_index()  # Open the existing index
    seen_movie_ids = set()  # To track already seen movie IDs

    with ix.searcher() as searcher:
        # Parse the query to search in both 'overview' and 'name' fields
        query_parsed = MultifieldParser(["overview", "name"], ix.schema).parse(expanded_query)

        # Fetch scored results, sorted by score in descending order (default behavior)
        results = searcher.search(query_parsed, scored=True)

        # Check if any results were found
        if results:
            count = 0
            for result in results:
                # Add the movie to the result set and ensure we are not showing duplicates
                # if result['movie_id'] not in seen_movie_ids:
                if True:
                    seen_movie_ids.add(result['movie_id'])
                    print(f"Movie found: {result['name']}, Movie ID: {result['movie_id']}, Score: {result.score}")
                    count += 1
                # Stop once we have collected the top_k results
                if count >= top_k:
                    break
        else:
            print("No results found for the query.")


# Run index update and search example
#update_index()  # Update the index with new/modified movie data
search_movies("Dog")  # Example search query
