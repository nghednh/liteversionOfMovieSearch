from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
from transformers import pipeline
from pymongo import MongoClient
import os
import spacy

# Load spaCy for query expansion
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
        except Exception as e:
            print(f"Error opening index: {e}")  # Log the error
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
            subtitle=movie.get('subtitle', '')  # Subtitle might not exist for all movies
        )

    writer.commit()
    print("Index update complete.")  # Debugging statement


# Process the user's query (NLP model analysis)
def process_query(query):
    # Ensure query is a string
    query_str = str(query)
    embeddings = nlp_model(query_str)
    return embeddings[0] if isinstance(embeddings, list) else embeddings


# Function to expand query using NLP (e.g., synonym expansion)
def expand_query(query):
    doc = nlp(str(query))
    expanded_query = query
    for token in doc:
        # Example: If token has synonyms, add them to the query
        if token.has_vector:
            synonyms = get_synonyms(token.text)  # You can define a function for this
            if synonyms and isinstance(synonyms, (list, set)):  # Ensure synonyms is an iterable
                expanded_query += " " + " ".join(synonyms)  # Join synonyms as space-separated string
    return expanded_query


# Function to get synonyms (optional, for now returns an empty list)
def get_synonyms(text):
    return []  # Placeholder for any synonym-fetching logic


# Search movies based on user input
def search_movies(query, top_k=10):
    # Ensure query is a string
    query_str = str(query)

    # Expand and process query
    expanded_query = expand_query(query_str)
    query_embedding = process_query(expanded_query)

    ix = get_index()  # Open the existing index
    seen_movie_ids = set()  # To track already seen movie IDs and names
    results_with_scores = []

    with ix.searcher() as searcher:
        # Parse the query to search in both 'overview' and 'name' fields
        query_parsed = MultifieldParser(["overview", "name"], ix.schema).parse(expanded_query)

        # Fetch scored results, sorted by score in descending order (default behavior)
        results = searcher.search(query_parsed, limit=top_k)  # Limit to top_k results upfront

        # Check if any results were found
        if results:
            for result in results:
                # Add the movie to the result set and ensure we are not showing duplicates by ID or name
                movie_id = result['movie_id']
                movie_name = result['name']

                # Check if movie_id or movie_name has been seen
                if movie_id not in seen_movie_ids and movie_name not in seen_movie_ids:
                    seen_movie_ids.add(movie_id)
                    seen_movie_ids.add(movie_name)  # Also track the movie name to avoid name-based duplicates
                    results_with_scores.append((movie_id, movie_name, result.score))

        else:
            print("No results found for the query.")

    return results_with_scores  # Return movie ID, name, and score for each result


# Main execution
if __name__ == "__main__":
    # update_index()  # Uncomment this to update the index with the latest data
    query = "Dog"  # Example query
    results = search_movies(query)  # Search for movies matching the query

    if results:
        for movie_id, movie_name, score in results:
            print(f"Movie ID: {movie_id}, Movie Name: {movie_name}, Score: {score}")
    else:
        print("No movies found.")