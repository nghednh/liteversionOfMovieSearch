from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from transformers import pipeline
from pymongo import MongoClient
import os

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
schema = Schema(movie_id=TEXT(stored=True), name=TEXT(stored=True), overview=TEXT(stored=True), subtitle=TEXT(stored=True))

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
    for movie in collection.find().limit(10):  # Adjust limit as necessary
        writer.update_document(  # Use update_document to modify existing entries
            movie_id=str(movie['movie_id']),
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

# Search movies based on user input
def search_movies(query):
    query_embedding = process_query(query)
    ix = get_index()  # Open the existing index
    with ix.searcher() as searcher:
        query_parsed = QueryParser("overview", ix.schema).parse(query)  # Searching in the 'overview' field
        results = searcher.search(query_parsed)
        if results:
            for result in results:
                print(f"Movie found: {result['name']}, Movie ID: {result['movie_id']}")  # Debugging statement
        else:
            print("No results found for the query.")

# Run index update and search example
update_index()  # Update the index with new/modified movie data
search_movies("Red one")  # Example search query
