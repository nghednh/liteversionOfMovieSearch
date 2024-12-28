import os
import faiss
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from pymongo import MongoClient
import pickle
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from collections import Counter  # To implement voting mechanism
from search import search_movies  # Ensure search_movies function is correctly imported

# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://nghednh:123@cluster0.llhn1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['movie_db']  # Database name
collection = db['movies']  # Collection name

# Initialize the ResNet50 model for image feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Path to save the FAISS index and image URLs inside the image_index folder
IMAGE_INDEX_FOLDER = "image_index"
INDEX_FILE = os.path.join(IMAGE_INDEX_FOLDER, "movie_image_index.index")
IMAGE_URLS_FILE = os.path.join(IMAGE_INDEX_FOLDER, "image_urls.pkl")

# Ensure the image_index folder exists
if not os.path.exists(IMAGE_INDEX_FOLDER):
    os.makedirs(IMAGE_INDEX_FOLDER)

# Function to extract image embedding using ResNet50
def extract_image_embedding(img_url):
    # print(f"Extracting embedding for image URL: {img_url}")  # Debugging statement
    try:
        response = requests.get(img_url)
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))  # BytesIO used here
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        embedding = model.predict(img_array)
        return embedding.flatten()
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return np.zeros((model.output_shape[-1],))  # Return a zero vector if error occurs


# Function to process images concurrently using ThreadPoolExecutor
def process_images_concurrently(image_urls):
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers for better concurrency
        embeddings = list(executor.map(extract_image_embedding, image_urls))
    return np.array(embeddings)


# Function to process images in batches and extract embeddings
def batch_process_images(image_urls, batch_size=10):
    all_embeddings = []
    # Process images in batches to increase efficiency
    for i in range(0, len(image_urls), batch_size):
        batch = image_urls[i:i + batch_size]
        batch_embeddings = process_images_concurrently(batch)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)


# Index movie posters in FAISS and save to disk
def index_images():
    # print("Indexing movie images.")  # Debugging statement
    image_embeddings = []
    image_urls = []

    # Load existing index if it exists
    index, existing_image_urls = load_index()

    if index is None or existing_image_urls is None:
        print("No existing index or image URLs, starting fresh.")

    # Set of existing image URLs for comparison
    existing_image_urls_set = set(existing_image_urls) if existing_image_urls else set()

    # Loop through movies and extract image URLs
    for movie in collection.find().limit(15):
        for img_url in movie['images']:
            if img_url not in existing_image_urls_set:
                image_urls.append(img_url)

    # If no new images, return without doing anything
    if not image_urls:
        # print("No new images to index.")
        return

    # Use concurrent processing to extract embeddings for new images (Batching added here)
    image_embeddings = batch_process_images(image_urls)  # Batching and concurrency here

    # If the index is empty, create a new one
    if index is None:
        index = faiss.IndexFlatL2(image_embeddings.shape[1])  # L2 distance for similarity

    # Add the new embeddings to the existing index
    index.add(image_embeddings)

    # Save the updated FAISS index to a file
    faiss.write_index(index, INDEX_FILE)

    all_image_urls = (existing_image_urls if existing_image_urls else []) + image_urls

    with open(IMAGE_URLS_FILE, "wb") as f:
        pickle.dump(all_image_urls, f)

    print("Image indexing complete.")  # Debugging statement


# Load the FAISS index and image URLs from disk
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(IMAGE_URLS_FILE):
        return None, None

    # print("Loading FAISS index and image URLs.")

    # Load FAISS index from file
    index = faiss.read_index(INDEX_FILE)

    # Load image URLs from file
    with open(IMAGE_URLS_FILE, "rb") as f:
        image_urls = pickle.load(f)

    # print("Index loaded.")
    return index, image_urls


# Search for similar images based on a query image
def search_similar_image(query_img_url, index, image_urls):
    # print(f"Searching for similar image to: {query_img_url}")  # Debugging statement
    query_embedding = extract_image_embedding(query_img_url)
    query_embedding = np.array([query_embedding])

    D, I = index.search(query_embedding, k=3)  # k=3 returns top 3 most similar images
    return I[0], D[0]  # Return the indices and distances of the most similar images


# Function to retrieve movie documents based on image URLs
def get_movies_from_image_urls(image_urls):
    movie_docs = []
    for img_url in image_urls:
        movie = collection.find_one({"images": img_url})  # Find the movie that contains the image
        if movie:
            movie_docs.append(movie)
    return movie_docs


# Function to get movie ID and score from image similarity search

def get_movie_id_and_score(query_img_url, index, image_urls, top_k=10):
    # Perform search for similar images (this will return numpy arrays or lists)
    similar_image_indices, distances = search_similar_image(query_img_url, index, image_urls)

    # Check if the results are empty (either similar_image_indices or distances are empty arrays)
    if similar_image_indices.size == 0 or distances.size == 0:  # Ensure .size is used for numpy arrays
        print("No similar images found.")
        return []

    # Retrieve the movie documents corresponding to the returned image URLs
    similar_image_urls = [image_urls[idx] for idx in similar_image_indices]
    similar_movies = get_movies_from_image_urls(similar_image_urls)

    # Use a set to track seen movie IDs and names to avoid duplicates
    seen_movie_ids = set()
    seen_movie_names = set()
    movie_scores = []

    # Loop over the movies and their distances
    for movie, distance in zip(similar_movies, distances):
        movie_id = movie.get('movie_id', 'Unknown ID')  # Fetch the movie_id field (not _id)
        movie_name = movie.get('name', 'Unknown')  # Fetch the movie name, default to 'Unknown' if not found

        # Check if the movie has already been added based on movie_id or movie_name
        if movie_id not in seen_movie_ids and movie_name not in seen_movie_names:
            # If the movie is not a duplicate, add it to the results and track it
            seen_movie_ids.add(movie_id)
            seen_movie_names.add(movie_name)
            movie_scores.append((movie_id, movie_name, distance))

        # Stop once we have collected the top_k results
        if len(movie_scores) >= top_k:
            break

    return movie_scores  # Return movie ID, name, and score for each result



# Main workflow (to display movie name, ID, and score)
if __name__ == "__main__":

    query_img_url = "https://i.ytimg.com/vi/m6MF1MqsDhc/maxresdefault.jpg"

    # Check if reindexing is needed (i.e., if the index files already exist)
    index_images()

    # Load the pre-built index (either after reindexing or from existing index)
    index, image_urls = load_index()

    # Ensure that the index and image URLs are not None before proceeding
    if index is None or image_urls is None:
        print("Error: Index or image URLs could not be loaded. Exiting.")
    else:
        # Get the movie ID, name, and score from the image similarity search
        movie_id_and_score = get_movie_id_and_score(query_img_url, index, image_urls)

        # Print out movie ID, name, and score
        for movie_id, movie_name, score in movie_id_and_score:
            print(f"Movie ID: {movie_id}, Movie Name: {movie_name}, Score: {score}")
