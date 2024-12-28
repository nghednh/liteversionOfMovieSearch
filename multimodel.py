from pymongo import MongoClient
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
import faiss
from transformers import pipeline
from image_search import load_index, get_movie_id_and_score
from search import search_movies

# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://nghednh:123@cluster0.llhn1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['movie_db']  # Database name
collection = db['movies']  # Collection name

# Load the BERT-based NLP model for query analysis
nlp_model = pipeline('feature-extraction', model='bert-base-uncased', framework='pt')

# Function to perform multimodal search
def multimodal_search(query_text, query_img_url, top_k=10):
    # Load the FAISS image index and image URLs
    image_index, image_urls = load_index()
    if image_index is None or image_urls is None:
        print("Image index or image URLs not found.")
        return []

    # If both text and image queries are None or empty, return an empty result
    if not query_text and not query_img_url:
        print("Both text and image queries are empty.")
        return []

    # Initialize result placeholders
    text_results = []
    image_results = []

    # Perform text-based search (using Whoosh) if query_text is provided
    if query_text:
        text_results = search_movies(query_text, top_k)
        print("\n--- Text-based search results ---")
        for movie_id, movie_name, score in text_results:
            print(f"Movie Name: {movie_name}, Score: {score}")

    # Perform image-based search (using FAISS) if query_img_url is provided
    if query_img_url:
        print(f"\nPerforming image search for query: {query_img_url}")
        image_results = get_movie_id_and_score(query_img_url, image_index, image_urls, top_k)
        print("\n--- Image-based search results ---")
        for movie_id, movie_name, score in image_results:
            print(f"Movie Name: {movie_name}, Score: {score}")

    # Combine the results with improved scoring logic
    combined_results = {}

    # Process text search results
    print("\n--- Processing Text Results ---")
    for movie_id, movie_name, score in text_results:
        combined_results[movie_name.lower()] = {
            "movie_id": movie_id,
            "name": movie_name,
            "text_score": score,
            "image_score": 0,
            "score": score * 0.6  # Apply text weight
        }
        print(f"Added {movie_name} from text search with score {score}")

    # Process image search results
    print("\n--- Processing Image Results ---")
    for movie_id, movie_name, score in image_results:
        movie_key = movie_name.lower()
        if movie_key in combined_results:
            # If movie exists in text results, combine scores
            combined_results[movie_key]["image_score"] = score
            text_score = combined_results[movie_key]["text_score"]
            combined_results[movie_key]["score"] = (text_score * 0.6*100) + (score * 0.4)
            print(f"Updated {movie_name} with combined text and image scores")
        else:
            combined_results[movie_key] = {
                "movie_id": movie_id,
                "name": movie_name,
                "text_score": 0,
                "image_score": score,
                "score": score * 0.4  # Apply image weight
            }
            print(f"Added {movie_name} from image search with score {score}")

    # Sort results by the combined score
    sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)

    # Format the results for output
    final_results = []
    print(f"\n--- Combined Results (Top {top_k}) ---")
    for details in sorted_results[:top_k]:
        final_results.append({
            "movie_id": details["movie_id"],
            "movie_name": details["name"],
            "score": details["score"],
            "text_score": details["text_score"],
            "image_score": details["image_score"]
        })
        print(f"Movie Name: {details['name']}")
        print(f"- Text Score: {details['text_score']:.3f}")
        print(f"- Image Score: {details['image_score']:.3f}")
        print(f"- Final Score: {details['score']:.3f}\n")

    return final_results

# Example of running the multimodal search
if __name__ == "__main__":
    query_text = "Moana"  # Example text query
    query_img_url = "https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSHUKJVY_6ELobIKty_QL81r4ZRKjFTQKlFoq_G_rEkajxdUF92X7KuF-e8uBtQzU9PTjC4"

    # Perform multimodal search
    results = multimodal_search(query_text, query_img_url, top_k=10)