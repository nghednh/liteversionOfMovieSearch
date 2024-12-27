import requests
from pymongo import MongoClient

# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://nghednh:123@cluster0.llhn1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client['movie_db']  # Database name
collection = db['movies']  # Collection name

API_KEY = '50767d9d58baff413d3f12803125ce4d'  # Replace with your TMDb API key
BASE_URL = "https://api.themoviedb.org/3/movie/"


# Function to get movie data from TMDb API
def get_movie_data(movie_id):
    print(f"Fetching data for movie ID: {movie_id}")  # Debugging statement
    url = f"{BASE_URL}{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()

    if 'status_code' in response and response['status_code'] == 34:
        print(f"Movie ID {movie_id} not found in TMDb.")  # Debugging statement
        return None  # Movie not found

    movie_data = {
        'movie_id': movie_id,
        'name': response.get('original_title'),
        'overview': response.get('overview'),
        'subtitle': response.get('tagline'),
        'images': get_movie_images(movie_id)
    }
    return movie_data


# Function to get all images for a movie
def get_movie_images(movie_id):
    print(f"Fetching images for movie ID: {movie_id}")  # Debugging statement
    url = f"{BASE_URL}{movie_id}/images?api_key={API_KEY}"
    response = requests.get(url).json()

    images = []
    for img in response.get('posters', []):
        images.append(f"https://image.tmdb.org/t/p/w500{img['file_path']}")
    return images


# Function to crawl a list of movie IDs and store in MongoDB Atlas
def crawl_movies(movie_ids):
    print("Starting to crawl movies.")  # Debugging statement
    for movie_id in movie_ids:
        print(f"Crawling movie ID: {movie_id}")  # Debugging statement
        movie_data = get_movie_data(movie_id)
        if movie_data:
            # Store the movie data into MongoDB
            collection.update_one({'movie_id': movie_id}, {'$set': movie_data}, upsert=True)

            # After storing the movie data, you can check if the movie exists in the database.
            movie_in_db = collection.find_one({'movie_id': movie_id})
            if movie_in_db:
                print(f"Movie data for ID {movie_id} successfully saved to DB: {movie_in_db}")
            else:
                print(f"Failed to save movie data for ID {movie_id} to DB.")
        else:
            print(f"Skipping movie ID {movie_id}.")  # Debugging statement


# Function to print all movies stored in the MongoDB collection
def print_movies_from_db():
    print("Movies stored in the database:")
    # Fetch all movies from the 'movies' collection
    movies = collection.find()  # Find all documents in the collection
    for movie in movies:
        # Print the movie details (you can customize the fields to display)
        print(f"Movie ID: {movie.get('movie_id')}")
        print(f"Name: {movie.get('name')}")
        print(f"Overview: {movie.get('overview')}")
        print(f"Subtitle: {movie.get('subtitle')}")
        #print(f"Images: {movie.get('images')}")
        print("-" * 50)  # Separator between movies


# Function to get popular movies in a specified language between page_x and page_y
def get_popular_movies(language, page_x, page_y):
    print(f"Fetching popular movies in {language} between pages {page_x} and {page_y}.")

    for page in range(page_x, page_y + 1):
        print(f"Fetching page {page} for language {language}...")  # Debugging statement

        # Build the URL for popular movies with pagination
        url = (
            f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}"
            f"&language={language}&page={page}"
        )

        response = requests.get(url).json()

        # Check for errors in the API response
        if 'status_code' in response and response['status_code'] != 200:
            print(f"Error fetching movies on page {page}. Response: {response}")
            break

        # Extract movie IDs and process each movie
        movie_ids = [movie['id'] for movie in response.get('results', [])]

        if not movie_ids:  # No movies found on the current page
            print(f"No movies found on page {page} for language {language}.")
            break

        # Crawl and save movie data for the current page
        crawl_movies(movie_ids)

    print(f"Completed fetching popular movies in {language} from pages {page_x} to {page_y}.")



# page_start = 2
# page_end = 2
#
# # Fetch popular movies in English
# get_popular_movies(language="en-US", page_x=page_start, page_y=page_end)
#
# # Fetch popular movies in Vietnamese
# get_popular_movies(language="vi-VN", page_x=page_start, page_y=page_end)
#
# print_movies_from_db()
# print("end")
document_count = collection.count_documents({})

print(f"Number of documents in the 'movies' collection: {document_count}")



