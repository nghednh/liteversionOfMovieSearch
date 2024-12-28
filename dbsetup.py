import pymongo
from pymongo import MongoClient, ASCENDING, TEXT

# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://nghednh:123@cluster0.llhn1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['movie_db']
collection = db['movies']


# Create indexes on the movie_id, name, and overview fields
def create_indexes():
    try:
        # First, let's check if there's an existing index and drop the ones we don't need
        indexes = list(collection.list_indexes())

        # Drop the "name_text" index if it exists
        # for index in indexes:
        #     if index.get("name") == "name_text":
        #         collection.drop_index("name_text")
        #
        # # Drop the "overview_text" index if it exists (in case we created one previously)
        # for index in indexes:
        #     if index.get("name") == "overview_text":
        #         collection.drop_index("overview_text")

        # Create a compound text index on both name and overview fields
        collection.create_index([("name", pymongo.TEXT), ("overview", pymongo.TEXT)])  # Compound text index

        # Create an index on `movie_id` for faster lookups
        collection.create_index([("movie_id", pymongo.ASCENDING)])  # Index on `movie_id`

        print("Indexes created successfully!")
    except Exception as e:
        print(f"Error creating indexes: {e}")


# Run the index creation function
create_indexes()
indexes = collection.list_indexes()
for index in indexes:
    print(index)
