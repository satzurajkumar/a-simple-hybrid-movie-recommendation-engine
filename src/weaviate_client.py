# src/weaviate_client.py
import weaviate
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional

load_dotenv()
logging.basicConfig(level=logging.INFO)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
# WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") # Uncomment if using API Key

if not WEAVIATE_URL:
    raise ValueError("WEAVIATE_URL environment variable not set")

# auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
# client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config) # Adjust auth method if needed
client = weaviate.Client(url=WEAVIATE_URL) # Simple client without auth

MOVIE_CLASS_SCHEMA = {
    "class": "Movie",
    "description": "Represents a movie with its content vector",
    "vectorizer": "none", # We provide vectors explicitly
    "properties": [
        {
            "name": "movie_id",
            "dataType": ["int"],
            "description": "The ID of the movie",
        },
        {
            "name": "title",
            "dataType": ["text"],
            "description": "The title of the movie",
        },
        {
             "name": "genres",
             "dataType": ["text"], # Store genres as text for filtering/readability
             "description": "Genres of the movie (comma-separated)",
        }
    ],
}

def get_weaviate_client():
    # Basic check if client is connected
    try:
        if not client.is_ready():
            logging.error("Weaviate client is not ready.")
            # Optional: Add reconnection logic here
            return None
        return client
    except Exception as e:
        logging.error(f"Error connecting to Weaviate: {e}")
        return None


def create_weaviate_schema():
    wv_client = get_weaviate_client()
    if not wv_client: return

    try:
        current_schema = wv_client.schema.get()
        class_exists = any(cls['class'] == 'Movie' for cls in current_schema.get('classes', []))

        if not class_exists:
            wv_client.schema.create_class(MOVIE_CLASS_SCHEMA)
            logging.info("Weaviate 'Movie' class created.")
        else:
            logging.info("Weaviate 'Movie' class already exists.")
    except Exception as e:
        logging.error(f"Error creating Weaviate schema: {e}")


def add_movie_vector(movie_id: int, title: str, genres: str, vector: List[float]):
    wv_client = get_weaviate_client()
    if not wv_client: return None

    movie_object = {
        "movie_id": movie_id,
        "title": title,
        "genres": genres,
    }

    try:
        result = wv_client.data_object.create(
            data_object=movie_object,
            class_name="Movie",
            vector=vector
            # uuid=weaviate.util.generate_uuid5(movie_id) # Optional: Use deterministic UUID
        )
        logging.debug(f"Added movie {movie_id} to Weaviate: {result}")
        return result
    except Exception as e:
        logging.error(f"Error adding movie {movie_id} to Weaviate: {e}")
        return None

def find_similar_movies(vector: List[float], limit: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
    wv_client = get_weaviate_client()
    if not wv_client: return []

    near_vector = {"vector": vector}

    # Build the 'where' filter to exclude specific movie IDs
    where_filter = None
    if exclude_ids:
        operands = [{"path": ["movie_id"], "operator": "NotEqual", "valueInt": mid} for mid in exclude_ids]
        if len(operands) == 1:
             where_filter = operands[0]
        elif len(operands) > 1:
             where_filter = {
                 "operator": "And",
                 "operands": operands
             }


    try:
        result = (
            wv_client.query
            .get("Movie", ["movie_id", "title", "genres"])
            .with_near_vector(near_vector)
            .with_limit(limit)
            .with_additional(["distance", "id"]) # Use distance for similarity score
            .with_where(where_filter) # Apply the filter here
            .do()
        )
        movies = result.get('data', {}).get('Get', {}).get('Movie', [])
        # logging.info(f"Found similar movies: {movies}")
        # Convert Weaviate distance (squared Euclidean) to cosine similarity approximation if needed,
        # or just use distance (lower is better)
        # For TF-IDF, cosine similarity is more common, but vector search often uses L2 distance.
        # Let's return movie_id and its distance for ranking.
        return [{"movie_id": movie["movie_id"], "distance": movie["_additional"]["distance"], "title": movie["title"], "genres": movie["genres"]}
                for movie in movies]

    except Exception as e:
        logging.error(f"Error finding similar movies in Weaviate: {e}")
        return []

# Ensure schema exists on startup (can be called from main.py or load_data.py)
# create_weaviate_schema() # Call this appropriately