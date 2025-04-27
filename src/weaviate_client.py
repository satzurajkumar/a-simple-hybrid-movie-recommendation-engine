# src/weaviate_client.py
import weaviate # Main import
import weaviate.classes as wvc # Import classes for v4 features like properties, config
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

# Global variable to hold the client instance once connected
_client: Optional[weaviate.WeaviateClient] = None

def get_weaviate_client() -> Optional[weaviate.WeaviateClient]:
    """Connects to Weaviate using v4 syntax and returns the client instance."""
    global _client
    # Check if client exists and is connected
    # Note: is_connected() might not be sufficient for long-running apps,
    # consider adding a basic readiness check like client.is_ready() if needed.
    if _client is not None and _client.is_connected():
        # logging.debug("Returning existing Weaviate client.")
        return _client

    logging.info(f"Attempting to connect to Weaviate at {WEAVIATE_URL}...")
    try:
        # --- v4 Connection ---
        parts = WEAVIATE_URL.split(':')
        if len(parts) < 3:
             raise ValueError(f"Invalid WEAVIATE_URL format: {WEAVIATE_URL}. Expected http(s)://host:port")

        http_host = parts[1].strip('/') # e.g., localhost
        http_port = int(parts[2].strip('/')) # e.g., 8080
        # Default gRPC port assumption (adjust if necessary)
        grpc_port = 50051 # Default gRPC port

        # --- Authentication (Optional) ---
        auth_config = None
        # WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
        # if WEAVIATE_API_KEY:
        #     auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
        #     logging.info("Using API Key authentication for Weaviate.")

        _client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=WEAVIATE_URL.startswith("https"),
            grpc_host=http_host,
            grpc_port=grpc_port,
            grpc_secure=WEAVIATE_URL.startswith("https"),
            auth_client_secret=auth_config, # Pass auth config here
            skip_init_checks=False
        )
        _client.connect() # Explicitly connect

        if _client.is_ready():
            logging.info("Successfully connected to Weaviate (v4).")
            return _client
        else:
            logging.error("Weaviate client connected but is not ready.")
            if _client.is_connected(): # Attempt to close if connected but not ready
                 _client.close()
            _client = None
            return None

    except Exception as e:
        logging.error(f"Error connecting to Weaviate (v4): {e}")
        if _client and _client.is_connected(): # Clean up connection if partially established
            _client.close()
        _client = None
        return None

# --- Added function to close the connection ---
def close_weaviate_connection():
    """Closes the global Weaviate client connection if it exists and is connected."""
    global _client
    if _client is not None and _client.is_connected():
        logging.info("Closing Weaviate client connection...")
        try:
            _client.close()
            _client = None # Reset global variable
            logging.info("Weaviate client connection closed.")
        except Exception as e:
            logging.error(f"Error closing Weaviate connection: {e}")
    elif _client is not None:
         # Client exists but is not connected, just reset the variable
         _client = None
    # else:
    #     logging.debug("No active Weaviate connection to close.")
# --- End added function ---


def create_weaviate_schema():
    """Creates the Weaviate schema using v4 syntax."""
    wv_client = get_weaviate_client()
    if not wv_client:
        logging.error("Cannot create schema, Weaviate client not available.")
        return

    collection_name = "Movie" # Use Title Case for class names (convention)

    try:
        if wv_client.collections.exists(collection_name):
            logging.info(f"Weaviate collection '{collection_name}' already exists.")
            return

        properties = [
            wvc.config.Property(name="movie_id", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="genres", data_type=wvc.config.DataType.TEXT),
        ]
        vectorizer_config = wvc.config.Configure.Vectorizer.none()

        logging.info(f"Creating Weaviate collection '{collection_name}'...")
        wv_client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=vectorizer_config
        )
        logging.info(f"Collection '{collection_name}' created.")

    except Exception as e:
        logging.error(f"Error creating Weaviate schema (v4): {e}")


def add_movie_vector(movie_id: int, title: str, genres: str, vector: List[float]):
    """Adds a single movie vector using v4 syntax."""
    wv_client = get_weaviate_client()
    if not wv_client:
        logging.error("Cannot add vector, Weaviate client not available.")
        return None

    collection_name = "Movie"
    try:
        movie_collection = wv_client.collections.get(collection_name)
        properties = { "movie_id": movie_id, "title": title, "genres": genres }
        uuid_returned = movie_collection.data.insert(properties=properties, vector=vector)
        logging.debug(f"Added movie {movie_id} to Weaviate (v4). UUID: {uuid_returned}")
        return uuid_returned
    except Exception as e:
        logging.error(f"Error adding movie {movie_id} to Weaviate (v4): {e}")
        return None

def find_similar_movies(vector: List[float], limit: int = 10, exclude_ids: List[int] = None) -> List[Dict]:
    """Finds similar movies using v4 query syntax."""
    wv_client = get_weaviate_client()
    if not wv_client:
        logging.error("Cannot find similar movies, Weaviate client not available.")
        return []

    collection_name = "Movie"
    try:
        movie_collection = wv_client.collections.get(collection_name)
        query_filter = None
        if exclude_ids:
            id_filters = [wvc.query.Filter.by_property("movie_id").not_equal(mid) for mid in exclude_ids]
            if len(id_filters) == 1: query_filter = id_filters[0]
            elif len(id_filters) > 1: query_filter = wvc.query.Filter.all_of(id_filters)
            logging.debug(f"Weaviate filter generated: {query_filter}")

        response = movie_collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=query_filter,
            return_metadata=wvc.query.MetadataQuery(distance=True),
            return_properties=["movie_id", "title", "genres"]
        )

        similar_movies = []
        for obj in response.objects:
            similar_movies.append({
                "movie_id": obj.properties.get("movie_id"),
                "title": obj.properties.get("title"),
                "genres": obj.properties.get("genres"),
                "distance": obj.metadata.distance if obj.metadata else None,
            })
        logging.debug(f"Weaviate v4 search found {len(similar_movies)} similar movies.")
        return similar_movies
    except Exception as e:
        logging.error(f"Error finding similar movies in Weaviate (v4): {e}")
        import traceback
        traceback.print_exc()
        return []

