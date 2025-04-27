# load_data.py
import pandas as pd
from sqlalchemy.orm import Session
from src.database import SessionLocal, engine, Base
from src import models, crud, schemas
# Import Weaviate client and the new close function
from src.weaviate_client import (
    create_weaviate_schema,
    add_movie_vector,
    get_weaviate_client,
    close_weaviate_connection # <-- Import close function
)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import time
import os # Added for path joining if needed

logging.basicConfig(level=logging.INFO)

DATA_DIR = 'data'
USER_FILE = f'{DATA_DIR}/u.user'
ITEM_FILE = f'{DATA_DIR}/u.item'
DATA_FILE = f'{DATA_DIR}/u.data'

# Genre mapping from MovieLens 100k README
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# --- TF-IDF Vectorizer ---
# Initialize here, fit later
vectorizer = TfidfVectorizer(stop_words='english', max_features=100) # Limit features

def load_users(db: Session):
    # (No changes needed in this function)
    logging.info("Loading users...")
    try:
        users_df = pd.read_csv(USER_FILE, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], encoding='latin-1')
        for _, row in users_df.iterrows():
            user_data = schemas.UserCreate(
                user_id=row['user_id'],
                age=row['age'],
                gender=row['gender'],
                occupation=row['occupation'],
                zip_code=row['zip_code']
            )
            db_user = crud.get_user(db, user_id=row['user_id'])
            if not db_user:
                crud.create_user(db=db, user=user_data)
        logging.info(f"Loaded {len(users_df)} users.")
    except FileNotFoundError:
        logging.error(f"User file not found: {USER_FILE}")
    except Exception as e:
        logging.error(f"Error loading users: {e}")


def load_movies_and_vectors(db: Session):
    logging.info("Loading movies and generating vectors...")
    wv_client = get_weaviate_client() # Establish connection early
    if not wv_client:
        logging.error("Weaviate client not available. Cannot load vectors.")
        # Decide if you want to proceed loading movies to SQL only
        # return # Or raise an error

    create_weaviate_schema() # Ensure schema exists

    try:
        item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRES
        movies_df = pd.read_csv(ITEM_FILE, sep='|', names=item_cols, encoding='latin-1', index_col='movie_id')

        movie_genres_text_list = [] # For fitting TF-IDF
        movies_data_for_sql_and_weaviate = [] # Store processed data

        logging.info("Processing movie data...")
        for movie_id, row in movies_df.iterrows():
            genre_list = [GENRES[i] for i, val in enumerate(row[GENRES]) if val == 1]
            genre_str_readable = ", ".join(genre_list) if genre_list else "Unknown"
            genre_str_for_tfidf = " ".join(genre_list).lower() # Use space separation for TF-IDF
            movie_genres_text_list.append(genre_str_for_tfidf)

            cleaned_title = row['title'].rsplit(' (', 1)[0]

            # Prepare data for SQL
            movie_sql_data = schemas.MovieCreate(
                movie_id=movie_id,
                title=cleaned_title,
                release_date=row['release_date'],
                imdb_url=row['imdb_url'],
                genres=genre_str_readable
            )

            # Store data needed for Weaviate insertion later
            movies_data_for_sql_and_weaviate.append({
                "sql_data": movie_sql_data,
                "title": cleaned_title,
                "genres_readable": genre_str_readable
                # Vector will be added after TF-IDF fitting
            })

        # --- Add Movies to SQL DB ---
        logging.info("Adding/Updating movies in SQL database...")
        added_sql_count = 0
        for movie_entry in movies_data_for_sql_and_weaviate:
            movie_data = movie_entry["sql_data"]
            db_movie = crud.get_movie(db, movie_id=movie_data.movie_id)
            if not db_movie:
                 crud.create_movie(db=db, movie=movie_data)
                 added_sql_count += 1
        logging.info(f"Added {added_sql_count} new movies to SQL database.")

        # --- Generate TF-IDF Vectors ---
        logging.info("Calculating TF-IDF vectors for genres...")
        if movie_genres_text_list:
            tfidf_matrix = vectorizer.fit_transform(movie_genres_text_list)
            tfidf_vectors = tfidf_matrix.toarray().tolist() # Convert to list for Weaviate
            logging.info(f"Generated TF-IDF vectors with shape: {tfidf_matrix.shape}")

            # --- Add Vectors to our movie data list ---
            if len(tfidf_vectors) == len(movies_data_for_sql_and_weaviate):
                for i, movie_entry in enumerate(movies_data_for_sql_and_weaviate):
                    vector = tfidf_vectors[i]
                    # Normalize vector (optional but good practice)
                    norm = np.linalg.norm(vector)
                    normalized_vector = (vector / norm).tolist() if norm > 0 else vector
                    movie_entry["vector"] = normalized_vector
            else:
                 logging.error("Mismatch between number of movies processed and TF-IDF vectors generated!")
                 # Handle error appropriately - skip Weaviate insertion?

        else:
             logging.warning("No genre text found to generate TF-IDF vectors.")

        # --- Add Movies and Vectors to Weaviate ---
        # Consider batching for large datasets if using Weaviate v3 client methods
        # v4 client handles batching internally more efficiently
        logging.info("Adding movies and vectors to Weaviate...")
        added_weaviate_count = 0
        skipped_weaviate_count = 0
        for movie_entry in movies_data_for_sql_and_weaviate:
             if "vector" in movie_entry:
                 result = add_movie_vector(
                     movie_id=movie_entry["sql_data"].movie_id,
                     title=movie_entry["title"],
                     genres=movie_entry["genres_readable"],
                     vector=movie_entry["vector"]
                 )
                 if result:
                     added_weaviate_count += 1
                 else:
                     skipped_weaviate_count += 1
             else:
                 skipped_weaviate_count += 1
                 logging.warning(f"Skipping Weaviate add for movie {movie_entry['sql_data'].movie_id} - no vector found.")

        logging.info(f"Added {added_weaviate_count} movies to Weaviate. Skipped {skipped_weaviate_count}.")

    except FileNotFoundError:
        logging.error(f"Item file not found: {ITEM_FILE}")
    except Exception as e:
        logging.error(f"Error loading movies and vectors: {e}")
        import traceback
        traceback.print_exc()


def load_ratings(db: Session):
    # (No changes needed in this function)
    logging.info("Loading ratings and updating SQL movie stats...")
    try:
        ratings_df = pd.read_csv(DATA_FILE, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        ratings_to_insert = []
        movie_rating_updates = {}

        for _, row in ratings_df.iterrows():
            ratings_to_insert.append(
                models.Rating(
                    user_id=int(row['user_id']),
                    movie_id=int(row['movie_id']),
                    rating=int(row['rating']),
                    timestamp=int(row['timestamp'])
                )
            )
            mid = int(row['movie_id'])
            if mid not in movie_rating_updates:
                movie_rating_updates[mid] = {'total_rating': 0, 'count': 0}
            movie_rating_updates[mid]['total_rating'] += int(row['rating'])
            movie_rating_updates[mid]['count'] += 1

        if ratings_to_insert:
            db.bulk_save_objects(ratings_to_insert)
            db.commit()
            logging.info(f"Loaded {len(ratings_to_insert)} ratings via bulk insert.")
        else:
            logging.info("No ratings found to load.")

        logging.info("Updating movie average ratings and counts in SQL...")
        movies_to_update = []
        for movie_id, data in movie_rating_updates.items():
             movie = crud.get_movie(db, movie_id)
             if movie:
                 movie.avg_rating = data['total_rating'] / data['count'] if data['count'] > 0 else 0.0
                 movie.rating_count = data['count']
                 movies_to_update.append(movie)

        if movies_to_update:
             for movie in movies_to_update:
                 db.merge(movie)
             db.commit()
             logging.info(f"Updated average ratings for {len(movies_to_update)} movies in SQL.")

    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_FILE}")
    except Exception as e:
        logging.error(f"Error loading ratings: {e}")
        db.rollback()


def main():
    logging.info("Initializing database...")
    Base.metadata.create_all(bind=engine) # Create tables if they don't exist
    db = SessionLocal()
    try:
        load_users(db)
        load_movies_and_vectors(db) # Loads movies to SQL and Weaviate
        load_ratings(db) # Loads ratings and updates movie averages
        logging.info("Data loading complete.")
    except Exception as e:
         logging.error(f"An error occurred during data loading: {e}")
         import traceback
         traceback.print_exc()
    finally:
        # --- Ensure connections are closed ---
        if db:
            db.close()
            logging.info("Database session closed.")
        close_weaviate_connection() # <-- Call the close function here
        # --- End ensure connections are closed ---

if __name__ == "__main__":
    main()
