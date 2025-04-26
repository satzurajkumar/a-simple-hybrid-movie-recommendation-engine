# load_data.py
import pandas as pd
from sqlalchemy.orm import Session
from src.database import SessionLocal, engine, Base
from src import models, crud, schemas
from src.weaviate_client import create_weaviate_schema, add_movie_vector, get_weaviate_client
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import time # For timestamp in rating

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

def load_users(db: Session):
    logging.info("Loading users...")
    try:
        # User columns: user_id | age | gender | occupation | zip_code
        users_df = pd.read_csv(USER_FILE, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], encoding='latin-1')
        for _, row in users_df.iterrows():
            user_data = schemas.UserCreate(
                user_id=row['user_id'],
                age=row['age'],
                gender=row['gender'],
                occupation=row['occupation'],
                zip_code=row['zip_code']
            )
            # Check if user already exists
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
    wv_client = get_weaviate_client()
    if not wv_client:
        logging.error("Weaviate client not available. Skipping vector loading.")
        # Optionally load movies without vectors
        # return


    create_weaviate_schema() # Ensure schema exists

    try:
        # Item columns: movie_id | movie_title | release_date | video_release_date | IMDb_URL | genres... (19 columns)
        item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRES
        movies_df = pd.read_csv(ITEM_FILE, sep='|', names=item_cols, encoding='latin-1', index_col='movie_id')

        # Prepare genre strings and data for TF-IDF
        movie_genres_text = []
        movie_ids_for_tfidf = []
        movies_to_db = []

        for movie_id, row in movies_df.iterrows():
            genre_list = [GENRES[i] for i, val in enumerate(row[GENRES]) if val == 1]
            genre_str = ", ".join(genre_list) if genre_list else "Unknown" # Use comma separation for readability
            movie_genres_text.append(" ".join(genre_list).lower()) # Use space separation for TF-IDF
            movie_ids_for_tfidf.append(movie_id)


            # Prepare data for SQL database
            movie_data = schemas.MovieCreate(
                movie_id=movie_id,
                title=row['title'].rsplit(' (', 1)[0], # Clean title '(YYYY)' part if needed
                release_date=row['release_date'],
                imdb_url=row['imdb_url'],
                genres=genre_str # Store readable genres in SQL
            )
            movies_to_db.append(movie_data)


            # --- Add Movies to SQL DB ---
            # Check if movie exists before adding
            db_movie = crud.get_movie(db, movie_id=movie_id)
            if not db_movie:
                 crud.create_movie(db=db, movie=movie_data)


        # --- Generate TF-IDF Vectors for Genres ---
        logging.info("Calculating TF-IDF vectors for genres...")
        if movie_genres_text:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100) # Limit features
            tfidf_matrix = vectorizer.fit_transform(movie_genres_text)
            tfidf_vectors = tfidf_matrix.toarray().tolist() # Convert to list for Weaviate
            logging.info(f"Generated TF-IDF vectors with shape: {tfidf_matrix.shape}")


            # --- Add Movies and Vectors to Weaviate ---
            logging.info("Adding movies and vectors to Weaviate...")
            # Consider batching for large datasets
            with wv_client.batch as batch:
                 batch.batch_size=100 # Configure batch size
                 for i, movie_id in enumerate(movie_ids_for_tfidf):
                      # Find the corresponding movie data (could optimize this lookup)
                      movie_sql_data = next((m for m in movies_to_db if m.movie_id == movie_id), None)
                      if movie_sql_data and i < len(tfidf_vectors):
                           movie_object = {
                               "movie_id": movie_id,
                               "title": movie_sql_data.title,
                               "genres": movie_sql_data.genres, # Use readable genres here too
                           }
                           vector = tfidf_vectors[i]
                           # Normalize vector (optional but good practice for cosine sim)
                           norm = np.linalg.norm(vector)
                           normalized_vector = (vector / norm).tolist() if norm > 0 else vector

                           batch.add_data_object(
                               data_object=movie_object,
                               class_name="Movie",
                               vector=normalized_vector # Store normalized vector
                           )
                      else:
                         logging.warning(f"Could not find movie data or vector for movie_id {movie_id} at index {i}")

            logging.info(f"Added/Updated {len(movie_ids_for_tfidf)} movies in Weaviate.")

        else:
             logging.warning("No genre text found to generate TF-IDF vectors.")


        logging.info(f"Loaded/Updated {len(movies_to_db)} movies in SQL database.")

    except FileNotFoundError:
        logging.error(f"Item file not found: {ITEM_FILE}")
    except Exception as e:
        logging.error(f"Error loading movies: {e}")
        import traceback
        traceback.print_exc()


def load_ratings(db: Session):
    logging.info("Loading ratings...")
    try:
        # Data columns: user_id | item_id | rating | timestamp
        ratings_df = pd.read_csv(DATA_FILE, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

        # --- Bulk insert ratings (more efficient) ---
        ratings_to_insert = []
        movie_rating_updates = {} # movie_id -> {'total_rating': x, 'count': y}

        for _, row in ratings_df.iterrows():
             # Check if user and movie exist (optional, assumes users/movies loaded first)
            # user_exists = crud.get_user(db, row['user_id']) is not None
            # movie_exists = crud.get_movie(db, row['movie_id']) is not None
            # if not user_exists or not movie_exists:
            #     logging.warning(f"Skipping rating for non-existent user {row['user_id']} or movie {row['movie_id']}")
            #     continue

            ratings_to_insert.append(
                models.Rating(
                    user_id=int(row['user_id']),
                    movie_id=int(row['movie_id']),
                    rating=int(row['rating']),
                    timestamp=int(row['timestamp'])
                )
            )
            # Aggregate ratings for average calculation
            mid = int(row['movie_id'])
            if mid not in movie_rating_updates:
                movie_rating_updates[mid] = {'total_rating': 0, 'count': 0}
            movie_rating_updates[mid]['total_rating'] += int(row['rating'])
            movie_rating_updates[mid]['count'] += 1


        if ratings_to_insert:
            # Use bulk_save_objects for efficiency
            db.bulk_save_objects(ratings_to_insert)
            db.commit()
            logging.info(f"Loaded {len(ratings_to_insert)} ratings via bulk insert.")
        else:
            logging.info("No ratings found to load.")


        # --- Update movie average ratings and counts ---
        logging.info("Updating movie average ratings and counts...")
        movies_to_update = []
        for movie_id, data in movie_rating_updates.items():
             movie = crud.get_movie(db, movie_id)
             if movie:
                 movie.avg_rating = data['total_rating'] / data['count'] if data['count'] > 0 else 0.0
                 movie.rating_count = data['count']
                 movies_to_update.append(movie)


        if movies_to_update:
             # Use bulk_update_mappings or iterate and merge/add
             # Using merge might be safer across different sessions/states
             for movie in movies_to_update:
                 db.merge(movie) # Updates existing or inserts if somehow missed
             db.commit()
             logging.info(f"Updated average ratings for {len(movies_to_update)} movies.")


    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_FILE}")
    except Exception as e:
        logging.error(f"Error loading ratings: {e}")
        db.rollback() # Rollback on error during rating load or update


def main():
    logging.info("Initializing database...")
    Base.metadata.create_all(bind=engine) # Create tables if they don't exist
    db = SessionLocal()
    try:
        load_users(db)
        load_movies_and_vectors(db) # Loads movies to SQL and Weaviate
        load_ratings(db) # Loads ratings and updates movie averages
        logging.info("Data loading complete.")
    finally:
        db.close()

if __name__ == "__main__":
    main()