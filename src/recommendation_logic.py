# src/recommendation_logic.py
import pickle
import os
import random
from typing import List, Dict, Optional, Set
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .weaviate_client import find_similar_movies, get_weaviate_client
import logging
from sklearn.feature_extraction.text import TfidfVectorizer # To get vector for CB pivot
import numpy as np # For vector normalization

logging.basicConfig(level=logging.INFO)

MODEL_PATH = 'models/cf_model.pkl'
GENRES = [ # Make sure this matches the order used during TF-IDF generation
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

# --- Load CF Model ---
cf_model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        cf_model = pickle.load(f)
        logging.info("Collaborative Filtering model loaded.")
else:
    logging.warning(f"CF model not found at {MODEL_PATH}. CF recommendations will be disabled.")

# --- TF-IDF Vectorizer (needs to be consistent with load_data.py) ---
# We need this to vectorize genres of liked movies for CB search pivot
# It's better to save/load the fitted vectorizer, but for simplicity, we recreate it here.
# NOTE: This assumes load_data.py ran and populated Weaviate.
# A better approach would be to store vectors alongside SQL data or retrieve vectors
# directly from Weaviate for liked movies if possible.
# This recreation is primarily for getting a vector *representation* of liked genres.

# Initialize a placeholder vectorizer. It won't be fitted here,
# but we need the object structure. Fitting happens implicitly if needed below,
# or ideally, load a saved one. For this example, we rely on vectors in Weaviate.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
# A dummy fit to avoid NotFittedError if transform is called on empty genres
try:
    tfidf_vectorizer.fit(["action adventure", "comedy romance", "drama thriller"]) # Fit with dummy data
except ValueError:
     logging.warning("Could not perform dummy fit on TFIDF Vectorizer")


def get_vector_for_genres(genre_list: List[str]) -> Optional[List[float]]:
    """Generates a TF-IDF vector for a list of genres using the pre-defined vectorizer."""
    if not genre_list:
        return None
    try:
        # Ensure vectorizer is fitted (might happen via dummy data or loading a saved one)
        if not hasattr(tfidf_vectorizer, 'vocabulary_') or not tfidf_vectorizer.vocabulary_:
             logging.warning("TF-IDF vectorizer not properly fitted. Cannot generate genre vector.")
             # Attempt to load all genres from DB as fallback fitting data? Might be slow.
             # For now, return None.
             return None

        genre_text = " ".join(genre_list).lower()
        vector = tfidf_vectorizer.transform([genre_text]).toarray()[0]
        # Normalize the vector
        norm = np.linalg.norm(vector)
        normalized_vector = (vector / norm).tolist() if norm > 0 else vector.tolist()
        return normalized_vector
    except Exception as e:
        logging.error(f"Error generating TF-IDF vector for genres '{genre_list}': {e}")
        return None

# --- Recommendation Functions ---

def get_cf_recommendations(user_id: int, n: int, all_movie_ids: Set[int], rated_movie_ids: Set[int]) -> List[Dict]:
    """Generate recommendations using the Surprise SVD model."""
    if not cf_model:
        return []

    recommendations = []
    movies_to_predict = list(all_movie_ids - rated_movie_ids) # Only predict for unrated movies

    if not movies_to_predict:
        logging.info(f"User {user_id} has rated all available movies or no movies left to predict.")
        return []

    try:
         # Predict ratings for unrated movies
        predictions = [cf_model.predict(user_id, movie_id) for movie_id in movies_to_predict]

        # Sort predictions by estimated rating (highest first)
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top N recommendations
        for pred in predictions[:n]:
            recommendations.append({"movie_id": pred.iid, "score": pred.est}) # Use estimated rating as score

    except Exception as e:
         # Handle cases where user might not be in the training set (cold start for user in CF)
        logging.warning(f"Could not generate CF recommendations for user {user_id} (maybe new user?): {e}")
        return [] # Fallback to other methods

    return recommendations

def get_cb_recommendations(user_id: int, n: int, db: Session, rated_movie_ids: Set[int]) -> List[Dict]:
    """Generate recommendations based on content similarity using Weaviate."""
    wv_client = get_weaviate_client()
    if not wv_client:
        logging.warning("Weaviate client not available. Skipping Content-Based recommendations.")
        return []

    # 1. Get user's highly-rated movies
    # Fetch ratings with movie details including genres
    user_ratings = db.query(models.Rating).join(models.Movie)\
                     .filter(models.Rating.user_id == user_id)\
                     .filter(models.Rating.rating >= 4)\
                     .order_by(desc(models.Rating.rating), desc(models.Rating.timestamp))\
                     .limit(10).all() # Use top 10 highly rated movies as pivot


    if not user_ratings:
        logging.info(f"User {user_id} has no highly rated movies for CB recommendations.")
        return []

    # 2. Create a representative genre vector for the user's preferences
    # Combine genres from liked movies (could also average their Weaviate vectors if retrieved)
    liked_genres = set()
    for rating in user_ratings:
        if rating.movie and rating.movie.genres:
             genres = [g.strip() for g in rating.movie.genres.split(',')]
             liked_genres.update(genres)

    if not liked_genres:
         logging.info(f"No genres found for user {user_id}'s liked movies.")
         return []

    # Generate a vector for the combined liked genres
    # pivot_vector = get_vector_for_genres(list(liked_genres)) # Requires fitted TF-IDF

    # --- Alternative/Preferred CB: Average vectors of liked movies ---
    # This requires fetching vectors from Weaviate for the liked movies.
    # Let's simulate this by generating the genre vector for now.
    pivot_vector = get_vector_for_genres(list(liked_genres))


    if not pivot_vector:
        logging.warning(f"Could not generate pivot vector for user {user_id}. Skipping CB.")
        return []

    # 3. Find similar movies in Weaviate
    # Increase limit slightly to allow for filtering/diversification later
    similar_movies = find_similar_movies(vector=pivot_vector, limit=n * 2, exclude_ids=list(rated_movie_ids))


    # Convert distance to a similarity score (e.g., 1 / (1 + distance)) - lower distance is better
    recommendations = [
        {"movie_id": movie["movie_id"], "score": 1 / (1 + movie.get("distance", 1.0))}
        for movie in similar_movies
    ]

    # Sort by score descending
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return recommendations[:n]


def get_popularity_recommendations(n: int, db: Session, rated_movie_ids: Set[int]) -> List[Dict]:
    """Get globally popular movies, excluding those already rated by the user."""
    popular_movies = crud.get_popular_movies(db, limit=n * 2) # Get more to filter
    recommendations = []
    for movie in popular_movies:
        if movie.movie_id not in rated_movie_ids:
            # Score can be based on avg_rating or a combination
            score = movie.avg_rating * (movie.rating_count / 100) # Simple weighted score
            recommendations.append({"movie_id": movie.movie_id, "score": score})
        if len(recommendations) >= n:
            break
    recommendations.sort(key=lambda x: x["score"], reverse=True) # Ensure sorted
    return recommendations[:n]


def generate_hybrid_recommendations(
    user_id: int,
    n: int,
    db: Session,
    cf_weight: float = 0.5,
    cb_weight: float = 0.3,
    pop_weight: float = 0.2
) -> schemas.RecommendationResponse:
    """Combines different recommendation strategies."""

    user = crud.get_user(db, user_id)
    if not user:
        # Handle user not found case - maybe return global popular?
        logging.warning(f"User {user_id} not found. Returning popular movies.")
        pop_recs_list = get_popularity_recommendations(n, db, set())
        movies = crud.get_movies(db, limit=10000) # Fetch details for popular recs
        movie_map = {m.movie_id: m for m in movies}
        recs_movies = [schemas.Movie.from_orm(movie_map[rec['movie_id']]) for rec in pop_recs_list if rec['movie_id'] in movie_map]
        return schemas.RecommendationResponse(user_id=user_id, recommendations=recs_movies, method_used="popularity_fallback")


    rated_movie_ids = set(crud.get_movies_rated_by_user(db, user_id))
    all_movies = crud.get_movies(db, limit=10000) # Adjust limit as needed
    all_movie_ids = set(movie.movie_id for movie in all_movies)
    movie_map = {m.movie_id: m for m in all_movies} # For quick lookups


    # --- Candidate Generation ---
    # Adjust N for each method based on weights and desired final N
    n_cf = int(n * (1 + cf_weight)) # Get slightly more candidates
    n_cb = int(n * (1 + cb_weight))
    n_pop = int(n * (1 + pop_weight))

    cf_recs = get_cf_recommendations(user_id, n_cf, all_movie_ids, rated_movie_ids)
    cb_recs = get_cb_recommendations(user_id, n_cb, db, rated_movie_ids)

    # Determine if user is "cold start" (e.g., < 5 ratings)
    is_cold_start = len(rated_movie_ids) < 5

    # Always get popularity for fallback/mixing, especially important for cold start
    pop_recs = get_popularity_recommendations(n_pop, db, rated_movie_ids)


    # --- Combine and Rank ---
    combined_recs = {} # movie_id -> {'total_score': float, 'methods': list}

    # Add weighted scores from each method
    for rec in cf_recs:
        mid = rec['movie_id']
        if mid not in combined_recs: combined_recs[mid] = {'total_score': 0.0, 'methods': []}
        # Normalize CF score (e.g., scale 1-5 to 0-1) before applying weight
        normalized_cf_score = (rec['score'] - 1) / 4.0
        combined_recs[mid]['total_score'] += normalized_cf_score * cf_weight
        combined_recs[mid]['methods'].append('cf')

    for rec in cb_recs:
        mid = rec['movie_id']
        if mid not in combined_recs: combined_recs[mid] = {'total_score': 0.0, 'methods': []}
        # CB score is already ~0-1 (from 1/(1+dist))
        combined_recs[mid]['total_score'] += rec['score'] * cb_weight
        combined_recs[mid]['methods'].append('cb')

    # Give more weight to popularity for cold start users
    current_pop_weight = pop_weight * 2 if is_cold_start else pop_weight
    max_pop_score = max([r['score'] for r in pop_recs], default=1.0) # Normalize pop score
    if max_pop_score == 0: max_pop_score = 1.0 # Avoid division by zero

    for rec in pop_recs:
        mid = rec['movie_id']
        if mid not in combined_recs: combined_recs[mid] = {'total_score': 0.0, 'methods': []}
        normalized_pop_score = rec['score'] / max_pop_score
        combined_recs[mid]['total_score'] += normalized_pop_score * current_pop_weight
        combined_recs[mid]['methods'].append('pop')


    # --- Filter out already rated movies (should be handled by generators, but double check) ---
    final_candidates = {mid: data for mid, data in combined_recs.items() if mid not in rated_movie_ids}

    # --- Sort by combined score ---
    sorted_recs = sorted(final_candidates.items(), key=lambda item: item[1]['total_score'], reverse=True)

    # --- Enhance Diversity (Simple Genre Mixing) ---
    final_recommendations = []
    recommended_genres = set()
    genre_limit_per_rec = 2 # Max count for a single genre in the top N

    genre_counts = {}

    for movie_id, data in sorted_recs:
        if len(final_recommendations) >= n:
            break

        movie_details = movie_map.get(movie_id)
        if not movie_details or not movie_details.genres:
             continue # Skip if no details or genres


        current_genres = {g.strip() for g in movie_details.genres.split(',')}
        # Check if adding this movie overly specializes genres
        can_add = True
        for genre in current_genres:
             if genre_counts.get(genre, 0) >= genre_limit_per_rec:
                 # Too many of this genre already, maybe skip this movie
                 # More complex logic could penalize score instead of skipping
                 can_add = False
                 break

        if can_add:
             final_recommendations.append(movie_id)
             # Update genre counts
             for genre in current_genres:
                 genre_counts[genre] = genre_counts.get(genre, 0) + 1


    # If diversity filter reduced results too much, fill with top non-diverse results
    if len(final_recommendations) < n:
         remaining_needed = n - len(final_recommendations)
         backup_recs = [mid for mid, data in sorted_recs if mid not in final_recommendations]
         final_recommendations.extend(backup_recs[:remaining_needed])


    # --- Prepare Response ---
    # Fetch full movie details for the final list
    recommended_movie_objects = []
    method_summary = "hybrid" # Basic summary
    if is_cold_start and not cf_recs and not cb_recs:
        method_summary = "popularity_cold_start"
    elif not cf_recs and cf_weight > 0.1:
         method_summary += "_no_cf"
    elif not cb_recs and cb_weight > 0.1:
         method_summary += "_no_cb"


    for movie_id in final_recommendations:
         movie_obj = movie_map.get(movie_id)
         if movie_obj:
            # Convert SQLAlchemy model instance to Pydantic schema instance
            recommended_movie_objects.append(schemas.Movie.from_orm(movie_obj))
         else:
             logging.warning(f"Movie details not found for recommended movie_id: {movie_id}")


    return schemas.RecommendationResponse(
        user_id=user_id,
        recommendations=recommended_movie_objects,
        method_used=method_summary
    )