# src/main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import logging

# Import project modules
from . import crud, models, schemas, recommendation_logic
from .database import SessionLocal, engine, get_db
from .weaviate_client import create_weaviate_schema # Ensure schema on startup

# API Rate Limiting setup
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)

# Create DB tables on startup if they don't exist (optional here if load_data does it)
# models.Base.metadata.create_all(bind=engine)

# Ensure Weaviate schema exists on startup
try:
    create_weaviate_schema()
except Exception as e:
    logging.error(f"Failed to ensure Weaviate schema on startup: {e}")


# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Simple Recommendation Engine")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommendation Engine API"}

@app.get("/recommendations/{user_id}", response_model=schemas.RecommendationResponse)
@limiter.limit("10/minute") # Example rate limit: 10 requests per minute per IP
def get_recommendations_for_user(
    request: Request, # Required for rate limiter
    user_id: int,
    n: int = 10, # Default number of recommendations
    db: Session = Depends(get_db)
):
    """
    Generates hybrid recommendations for a given user ID.
    """
    logging.info(f"Received recommendation request for user_id={user_id}, n={n}")
    start_time = time.time()

    if n <= 0 or n > 50: # Add limits to n
        raise HTTPException(status_code=400, detail="Number of recommendations (n) must be between 1 and 50.")

    recommendations = recommendation_logic.generate_hybrid_recommendations(user_id=user_id, n=n, db=db)

    if not recommendations.recommendations and recommendations.method_used != "popularity_fallback":
         # Check if user exists but just has no specific recs generated
         user = crud.get_user(db, user_id)
         if not user:
              raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
         # else: user exists, but no recommendations generated (e.g., rated everything?)


    end_time = time.time()
    logging.info(f"Generated {len(recommendations.recommendations)} recommendations for user {user_id} in {end_time - start_time:.4f} seconds using method: {recommendations.method_used}")

    return recommendations


@app.post("/ratings", response_model=schemas.Rating, status_code=201)
@limiter.limit("60/minute") # Allow more rating submissions
def create_new_rating(
    request: Request, # Required for rate limiter
    rating_data: schemas.RatingCreate,
    db: Session = Depends(get_db)
):
    """
    Allows a user to submit a new movie rating.
    This implicitly provides feedback to the system.
    Actual model retraining based on new ratings needs a separate process.
    """
    logging.info(f"Received rating submission: User {rating_data.user_id}, Movie {rating_data.movie_id}, Rating {rating_data.rating}")
    start_time = time.time()

    # Validate user and movie existence
    user = crud.get_user(db, user_id=rating_data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {rating_data.user_id} not found")

    movie = crud.get_movie(db, movie_id=rating_data.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with ID {rating_data.movie_id} not found")

    # Validate rating value
    if not 1 <= rating_data.rating <= 5:
         raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")


    try:
        db_rating = crud.create_rating(db=db, rating=rating_data)
        end_time = time.time()
        logging.info(f"Successfully created rating {db_rating.rating_id} in {end_time - start_time:.4f} seconds.")

        # Convert SQLAlchemy model to Pydantic schema for response, including related objects
        # We need to fetch the rating again or use the relationship attributes correctly
        # Easiest way: Fetch again with relationships loaded (or configure session for immediate loading)
        created_rating = db.query(models.Rating).options(
             # selectinload(models.Rating.user), # Uncomment if needed
             # selectinload(models.Rating.movie)
             ).filter(models.Rating.rating_id == db_rating.rating_id).first()

        # Manually construct response if relationships aren't loaded/needed in response
        response_rating = schemas.Rating(
            rating_id=created_rating.rating_id,
            user_id=created_rating.user_id,
            movie_id=created_rating.movie_id,
            rating=created_rating.rating,
            timestamp=created_rating.timestamp,
            # Populate nested models manually or ensure they are loaded
            user=schemas.User.from_orm(user), # Use the already fetched user
            movie=schemas.Movie.from_orm(movie) # Use the already fetched movie
        )


        return response_rating

    except Exception as e:
        # Catch potential errors during rating creation or avg update
        logging.error(f"Error creating rating: {e}")
        raise HTTPException(status_code=500, detail="Failed to create rating due to an internal error.")


# Optional: Add endpoints for getting user/movie details
@app.get("/movies/{movie_id}", response_model=schemas.Movie)
@limiter.limit("30/minute")
def read_movie(request: Request, movie_id: int, db: Session = Depends(get_db)):
    db_movie = crud.get_movie(db, movie_id=movie_id)
    if db_movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return db_movie

@app.get("/users/{user_id}", response_model=schemas.User)
@limiter.limit("30/minute")
def read_user(request: Request, user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


# --- TODO / Future Enhancements ---
# - Add endpoint for implicit interactions (clicks, views) -> POST /interactions
# - Implement background tasks (e.g., using Celery or FastAPI's BackgroundTasks) for:
#     - Periodically retraining the CF model
#     - Updating movie average ratings in batches
#     - Updating user profiles based on interactions
# - More sophisticated diversity logic (e.g., Maximal Marginal Relevance - MMR)
# - Better cold-start handling (e.g., asking new users for genre preferences)
# - A/B testing framework for different recommendation algorithms/weights
# - Caching layer (e.g., Redis) for recommendations and popular items
# - Asynchronous database operations (e.g., using databases/asyncpg or aiomysql with SQLAlchemy 2.0+)
# - Proper handling of Weaviate vector updates if movie metadata changes
# - Input validation beyond Pydantic (e.g., business logic validation)
# - Authentication and Authorization