# src/main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import logging
import contextlib # Used for async context manager for lifespan

# Import project modules
from . import crud, models, schemas, recommendation_logic
from .database import SessionLocal, engine, get_db
# Import Weaviate client and close function
from .weaviate_client import create_weaviate_schema, get_weaviate_client, close_weaviate_connection

# API Rate Limiting setup
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)

# --- Application Lifespan Management ---
# Use FastAPI's lifespan context manager for startup and shutdown events
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logging.info("Application startup...")
    # Ensure Weaviate connection and collection exist on startup
    try:
        # Establish connection (get_weaviate_client handles connection logic)
        client = get_weaviate_client()
        if client:
            create_weaviate_schema() # Ensure schema exists
        else:
            logging.error("Failed to connect to Weaviate on startup. Vector search may fail.")
    except Exception as e:
        logging.error(f"Failed to ensure Weaviate connection/schema on startup: {e}")

    # Database tables (optional here if load_data does it)
    # models.Base.metadata.create_all(bind=engine)
    logging.info("Application startup complete.")
    yield # Application runs here
    # --- Shutdown ---
    logging.info("Application shutdown...")
    close_weaviate_connection() # Close Weaviate connection on shutdown
    logging.info("Application shutdown complete.")

# --- FastAPI App Initialization ---
# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
# Pass the lifespan manager to the FastAPI app
app = FastAPI(title="Simple Recommendation Engine (Weaviate + TFIDF)", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- API Endpoints ---
# (Endpoints remain the same as before)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommendation Engine API (Weaviate + TFIDF)"}

@app.get("/recommendations/{user_id}", response_model=schemas.RecommendationResponse)
@limiter.limit("10/minute")
def get_recommendations_for_user(
    request: Request,
    user_id: int,
    n: int = 10,
    db: Session = Depends(get_db)
):
    """
    Generates hybrid recommendations for a given user ID.
    """
    logging.info(f"Received recommendation request for user_id={user_id}, n={n}")
    start_time = time.time()

    if n <= 0 or n > 50:
        raise HTTPException(status_code=400, detail="Number of recommendations (n) must be between 1 and 50.")

    recommendations = recommendation_logic.generate_hybrid_recommendations(user_id=user_id, n=n, db=db)

    if not recommendations.recommendations and recommendations.method_used != "popularity_fallback":
         user = crud.get_user(db, user_id)
         if not user:
              raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")

    end_time = time.time()
    logging.info(f"Generated {len(recommendations.recommendations)} recommendations for user {user_id} in {end_time - start_time:.4f} seconds using method: {recommendations.method_used}")

    return recommendations


@app.post("/ratings", response_model=schemas.Rating, status_code=201)
@limiter.limit("60/minute")
def create_new_rating(
    request: Request,
    rating_data: schemas.RatingCreate,
    db: Session = Depends(get_db)
):
    """
    Allows a user to submit a new movie rating.
    """
    logging.info(f"Received rating submission: User {rating_data.user_id}, Movie {rating_data.movie_id}, Rating {rating_data.rating}")
    start_time = time.time()
    user = crud.get_user(db, user_id=rating_data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {rating_data.user_id} not found")
    movie = crud.get_movie(db, movie_id=rating_data.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with ID {rating_data.movie_id} not found")
    if not 1 <= rating_data.rating <= 5:
         raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    try:
        db_rating = crud.create_rating(db=db, rating=rating_data)
        end_time = time.time()
        logging.info(f"Successfully created rating {db_rating.rating_id} in {end_time - start_time:.4f} seconds.")
        # Fetch again or use relationships for response
        created_rating = db.query(models.Rating).filter(models.Rating.rating_id == db_rating.rating_id).first()
        # Ensure user and movie objects are valid before creating response schema
        if not created_rating or not user or not movie:
             raise HTTPException(status_code=500, detail="Failed to retrieve rating details after creation.")

        response_rating = schemas.Rating(
            rating_id=created_rating.rating_id,
            user_id=created_rating.user_id,
            movie_id=created_rating.movie_id,
            rating=created_rating.rating,
            timestamp=created_rating.timestamp,
            user=schemas.User.from_orm(user),
            movie=schemas.Movie.from_orm(movie)
        )
        return response_rating
    except Exception as e:
        logging.error(f"Error creating rating: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
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
# (Keep previous TODOs)
# - Consider more robust connection pooling for database if needed
# - Add health check endpoint for Weaviate connection status
