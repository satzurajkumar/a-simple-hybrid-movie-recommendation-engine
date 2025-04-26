# src/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from . import models, schemas
from typing import List, Optional

# --- User CRUD ---
def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Movie CRUD ---
def get_movie(db: Session, movie_id: int) -> Optional[models.Movie]:
    return db.query(models.Movie).filter(models.Movie.movie_id == movie_id).first()

def get_movies(db: Session, skip: int = 0, limit: int = 100) -> List[models.Movie]:
     return db.query(models.Movie).offset(skip).limit(limit).all()


def create_movie(db: Session, movie: schemas.MovieCreate) -> models.Movie:
    db_movie = models.Movie(**movie.dict())
    db.add(db_movie)
    db.commit()
    db.refresh(db_movie)
    return db_movie

def get_popular_movies(db: Session, limit: int = 10, min_ratings: int = 10) -> List[models.Movie]:
    """Gets movies ranked by average rating, requiring a minimum number of ratings."""
    return db.query(models.Movie)\
             .filter(models.Movie.rating_count >= min_ratings)\
             .order_by(desc(models.Movie.avg_rating), desc(models.Movie.rating_count))\
             .limit(limit)\
             .all()


# --- Rating CRUD ---
def get_user_ratings(db: Session, user_id: int, limit: int = 100) -> List[models.Rating]:
    return db.query(models.Rating)\
             .filter(models.Rating.user_id == user_id)\
             .order_by(desc(models.Rating.timestamp))\
             .limit(limit)\
             .all()

def create_rating(db: Session, rating: schemas.RatingCreate) -> models.Rating:
    # Check if user and movie exist
    db_user = get_user(db, rating.user_id)
    db_movie = get_movie(db, rating.movie_id)
    if not db_user or not db_movie:
        # Or raise specific exceptions
        return None # Indicate failure


    # Use current time if timestamp not provided
    import time
    timestamp = rating.timestamp if rating.timestamp is not None else int(time.time())


    db_rating = models.Rating(
        user_id=rating.user_id,
        movie_id=rating.movie_id,
        rating=rating.rating,
        timestamp=timestamp # Ensure timestamp is included
    )
    db.add(db_rating)

    # --- Update Movie Average Rating (Important for Popularity) ---
    # This can be slow if done synchronously on every rating.
    # Consider background tasks or periodic batch updates for production.
    try:
        # Recalculate average and count for the specific movie
        result = db.query(func.avg(models.Rating.rating), func.count(models.Rating.rating))\
                   .filter(models.Rating.movie_id == rating.movie_id)\
                   .one()
        new_avg, new_count = result

        if new_avg is not None and new_count is not None:
           db_movie.avg_rating = new_avg
           db_movie.rating_count = new_count
           db.add(db_movie) # Add movie to session if not already there

        db.commit()
        db.refresh(db_rating)
        # db.refresh(db_movie) # Refresh movie if needed elsewhere
        return db_rating
    except Exception as e:
        db.rollback() # Rollback transaction on error
        print(f"Error updating movie average rating: {e}")
        # Optionally re-raise or handle
        raise e # Or return None depending on desired behavior

def get_movies_rated_by_user(db: Session, user_id: int) -> List[int]:
    """Returns a list of movie IDs rated by the user."""
    return [rating.movie_id for rating in db.query(models.Rating.movie_id).filter(models.Rating.user_id == user_id).all()]