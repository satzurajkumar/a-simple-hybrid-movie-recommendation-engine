# src/schemas.py
from pydantic import BaseModel
from typing import List, Optional

# --- Movie Schemas ---
class MovieBase(BaseModel):
    title: str
    genres: str
    release_date: Optional[str] = None
    imdb_url: Optional[str] = None
    avg_rating: Optional[float] = 0.0
    rating_count: Optional[int] = 0


class MovieCreate(MovieBase):
    movie_id: int

class Movie(MovieBase):
    movie_id: int

    class Config:
        #orm_mode = True # Changed from from_attributes=True for compatibility
        from_attributes = True
# --- User Schemas ---
class UserBase(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    zip_code: Optional[str] = None

class UserCreate(UserBase):
    user_id: int

class User(UserBase):
    user_id: int

    class Config:
        #orm_mode = True
        from_attributes = True

# --- Rating Schemas ---
class RatingBase(BaseModel):
    rating: int

class RatingCreate(RatingBase):
    user_id: int
    movie_id: int
    timestamp: Optional[int] = None # Allow optional timestamp

class Rating(RatingBase):
    rating_id: int
    user_id: int
    movie_id: int
    timestamp: int
    movie: Movie # Include related movie details
    user: User # Include related user details


    class Config:
        #orm_mode = True
        from_attributes = True

# --- Recommendation Schemas ---
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Movie]
    method_used: Optional[str] = None # Indicate which method(s) primarily contributed

class Interaction(BaseModel):
    user_id: int
    movie_id: int
    interaction_type: str # e.g., 'click', 'watch_complete', 'add_watchlist'
    timestamp: Optional[int] = None