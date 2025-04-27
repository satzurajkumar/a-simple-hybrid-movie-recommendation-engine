# src/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, BigInteger, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(1), nullable=True)
    occupation = Column(String(100), nullable=True)
    zip_code = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    ratings = relationship("Rating", back_populates="user")

class Movie(Base):
    __tablename__ = "movies"
    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    release_date = Column(String(30), nullable=True) # Storing as string from dataset
    imdb_url = Column(String(255), nullable=True)
    genres = Column(String(255)) # Comma-separated or use separate Genre table for normalization
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Add avg_rating and rating_count columns if needed for popularity, updated periodically
    avg_rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)


    ratings = relationship("Rating", back_populates="movie")

class Rating(Base):
    __tablename__ = "ratings"
    rating_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), index=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), index=True)
    rating = Column(Integer)
    timestamp = Column(BigInteger) # Unix timestamp from dataset
    created_at = Column(DateTime(timezone=True), server_default=func.now())


    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")