from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON

class Movie(Base):
    """
    Local cache of TMDB movie metadata, enriched with cast/genre/keyword data.
    This is what the content-based recommender will run similarity computations against,
    instead of hitting the TMDB API live on every request.
    """
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)  # tmdb movie id, used directly
    title = Column(String, nullable=False)
    overview = Column(Text)
    poster_path = Column(String)
    release_date = Column(String)
    vote_average = Column(Float)
    runtime = Column(Integer)

    genres = Column(JSON)      # list of strings, e.g. ["Action", "Sci-Fi"]
    cast = Column(JSON)        # list of top ~5 cast member names
    director = Column(String)
    keywords = Column(JSON)    # list of TMDB keyword strings

    last_updated = Column(DateTime(timezone=True), server_default=func.now())

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    watched_movies = relationship("WatchedMovie", back_populates="user", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="user", cascade="all, delete-orphan")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tmdb_movie_id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    poster_path = Column(String)
    vote_average = Column(Float)
    release_date = Column(String)
    overview = Column(Text)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="watchlist_items")


class WatchedMovie(Base):
    __tablename__ = "watched_movies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tmdb_movie_id = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    poster_path = Column(String)
    vote_average = Column(Float)
    release_date = Column(String)
    overview = Column(Text)
    user_rating = Column(Float, nullable=False)
    watched_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="watched_movies")


class Vote(Base):
    """Powers the FaceOff feature — works for both movies and actors."""
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    entity_type = Column(String, nullable=False)   # 'movie' or 'actor'
    contender_a_id = Column(Integer, nullable=False)   # tmdb id
    contender_a_name = Column(String, nullable=False)
    contender_a_image = Column(String)
    contender_b_id = Column(Integer, nullable=False)
    contender_b_name = Column(String, nullable=False)
    contender_b_image = Column(String)
    winner_id = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="votes")