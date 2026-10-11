from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


# ---------- Auth ----------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


# ---------- Watchlist ----------
class WatchlistItemCreate(BaseModel):
    tmdb_movie_id: int
    title: str
    poster_path: Optional[str] = None
    vote_average: Optional[float] = None
    release_date: Optional[str] = None
    overview: Optional[str] = None

class WatchlistItemOut(WatchlistItemCreate):
    id: int
    added_at: datetime
    class Config:
        from_attributes = True


# ---------- Watched ----------
class WatchedMovieCreate(BaseModel):
    tmdb_movie_id: int
    title: str
    poster_path: Optional[str] = None
    vote_average: Optional[float] = None
    release_date: Optional[str] = None
    overview: Optional[str] = None
    user_rating: float

class WatchedMovieOut(WatchedMovieCreate):
    id: int
    watched_at: datetime
    class Config:
        from_attributes = True


# ---------- Votes ----------
class VoteCreate(BaseModel):
    entity_type: str  # 'movie' or 'actor'
    contender_a_id: int
    contender_a_name: str
    contender_a_image: Optional[str] = None
    contender_b_id: int
    contender_b_name: str
    contender_b_image: Optional[str] = None
    winner_id: int

class VoteOut(VoteCreate):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True