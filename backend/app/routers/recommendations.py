from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from ..recommender import recommender
from ..hybrid import hybrid
from .. import models
from ..database import get_db
from ..dependencies import get_current_user

router = APIRouter()


@router.get("/for-user/me")
def get_recommendations_for_user(
    top_n: int = Query(10, ge=1, le=50),
    method: str = Query("hybrid", pattern="^(hybrid|content)$"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    watched = (
        db.query(models.WatchedMovie)
        .filter(models.WatchedMovie.user_id == current_user.id)
        .all()
    )
    if not watched:
        raise HTTPException(
            status_code=404,
            detail="No ratings yet. Rate some movies to get personalized recommendations.",
        )

    ratings10 = {w.tmdb_movie_id: float(w.user_rating) for w in watched}
    watchlist_ids = {
        w.tmdb_movie_id
        for w in db.query(models.WatchlistItem)
        .filter(models.WatchlistItem.user_id == current_user.id)
        .all()
    }
    exclude = set(ratings10.keys()) | watchlist_ids

    results = hybrid.recommend(ratings10, exclude, top_n=top_n, use_cf=(method == "hybrid"))
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Not enough signal yet. Rate a few more movies (7+ works best).",
        )
    return results


@router.get("/{movie_id}")
def get_recommendations(movie_id: int, top_n: int = Query(10, ge=1, le=50)):
    results = recommender.recommend(movie_id, top_n=top_n)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="Movie not found in recommendation dataset, or dataset is empty.",
        )
    return results