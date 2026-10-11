from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user

router = APIRouter()


@router.get("/", response_model=List[schemas.WatchlistItemOut])
def get_watchlist(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return (
        db.query(models.WatchlistItem)
        .filter(models.WatchlistItem.user_id == current_user.id)
        .order_by(models.WatchlistItem.added_at.desc())
        .all()
    )


@router.post("/", response_model=schemas.WatchlistItemOut, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    item: schemas.WatchlistItemCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    existing = (
        db.query(models.WatchlistItem)
        .filter(
            models.WatchlistItem.user_id == current_user.id,
            models.WatchlistItem.tmdb_movie_id == item.tmdb_movie_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Movie already in watchlist")

    new_item = models.WatchlistItem(user_id=current_user.id, **item.dict())
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item


@router.delete("/{tmdb_movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(
    tmdb_movie_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    item = (
        db.query(models.WatchlistItem)
        .filter(
            models.WatchlistItem.user_id == current_user.id,
            models.WatchlistItem.tmdb_movie_id == tmdb_movie_id,
        )
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Movie not in watchlist")

    db.delete(item)
    db.commit()
    return None