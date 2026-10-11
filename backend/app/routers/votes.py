from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from typing import List, Optional

from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user

router = APIRouter()


def _check_entity_type(entity_type: str):
    if entity_type not in ("movie", "actor"):
        raise HTTPException(status_code=400, detail="entity_type must be 'movie' or 'actor'")


def _entity_rows(db: Session, entity_type: str, ids: Optional[list] = None):
    """
    One row per contender: (id, name, image, appearances, wins).
    Each vote appears twice (once per contender side), so we stack both
    sides with UNION ALL and let the database do the counting.
    """
    V = models.Vote

    def side(id_col, name_col, image_col):
        q = db.query(
            id_col.label("cid"),
            name_col.label("name"),
            image_col.label("image"),
            case((V.winner_id == id_col, 1), else_=0).label("won"),
        ).filter(V.entity_type == entity_type)
        if ids is not None:
            q = q.filter(id_col.in_(ids))
        return q

    a = side(V.contender_a_id, V.contender_a_name, V.contender_a_image)
    b = side(V.contender_b_id, V.contender_b_name, V.contender_b_image)
    u = a.union_all(b).subquery()

    return (
        db.query(
            u.c.cid,
            func.max(u.c.name),
            func.max(u.c.image),
            func.count(),
            func.sum(u.c.won),
        )
        .group_by(u.c.cid)
        .all()
    )


def _format(rows):
    out = []
    for cid, name, image, total, wins in rows:
        total = int(total)
        wins = int(wins or 0)
        out.append({
            "id": cid,
            "name": name,
            "image": image,
            "wins": wins,
            "losses": total - wins,
            "total": total,
            "win_rate": round(wins / total * 100, 1) if total else 0,
        })
    return out


def _stats_for(db: Session, entity_type: str, ids: list):
    found = {r["id"]: r for r in _format(_entity_rows(db, entity_type, ids))}
    return [
        found.get(i, {"id": i, "wins": 0, "losses": 0, "total": 0, "win_rate": 0})
        for i in ids
    ]


# ---------- Writes ----------

@router.post("/", response_model=schemas.VoteOut, status_code=status.HTTP_201_CREATED)
def cast_vote(
    vote: schemas.VoteCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _check_entity_type(vote.entity_type)
    if vote.winner_id not in (vote.contender_a_id, vote.contender_b_id):
        raise HTTPException(status_code=400, detail="winner_id must match one of the two contenders")

    new_vote = models.Vote(user_id=current_user.id, **vote.dict())
    db.add(new_vote)
    db.commit()
    db.refresh(new_vote)
    return new_vote


@router.post("/cast")
def cast_vote_with_stats(
    vote: schemas.VoteCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Saves a vote and returns community stats for both contenders + total count, in one trip."""
    _check_entity_type(vote.entity_type)
    if vote.winner_id not in (vote.contender_a_id, vote.contender_b_id):
        raise HTTPException(status_code=400, detail="winner_id must match one of the two contenders")

    db.add(models.Vote(user_id=current_user.id, **vote.dict()))
    db.commit()

    stats = _stats_for(db, vote.entity_type, [vote.contender_a_id, vote.contender_b_id])
    count = (
        db.query(func.count(models.Vote.id))
        .filter(models.Vote.entity_type == vote.entity_type)
        .scalar()
    )
    return {"stats": stats, "count": count or 0}


# ---------- Reads ----------

@router.get("/history", response_model=List[schemas.VoteOut])
def get_my_votes(
    entity_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    query = db.query(models.Vote).filter(models.Vote.user_id == current_user.id)
    if entity_type:
        query = query.filter(models.Vote.entity_type == entity_type)
    return query.order_by(models.Vote.created_at.desc()).all()


@router.get("/leaderboard")
def get_leaderboard(entity_type: str = "movie", db: Session = Depends(get_db)):
    _check_entity_type(entity_type)
    board = _format(_entity_rows(db, entity_type))
    board.sort(key=lambda x: (-x["win_rate"], -x["total"]))
    return board[:10]


@router.get("/stats")
def get_entity_stats(entity_type: str, ids: str, db: Session = Depends(get_db)):
    """Community wins/losses for specific contenders. ids is comma-separated, e.g. ids=27205,157336"""
    _check_entity_type(entity_type)
    try:
        id_list = [int(i) for i in ids.split(",") if i.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be comma-separated integers")
    return _stats_for(db, entity_type, id_list)


@router.get("/count")
def get_vote_count(entity_type: str = "movie", db: Session = Depends(get_db)):
    count = (
        db.query(func.count(models.Vote.id))
        .filter(models.Vote.entity_type == entity_type)
        .scalar()
    )
    return {"count": count or 0}