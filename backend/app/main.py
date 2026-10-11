from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base, SessionLocal
from . import models
from .routers import auth, watchlist, watched, votes, recommendations, tmdb
from .recommender import recommender
from .hybrid import hybrid

app = FastAPI(title="Cinephiles API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(watched.router, prefix="/api/watched", tags=["watched"])
app.include_router(votes.router, prefix="/api/votes", tags=["votes"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(tmdb.router, prefix="/api/tmdb", tags=["tmdb"])


@app.on_event("startup")
def build_recommenders_on_startup():
    db = SessionLocal()
    try:
        print("Building content-based similarity matrix...")
        recommender.build(db)
        print("Content recommender ready with %d movies." % (len(recommender.df) if recommender.df is not None else 0))
    finally:
        db.close()
    hybrid.load()


@app.get("/")
def root():
    return {"message": "Cinephiles API is running"}