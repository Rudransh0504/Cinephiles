"""
Loads the content-based similarity matrix once (at app startup) and
keeps it in memory. Reused by the /recommend endpoint on every request
instead of recomputing TF-IDF from scratch each time.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy.orm import Session
from . import models


class ContentRecommender:
    def __init__(self):
        self.df = None
        self.similarity = None

    def build(self, db: Session):
        movies = db.query(models.Movie).all()
        data = []
        for m in movies:
            data.append({
                "id": m.id,
                "title": m.title,
                "genres": m.genres or [],
                "cast": m.cast or [],
                "director": m.director or "",
                "keywords": m.keywords or [],
                "poster_path": m.poster_path,
                "vote_average": m.vote_average,
            })
        self.df = pd.DataFrame(data)

        if self.df.empty:
            self.similarity = None
            return

        self.df["tags"] = self.df.apply(self._build_tags, axis=1)

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.df["tags"])

        self.similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    @staticmethod
    def _build_tags(row):
        genres = [g.replace(" ", "").lower() for g in row["genres"]]
        cast = [c.replace(" ", "").lower() for c in row["cast"]]
        director = [row["director"].replace(" ", "").lower()] if row["director"] else []
        keywords = [k.replace(" ", "").lower() for k in row["keywords"]]

        tags = genres * 2 + cast + director * 2 + keywords
        return " ".join(tags)

    def recommend(self, movie_id: int, top_n: int = 10):
        if self.df is None or self.similarity is None:
            return []

        matches = self.df.index[self.df["id"] == movie_id]
        if len(matches) == 0:
            return []

        idx = matches[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_matches = scores[1:top_n + 1]

        results = []
        for i, score in top_matches:
            row = self.df.iloc[i]
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "poster_path": row["poster_path"],
                "vote_average": row["vote_average"],
                "similarity_score": round(float(score), 4),
            })
        return results


# Single shared instance, built once at app startup
recommender = ContentRecommender()