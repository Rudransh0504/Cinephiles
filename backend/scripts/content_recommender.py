"""
Content-based movie recommender using TF-IDF + cosine similarity
over genres, cast, director, and keywords.

Run standalone to inspect recommendations:
    python -m scripts.content_recommender

Later, the core logic here (build_similarity_matrix, get_recommendations)
gets imported into a FastAPI route.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.database import SessionLocal
from app import models


def load_movies_dataframe():
    """Pull all movies from the DB into a pandas DataFrame."""
    db = SessionLocal()
    try:
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
        return pd.DataFrame(data)
    finally:
        db.close()


def build_tags(row):
    """
    Combine all metadata into one 'soup' string per movie.
    Names are squashed to single tokens (e.g. "Leonardo DiCaprio" -> "leonardodicaprio")
    so TF-IDF treats a full name as one feature, not two separate common words.
    """
    genres = [g.replace(" ", "").lower() for g in row["genres"]]
    cast = [c.replace(" ", "").lower() for c in row["cast"]]
    director = [row["director"].replace(" ", "").lower()] if row["director"] else []
    keywords = [k.replace(" ", "").lower() for k in row["keywords"]]

    # weight genres and director a bit more by repeating them
    tags = genres * 2 + cast + director * 2 + keywords
    return " ".join(tags)


def build_similarity_matrix(df: pd.DataFrame):
    df["tags"] = df.apply(build_tags, axis=1)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["tags"])

    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


def get_recommendations(df: pd.DataFrame, similarity, movie_id: int, top_n: int = 10):
    if movie_id not in df["id"].values:
        return []

    idx = df.index[df["id"] == movie_id][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # skip index 0 since that's the movie itself (similarity = 1.0)
    top_matches = scores[1:top_n + 1]

    results = []
    for i, score in top_matches:
        row = df.iloc[i]
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "poster_path": row["poster_path"],
            "vote_average": row["vote_average"],
            "similarity_score": round(float(score), 4),
        })
    return results


def main():
    print("Loading movies from database...")
    df = load_movies_dataframe()
    print(f"Loaded {len(df)} movies.\n")

    print("Building TF-IDF similarity matrix...")
    similarity = build_similarity_matrix(df)
    print("Done.\n")

    # --- Inspect a few test cases ---
    test_titles = ["Inception", "The Dark Knight", "Interstellar"]

    for title in test_titles:
        matches = df[df["title"].str.lower() == title.lower()]
        if matches.empty:
            print(f"'{title}' not found in dataset — skipping.\n")
            continue

        movie_id = int(matches.iloc[0]["id"])
        print(f"Recommendations for '{title}':")
        recs = get_recommendations(df, similarity, movie_id, top_n=5)
        for r in recs:
            print(f"  {r['title']} (score: {r['similarity_score']})")
        print()


if __name__ == "__main__":
    main()