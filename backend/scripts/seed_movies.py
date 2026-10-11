"""
One-off / periodically re-run script to populate the `movies` table
with enriched TMDB metadata (genres, cast, director, keywords).

Run with:
    python -m scripts.seed_movies
from the backend/ folder, with venv activated.
"""

import sys
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Allow importing from the app package when running this script directly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal, engine, Base
from app import models

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    raise SystemExit("TMDB_API_KEY is missing from backend/.env")
BASE_URL = "https://api.themoviedb.org/3"

# How many pages of each list to pull (20 movies per page)
PAGES_PER_CATEGORY = 25

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Session with automatic retries on connection failures / rate limits
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,  # waits 1s, 2s, 4s, 8s, 16s between retries
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


def fetch_movie_list(category: str, page: int):
    url = f"{BASE_URL}/movie/{category}"
    params = {"api_key": API_KEY, "language": "en-US", "page": page}
    res = session.get(url, params=params, headers=HEADERS, timeout=15)
    res.raise_for_status()
    return res.json()["results"]


def fetch_movie_details(movie_id: int):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "append_to_response": "credits,keywords",
    }
    res = session.get(url, params=params, headers=HEADERS, timeout=15)
    res.raise_for_status()
    return res.json()


def extract_director(credits: dict) -> str:
    for member in credits.get("crew", []):
        if member.get("job") == "Director":
            return member.get("name")
    return None


def build_movie_record(details: dict) -> dict:
    credits = details.get("credits", {})
    keywords_data = details.get("keywords", {}).get("keywords", [])

    return {
        "id": details["id"],
        "title": details.get("title"),
        "overview": details.get("overview"),
        "poster_path": details.get("poster_path"),
        "release_date": details.get("release_date"),
        "vote_average": details.get("vote_average"),
        "runtime": details.get("runtime"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "cast": [c["name"] for c in credits.get("cast", [])[:5]],
        "director": extract_director(credits),
        "keywords": [k["name"] for k in keywords_data],
    }


def upsert_movie(db, record: dict):
    existing = db.query(models.Movie).filter(models.Movie.id == record["id"]).first()
    if existing:
        for key, value in record.items():
            setattr(existing, key, value)
    else:
        db.add(models.Movie(**record))


def main():
    Base.metadata.create_all(bind=engine)  # ensures movies table exists
    db = SessionLocal()

    categories = ["popular", "top_rated"]
    seen_ids = set()
    total_saved = 0
    total_failed = 0

    try:
        for category in categories:
            for page in range(1, PAGES_PER_CATEGORY + 1):
                print(f"Fetching {category} page {page}...")

                try:
                    movies = fetch_movie_list(category, page)
                except Exception as e:
                    print(f"  Failed to fetch page {page}: {e} — skipping")
                    continue

                for movie in movies:
                    if movie["id"] in seen_ids:
                        continue
                    seen_ids.add(movie["id"])

                    try:
                        details = fetch_movie_details(movie["id"])
                        record = build_movie_record(details)
                        upsert_movie(db, record)
                        total_saved += 1
                        print(f"  Saved: {record['title']}")
                    except Exception as e:
                        total_failed += 1
                        print(f"  Failed on movie {movie['id']}: {e}")

                    time.sleep(0.05)  # light rate-limit courtesy

                db.commit()

        print(f"\nDone. {total_saved} movies saved/updated. {total_failed} failed.")

    finally:
        db.close()


if __name__ == "__main__":
    main()