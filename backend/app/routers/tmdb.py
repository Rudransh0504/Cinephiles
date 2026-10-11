import os
import time
import requests
from fastapi import APIRouter, HTTPException, Request
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_KEY = os.getenv("TMDB_API_KEY")

ALLOWED_PREFIXES = ("movie/", "person/", "search/", "trending/", "tv/")
CACHE_TTL = 600  # seconds
_cache = {}


@router.get("/{path:path}")
def tmdb_proxy(path: str, request: Request):
    if not TMDB_KEY:
        raise HTTPException(status_code=500, detail="TMDB_API_KEY is not configured on the server")
    if ".." in path or not path.startswith(ALLOWED_PREFIXES):
        raise HTTPException(status_code=404, detail="Not allowed")

    params = {k: v for k, v in request.query_params.items() if k != "api_key"}
    cache_key = path + "?" + "&".join("%s=%s" % (k, params[k]) for k in sorted(params))

    hit = _cache.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    try:
        res = requests.get(
            "%s/%s" % (TMDB_BASE, path),
            params=dict(params, api_key=TMDB_KEY),
            timeout=15,
        )
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not reach TMDB")

    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail="TMDB request failed")

    data = res.json()
    if len(_cache) > 500:
        _cache.clear()
    _cache[cache_key] = (time.time(), data)
    return data