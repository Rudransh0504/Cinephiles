"""
Hybrid recommender used by the API: blends content-based similarity (over the seeded
movies table) with a collaborative-filtering model trained on MovieLens.
"""

import os
import numpy as np

from .mf import ALSRecommender
from .recommender import recommender

LIKE_THRESHOLD_10 = 7     # app ratings are on a 1-10 scale
RAMP_RATINGS = 15         # CF influence ramps up until the user has this many ratings
MIN_CF_RATINGS = 3

ARTIFACT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "cf_model.npz")
)


def _pct(values, mask):
    """Percentile rank (0..1) among entries where mask is True; 0 elsewhere."""
    out = np.zeros(len(values))
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return out
    out[idx] = np.argsort(np.argsort(values[idx])) / (len(idx) - 1.0)
    return out


class HybridRecommender:
    def __init__(self):
        self.model = None
        self.item_index = {}
        self.w_star = 0.5

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        if not os.path.exists(ARTIFACT_PATH):
            print("CF model not found at %s; using content-based only." % ARTIFACT_PATH)
            return False
        self.model, item_ids, meta = ALSRecommender.load(ARTIFACT_PATH)
        self.item_index = {int(t): i for i, t in enumerate(item_ids)}
        self.w_star = meta["w_star"]
        print("CF model loaded: %d movies, blend weight %.2f" % (len(item_ids), self.w_star))
        return True

    def recommend(self, ratings10, exclude_ids, top_n=10, use_cf=True):
        """
        ratings10:   {tmdb_id: rating on the app's 1-10 scale}
        exclude_ids: set of tmdb ids to leave out (already watched / on watchlist)
        """
        df = recommender.df
        sim = recommender.similarity
        if df is None or sim is None or len(df) == 0:
            return []

        ids = df["id"].to_numpy()
        pos = {int(t): j for j, t in enumerate(ids)}
        cand = np.array([int(t) not in exclude_ids for t in ids])
        if not cand.any():
            return []

        # content-based score for every seeded movie
        liked = [(pos[t], r) for t, r in ratings10.items() if t in pos and r >= LIKE_THRESHOLD_10]
        has_content = len(liked) > 0
        content = np.zeros(len(ids))
        if has_content:
            rows = np.array([p for p, _ in liked])
            weights = np.array([r / 10.0 for _, r in liked], dtype=float)
            content = weights @ sim[rows]

        # collaborative-filtering score for movies that exist in MovieLens
        known = [(self.item_index[t], r / 2.0) for t, r in ratings10.items() if t in self.item_index]
        cf_ready = use_cf and self.ready and len(known) >= MIN_CF_RATINGS

        cf = np.zeros(len(ids))
        has_cf = np.zeros(len(ids), dtype=bool)
        w = 0.0
        if cf_ready:
            items = np.array([i for i, _ in known])
            rats = np.array([r for _, r in known], dtype=float)
            b_u, p_vec = self.model.fold_in(items, rats)
            all_scores = self.model.score_all(b_u, p_vec)
            for j, t in enumerate(ids):
                idx = self.item_index.get(int(t))
                if idx is not None:
                    cf[j] = all_scores[idx]
                    has_cf[j] = True
            has_cf &= cand
            w = self.w_star * min(len(known) / float(RAMP_RATINGS), 1.0)
            if not has_content:
                w = 1.0

        if not has_content and not has_cf.any():
            return []

        content_pct = _pct(content, cand) if has_content else np.zeros(len(ids))
        final = content_pct.copy()
        if has_cf.any():
            cf_pct = _pct(cf, has_cf)
            final = np.where(has_cf, (1.0 - w) * content_pct + w * cf_pct, content_pct)

        order = np.argsort(-np.where(cand, final, -np.inf))[:top_n]
        results = []
        for j in order:
            if not cand[j]:
                continue
            row = df.iloc[int(j)]
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "poster_path": row["poster_path"],
                "vote_average": row["vote_average"],
                "recommendation_score": round(float(final[j]), 4),
            })
        return results


hybrid = HybridRecommender()