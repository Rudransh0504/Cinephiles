"""
Trains and evaluates the collaborative-filtering model on MovieLens, compares it
against baselines and the content-based recommender, then saves the final model.

Run from backend/ with the venv active:
    python -m scripts.evaluate_cf
"""

import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from app.database import SessionLocal
from app.recommender import recommender
from app.mf import ALSRecommender

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BACKEND_DIR, "data", "ml-latest-small")
ARTIFACT_DIR = os.path.join(BACKEND_DIR, "artifacts")

SEED = 42
LIKE_THRESHOLD = 3.5   # 3.5 stars on MovieLens = 7/10 in the app
MIN_RATINGS = 20       # users need at least this many ratings to be evaluated
TOP_K = 10
K_FACTORS = 20
REG_GRID = [0.05, 0.15, 0.5, 1.5]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  # blend weight on CF (0 = content only, 1 = CF only)


# ---------------------------------------------------------------- data
def load_movielens():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    links = pd.read_csv(os.path.join(DATA_DIR, "links.csv")).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    links = links.drop_duplicates(subset=["tmdbId"])
    merged = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")
    return merged[["userId", "tmdbId", "rating"]].copy()


def split_user(items, rats, rng, frac=0.2):
    idx = rng.permutation(len(items))
    n_test = max(1, int(round(frac * len(items))))
    te, tr = idx[:n_test], idx[n_test:]
    return items[tr], rats[tr], items[te], rats[te]


def make_splits(user_array, groups, rng):
    splits = []
    for u in sorted(int(x) for x in user_array):
        items, rats = groups[u]
        if len(items) >= MIN_RATINGS:
            splits.append(split_user(items, rats, rng))
    return splits


# ---------------------------------------------------------------- rating prediction
def rmse_mae(model, splits, mode):
    errs = []
    for tr_i, tr_r, te_i, te_r in splits:
        if mode == "global":
            pred = np.full(len(te_i), model.mu)
        else:
            b_u, p = model.fold_in(tr_i, tr_r)
            if mode == "bias":
                pred = np.clip(model.mu + b_u + model.item_bias[te_i], 0.5, 5.0)
            else:
                pred = model.predict_for_items(b_u, p, te_i)
        errs.append(pred - te_r)
    e = np.concatenate(errs)
    return float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean())


# ---------------------------------------------------------------- ranking
def pct_rank(x):
    n = len(x)
    if n < 2:
        return np.zeros(n)
    return np.argsort(np.argsort(x)) / (n - 1.0)


def ranking_eval(splits, model, cat_item, cat_pos, sim, pop_counts):
    names = ["popularity", "content", "cf"] + ["hybrid_w%.2f" % w for w in W_GRID]
    acc = {n: {"precision": [], "recall": [], "hit_rate": []} for n in names}

    for tr_i, tr_r, te_i, te_r in splits:
        rated = dict(zip(tr_i.tolist(), tr_r.tolist()))
        r_vec = np.array([rated.get(int(x), 0.0) for x in cat_item])
        liked_mask = r_vec >= LIKE_THRESHOLD
        cand = ~np.isin(cat_item, tr_i)
        relevant = np.isin(cat_item, te_i[te_r >= LIKE_THRESHOLD]) & cand

        if liked_mask.sum() < 3 or relevant.sum() < 1 or cand.sum() <= TOP_K:
            continue

        # content-based: similarity to the movies this user liked, weighted by rating
        S = sim[np.ix_(cat_pos[liked_mask], cat_pos)]
        content = (r_vec[liked_mask] / 5.0) @ S

        # collaborative filtering: fold the user in, score every catalog movie
        b_u, p = model.fold_in(tr_i, tr_r)
        cf = model.score_all(b_u, p)[cat_item]

        pop = pop_counts[cat_item].astype(float)

        c_pct = pct_rank(content[cand])
        f_pct = pct_rank(cf[cand])

        scores = {"popularity": pop[cand], "content": content[cand], "cf": cf[cand]}
        for w in W_GRID:
            scores["hybrid_w%.2f" % w] = (1 - w) * c_pct + w * f_pct

        rel_c = relevant[cand]
        for name, s in scores.items():
            top = np.argsort(-s)[:TOP_K]
            hits = int(rel_c[top].sum())
            acc[name]["precision"].append(hits / float(TOP_K))
            acc[name]["recall"].append(hits / float(rel_c.sum()))
            acc[name]["hit_rate"].append(1.0 if hits > 0 else 0.0)

    n_users = len(acc["popularity"]["precision"])
    result = {n: {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()} for n, d in acc.items()}
    return result, n_users


def print_table(title, res, n_users):
    print("\n%s (%d users)" % (title, n_users))
    print("%-18s %10s %12s %12s" % ("method", "P@%d" % TOP_K, "Recall@%d" % TOP_K, "HitRate@%d" % TOP_K))
    for name, m in res.items():
        print("%-18s %10.4f %12.4f %12.4f" % (name, m["precision"], m["recall"], m["hit_rate"]))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        print("Could not find MovieLens files in %s" % DATA_DIR)
        print("Put the unzipped ml-latest-small folder at backend/data/ml-latest-small/")
        return

    ratings = load_movielens()
    item_ids = np.sort(ratings["tmdbId"].unique())
    item_index = {int(t): i for i, t in enumerate(item_ids)}
    user_ids = np.sort(ratings["userId"].unique())
    user_index = {int(u): i for i, u in enumerate(user_ids)}

    ratings["u"] = ratings["userId"].map(user_index).astype(int)
    ratings["i"] = ratings["tmdbId"].map(item_index).astype(int)
    n_users, n_items = len(user_ids), len(item_ids)
    print("MovieLens: %d users, %d movies (with TMDB ids), %d ratings" % (n_users, n_items, len(ratings)))

    # ---- split users into train / validation / test
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_users)
    n_test, n_val = int(0.2 * n_users), int(0.1 * n_users)
    test_users = perm[:n_test]
    val_users = perm[n_test:n_test + n_val]
    train_users = perm[n_test + n_val:]

    u_arr = ratings["u"].to_numpy()
    i_arr = ratings["i"].to_numpy()
    r_arr = ratings["rating"].to_numpy(dtype=float)

    train_mask = np.isin(u_arr, train_users)
    tr_u, tr_i, tr_r = u_arr[train_mask], i_arr[train_mask], r_arr[train_mask]

    groups = {}
    for u, g in ratings.groupby("u"):
        groups[int(u)] = (g["i"].to_numpy(), g["rating"].to_numpy(dtype=float))

    split_rng = np.random.RandomState(SEED + 1)
    val_splits = make_splits(val_users, groups, split_rng)
    test_splits = make_splits(test_users, groups, split_rng)
    print("Users: %d train / %d validation / %d test" % (len(train_users), len(val_splits), len(test_splits)))

    # ---- choose regularization on validation users
    print("\nTuning regularization (k=%d)..." % K_FACTORS)
    best = None
    for reg in REG_GRID:
        m = ALSRecommender(k=K_FACTORS, reg=reg).fit(tr_u, tr_i, tr_r, n_users, n_items)
        v_rmse, _ = rmse_mae(m, val_splits, "mf")
        print("  reg=%.2f -> validation RMSE %.4f" % (reg, v_rmse))
        if best is None or v_rmse < best["rmse"]:
            best = {"rmse": v_rmse, "reg": reg, "model": m}
    model = best["model"]
    print("Best reg: %.2f" % best["reg"])

    # ---- rating prediction on test users
    report = {"k": K_FACTORS, "reg": best["reg"], "rmse": {}}
    print("\nRating prediction on held-out test ratings")
    for label, mode in [("global mean", "global"), ("biases only", "bias"), ("matrix factorization", "mf")]:
        rm, ma = rmse_mae(model, test_splits, mode)
        report["rmse"][label] = {"rmse": rm, "mae": ma}
        print("  %-22s RMSE %.4f   MAE %.4f" % (label, rm, ma))

    # ---- ranking evaluation against the content-based recommender
    w_star = 0.5
    db = SessionLocal()
    try:
        recommender.build(db)
    finally:
        db.close()

    df = recommender.df
    sim = recommender.similarity
    catalog_pos = {int(t): j for j, t in enumerate(df["id"])}
    cat_tmdb = [int(t) for t in df["id"] if int(t) in item_index]
    print("\nCatalog: %d seeded movies, %d of them exist in MovieLens" % (len(df), len(cat_tmdb)))

    if len(cat_tmdb) >= 30:
        cat_item = np.array([item_index[t] for t in cat_tmdb])
        cat_pos = np.array([catalog_pos[t] for t in cat_tmdb])
        pop_counts = np.bincount(tr_i, minlength=n_items)

        val_res, n_val_used = ranking_eval(val_splits, model, cat_item, cat_pos, sim, pop_counts)
        print_table("Validation ranking (used to choose the blend weight)", val_res, n_val_used)

        w_star = max(W_GRID, key=lambda w: val_res["hybrid_w%.2f" % w]["precision"])
        print("\nChosen blend weight on CF: %.2f" % w_star)

        test_res, n_test_used = ranking_eval(test_splits, model, cat_item, cat_pos, sim, pop_counts)
        shown = {k: v for k, v in test_res.items() if k in ("popularity", "content", "cf", "hybrid_w%.2f" % w_star)}
        print_table("TEST ranking (final numbers)", shown, n_test_used)

        report["blend_weight_cf"] = w_star
        report["ranking_test"] = shown
        report["ranking_users"] = n_test_used
        report["catalog_size"] = len(cat_tmdb)
    else:
        print("Catalog overlap too small for ranking evaluation. Re-run the seed script with more pages.")

    # ---- final model: retrain on every MovieLens user and save
    print("\nTraining final model on all users...")
    final = ALSRecommender(k=K_FACTORS, reg=best["reg"]).fit(u_arr, i_arr, r_arr, n_users, n_items)
    final.save(os.path.join(ARTIFACT_DIR, "cf_model.npz"), item_ids, w_star=w_star)

    with open(os.path.join(ARTIFACT_DIR, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Saved artifacts/cf_model.npz and artifacts/eval_report.json")
    print("Done in %.0f seconds." % (time.time() - t0))


if __name__ == "__main__":
    main()