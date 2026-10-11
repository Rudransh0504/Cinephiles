"""
Matrix factorization recommender: damped biases + latent factors learned with
Alternating Least Squares (ALS with weighted regularization).

    prediction(u, i) = mu + b_u + b_i + p_u . q_i

New users who were not in the training data are handled by "fold-in": the item
factors stay fixed and we solve for that one user's bias and factors in closed form.
"""

import numpy as np


def _group(rows, cols, vals, n_rows):
    """Group (cols, vals) by row index into CSR-style arrays (ptr, cols, vals)."""
    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]
    ptr = np.searchsorted(rows, np.arange(n_rows + 1))
    return ptr, cols, vals


class ALSRecommender:
    def __init__(self, k=20, reg=0.1, bias_reg=10.0, n_iters=12, seed=42):
        self.k = k
        self.reg = reg
        self.bias_reg = bias_reg
        self.n_iters = n_iters
        self.seed = seed

        self.mu = 0.0
        self.item_bias = None
        self.item_factors = None
        self.n_items = 0

    # ---------- training ----------
    def fit(self, user_idx, item_idx, ratings, n_users, n_items):
        user_idx = np.asarray(user_idx)
        item_idx = np.asarray(item_idx)
        ratings = np.asarray(ratings, dtype=np.float64)
        self.n_items = n_items

        self.mu = float(ratings.mean())

        # damped item bias, then damped user bias
        item_sum = np.bincount(item_idx, weights=ratings - self.mu, minlength=n_items)
        item_cnt = np.bincount(item_idx, minlength=n_items)
        self.item_bias = item_sum / (item_cnt + self.bias_reg)

        resid = ratings - self.mu - self.item_bias[item_idx]
        user_sum = np.bincount(user_idx, weights=resid, minlength=n_users)
        user_cnt = np.bincount(user_idx, minlength=n_users)
        user_bias = user_sum / (user_cnt + self.bias_reg)

        # what the latent factors have to explain
        resid = resid - user_bias[user_idx]

        by_user = _group(user_idx, item_idx, resid, n_users)
        by_item = _group(item_idx, user_idx, resid, n_items)

        rng = np.random.RandomState(self.seed)
        Q = rng.normal(0, 0.1, size=(n_items, self.k))
        P = np.zeros((n_users, self.k))

        for _ in range(self.n_iters):
            self._als_step(P, Q, by_user)  # update users, items fixed
            self._als_step(Q, P, by_item)  # update items, users fixed

        Q[item_cnt == 0] = 0.0  # items never rated in training carry no signal
        self.item_factors = Q
        return self

    def _als_step(self, target, fixed, grouped):
        ptr, nbr, vals = grouped
        eye = np.eye(self.k)
        for t in range(len(ptr) - 1):
            lo, hi = ptr[t], ptr[t + 1]
            n = hi - lo
            if n == 0:
                continue
            F = fixed[nbr[lo:hi]]
            A = F.T @ F + self.reg * n * eye
            target[t] = np.linalg.solve(A, F.T @ vals[lo:hi])

    # ---------- inference ----------
    def fold_in(self, item_idx, ratings):
        """Estimate (user_bias, user_factors) for a user who is not in the training set."""
        item_idx = np.asarray(item_idx)
        r = np.asarray(ratings, dtype=np.float64)
        resid = r - self.mu - self.item_bias[item_idx]
        b_u = resid.sum() / (len(r) + self.bias_reg)
        resid = resid - b_u
        F = self.item_factors[item_idx]
        A = F.T @ F + self.reg * len(r) * np.eye(self.k)
        p = np.linalg.solve(A, F.T @ resid)
        return b_u, p

    def score_all(self, b_u, p):
        return self.mu + b_u + self.item_bias + self.item_factors @ p

    def predict_for_items(self, b_u, p, item_idx):
        s = self.mu + b_u + self.item_bias[item_idx] + self.item_factors[item_idx] @ p
        return np.clip(s, 0.5, 5.0)

    # ---------- persistence ----------
    def save(self, path, item_ids, w_star=0.5):
        np.savez(
            path,
            mu=np.array(self.mu),
            item_bias=self.item_bias,
            item_factors=self.item_factors,
            item_ids=np.asarray(item_ids),
            k=np.array(self.k),
            reg=np.array(self.reg),
            bias_reg=np.array(self.bias_reg),
            w_star=np.array(w_star),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        model = cls(k=int(d["k"]), reg=float(d["reg"]), bias_reg=float(d["bias_reg"]))
        model.mu = float(d["mu"])
        model.item_bias = d["item_bias"]
        model.item_factors = d["item_factors"]
        model.n_items = len(model.item_bias)
        meta = {"w_star": float(d["w_star"])}
        return model, d["item_ids"], meta