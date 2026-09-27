"""C8 learned scorer (LightGBM) + threshold selection on out-of-fold DEV-TRAIN scores + assignment."""
import lightgbm as lgb
import numpy as np
import polars as pl

from .metric import per_s1

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=12)
ROUNDS = 400


def fit(p, feats, seed=0):
    w = p["w"].to_numpy() if "w" in p.columns else None
    ds = lgb.Dataset(p.select(feats).to_numpy(), label=p["y"].to_numpy(), weight=w, feature_name=feats)
    return lgb.train({**PARAMS, "seed": seed}, ds, ROUNDS)


def predict(model, p, feats):
    return p.select("rec_id", "s1_id").with_columns(
        p=pl.Series(model.predict(p.select(feats).to_numpy(), num_threads=12), dtype=pl.Float32))


def oof(p, feats, k=2):
    """Out-of-fold scores, folds by record."""
    fold = (p["rec_id"].hash(seed=13) % k).to_numpy()
    out = []
    for f in range(k):
        m = fit(p.filter(pl.Series(fold != f)), feats)
        out.append(predict(m, p.filter(pl.Series(fold == f)), feats))
    return pl.concat(out)


def assign(pred, tau):
    """Each record -> its argmax S1 if p >= tau (records belong to at most one S1)."""
    return (pred.filter(pl.col("p") >= tau).sort("p", descending=True)
                .unique("rec_id", keep="first").select("s1_id", "rec_id"))


def choose_tau(pred, s1e, truth, grid=np.arange(0.05, 0.96, 0.05)):
    scores = {round(float(t), 2): per_s1(s1e, truth, assign(pred, t))["f05"].mean() for t in grid}
    best = max(scores, key=scores.get)
    return best, scores
