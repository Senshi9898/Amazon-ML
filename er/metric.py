"""Competition metric (macro F0.5 per S1, singletons included) + blocking metrics.

Per S1: F0.5 = 1.25*tp / (0.25*n_true + n_pred); an S1 with n_true = n_pred = 0 scores 1.
(Algebraically identical to the README formula 1.25PR/(0.25P+R).)
"""
import numpy as np
import polars as pl


def per_s1(s1, truth, pred):
    """s1: DataFrame[s1_id, (country)]; truth/pred: DataFrame[s1_id, rec_id]. Returns s1 with n_true, n_pred, tp, f05."""
    def count(df, name):
        return df.unique(["s1_id", "rec_id"]).group_by("s1_id").len(name)
    tp = truth.join(pred, on=["s1_id", "rec_id"]).unique().group_by("s1_id").len("tp")
    out = (s1.join(count(truth, "n_true"), on="s1_id", how="left")
             .join(count(pred, "n_pred"), on="s1_id", how="left")
             .join(tp, on="s1_id", how="left")
             .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0)))
    return out.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
          .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
          .alias("f05"))


def bootstrap_ci(values, n=1000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(s1, truth, pred, ci=True):
    """Full metric report. s1 must contain s1_id and country."""
    r = per_s1(s1, truth, pred)
    tot = r.select(pl.col("tp", "n_true", "n_pred").sum()).row(0)
    out = {
        "n_s1": r.height,
        "macro_f05": r["f05"].mean(),
        "precision": tot[0] / max(tot[2], 1),  # pair-level (micro) diagnostics
        "recall": tot[0] / max(tot[1], 1),
        "f05_singletons": r.filter(pl.col("n_true") == 0)["f05"].mean(),
    }
    for c, g in r.group_by("country"):
        out[f"f05_{c[0].lower()}"] = g["f05"].mean()
    if ci:
        out["f05_ci_lo"], out["f05_ci_hi"] = bootstrap_ci(r["f05"].to_numpy())
    return out


def blocking(s1, truth, cands, strata=None):
    """Recall of truth pairs in cands, volume, and oracle F0.5 (perfect scorer on these cands).
    strata: optional DataFrame[rec_id, stratum] for stratified recall."""
    c = cands.select("s1_id", "rec_id").unique()
    hit =truth.join(c.with_columns(pl.lit(True).alias("hit")), on=["s1_id", "rec_id"], how="left") \
               .with_columns(pl.col("hit").fill_null(False))
    out = {
        "n_cands": c.height,
        "cands_per_s1": c.height / max(s1.height, 1),
        "cands_per_rec": c.height / max(c["rec_id"].n_unique(), 1),
        "blocking_recall": hit["hit"].mean(),
        "oracle_f05": per_s1(s1, truth, hit.filter("hit").select("s1_id", "rec_id"))["f05"].mean(),
    }
    if strata is not None:
        for (k,), g in hit.join(strata, on="rec_id", how="left").group_by("stratum"):
            out[f"recall_{k}"] = g["hit"].mean()
    return out


def _selfcheck():
    s1 = pl.DataFrame({"s1_id": ["a", "b", "c", "d"], "country": ["US"] * 4})
    truth = pl.DataFrame({"s1_id": ["a", "a", "d"], "rec_id": ["S2-47", "S3-812", "S2-9"]})
    pred = pl.DataFrame({"s1_id": ["a", "a", "a", "c"], "rec_id": ["S2-47", "S2-193", "S3-812", "S2-1"]})
    f = per_s1(s1, truth, pred).sort("s1_id")["f05"].to_list()
    # README example = 0.714; b singleton empty = 1; c singleton predicted = 0; d missed = 0
    assert abs(f[0] - 0.7142857) < 1e-6 and f[1:] == [1.0, 0.0, 0.0], f
    b = blocking(s1, truth, pred)
    # oracle keeps only true hits: a=1, b=1, c=1 (wrong cand dropped), d=0 (never retrieved)
    assert abs(b["blocking_recall"] - 2 / 3) < 1e-9 and abs(b["oracle_f05"] - 0.75) < 1e-9, b
    print("metric selfcheck OK", f)


if __name__ == "__main__":
    _selfcheck()
