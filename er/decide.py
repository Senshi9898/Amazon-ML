"""C10-C12 decision layer on scored candidate pairs (rec_id, s1_id, p).

global τ  : record -> argmax S1 if p >= τ                                  (E-07 default)
margin    : ... and p - p_second_best >= margin                            (E-09)
ef05      : per S1, accept the prefix of its argmax records (sorted by p) that maximises expected
            F0.5 ≈ 1.25·Σ_{i<=k} p_i / (0.25·T + k), T = Σ p over all its candidates; k = 0 scores
            P(no true match) = Π(1 - p)                                      (E-12)
isotonic  : calibrate p on out-of-fold DEV-TRAIN scores                    (E-13)
"""
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .metric import per_s1


def record_best(pred):
    """Per record: argmax S1, its p, and the runner-up p2 (0 if none)."""
    return (pred.sort("p", descending=True).group_by("rec_id", maintain_order=True)
                .agg(s1_id=pl.col("s1_id").first(), p=pl.col("p").first(),
                     p2=pl.col("p").get(1, null_on_oob=True))
                .with_columns(pl.col("p2").fill_null(0.0)))


def margin(pred, tau, m):
    b = record_best(pred)
    return b.filter((pl.col("p") >= tau) & (pl.col("p") - pl.col("p2") >= m)).select("s1_id", "rec_id")


def ef05(pred, floor=0.0):
    tot = pred.group_by("s1_id").agg(T=pl.col("p").sum(),
                                     P0=(1 - pl.col("p")).clip(1e-9, 1).log().sum().exp())
    b = (record_best(pred).filter(pl.col("p") >= floor).sort("p", descending=True)
            .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_id"),
                          cum=pl.col("p").cum_sum().over("s1_id"))
            .join(tot, on="s1_id")
            .with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("T") + pl.col("k"))))
    best = b.group_by("s1_id").agg(kbest=pl.col("k").get(pl.col("ef").arg_max()), efbest=pl.col("ef").max(),
                                   P0=pl.col("P0").first())
    keep = best.filter(pl.col("efbest") > pl.col("P0")).select("s1_id", "kbest")
    return b.join(keep, on="s1_id").filter(pl.col("k") <= pl.col("kbest")).select("s1_id", "rec_id")


def calibrate(oof, truth, pred):
    """Isotonic map fitted on out-of-fold DEV-TRAIN scores, applied to pred."""
    y = oof.join(truth.with_columns(y=pl.lit(1)), on=["s1_id", "rec_id"], how="left")["y"].fill_null(0)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof["p"].to_numpy(), y.to_numpy())
    return pred.with_columns(p=pl.Series(iso.predict(pred["p"].to_numpy()), dtype=pl.Float32))


def tune(fn, grid, pred, s1e, truth):
    """Pick the params (dict) maximising macro F0.5 of fn(pred, **params)."""
    res = {tuple(g.items()): per_s1(s1e, truth, fn(pred, **g))["f05"].mean() for g in grid}
    best = max(res, key=res.get)
    return dict(best), res
