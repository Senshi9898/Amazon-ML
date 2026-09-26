"""C11 entity-level inference as second-stage features on stage-1 scores (spec E-10, E-11).

Input: pairs (rec_id, s1_id, feats...) + stage-1 p (out-of-fold on DEV-TRAIN, full model on DEV-VAL).
CTX2  : record competition (runner-up p, margin, rank)
SIB   : within-source siblings: other records of the same source + same normalised address that score
        high for the same S1 (the generator copies one corrupted base address per source)     (E-10)
XSRC  : support for the S1 from the *other* source vs the same source                          (E-11)
"""
import polars as pl

from .views import _fold

HI = 0.5
FLOOR = 0.01  # stage-1 score below which a pair is rejected without stage 2
CTX2 = ["p1", "p_r2", "p_margin", "p_rank"]
SIB = ["sib_n", "sib_max"]
XSRC = ["s_hi_same", "s_hi_other", "s_max_other", "s_sum_other"]


def addr_sig(recs):
    """rec_id, sig = hash of sorted unique alnum address tokens (empty address -> null)."""
    return recs.select(rec_id="id", sig=pl.when(pl.col("addr").str.strip_chars() != "").then(
        _fold(pl.col("addr")).str.extract_all(r"[a-z0-9]+").list.unique().list.sort().list.join(" ").hash(seed=1)))


def context(pairs, p, sig):
    """pairs + p (rec_id, s1_id, p) + sig -> pairs with CTX2 + SIB + XSRC columns."""
    d = pairs.join(p, on=["rec_id", "s1_id"]).join(sig, on="rec_id").rename({"p": "p1"})
    d = d.with_columns(
        p_rank=pl.col("p1").rank("ordinal", descending=True).over("rec_id"),
        p_r2=pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("rec_id").fill_null(0.0),
    ).with_columns(p_margin=pl.col("p1") - pl.when(pl.col("p_rank") == 1).then("p_r2")
                   .otherwise(pl.col("p1").max().over("rec_id")))
    hi = (pl.col("p1") >= HI).cast(pl.Int32)
    g = ("s1_id", "src", "sig")
    mx = {k: pl.col("p1").filter(pl.col("src") == k).max().over("s1_id").fill_null(0.0) for k in (2, 3)}
    d = d.with_columns(  # S1 support by source, excluding the record itself
        _hs=hi.sum().over("s1_id", "src"), _ha=hi.sum().over("s1_id"),
        _ss=pl.col("p1").sum().over("s1_id", "src"), _sa=pl.col("p1").sum().over("s1_id"),
        s_max_other=pl.when(pl.col("src") == 2).then(mx[3]).otherwise(mx[2]),
        # siblings: same S1, same source, same address signature (excluding self)
        _gn=hi.sum().over(*g), _g1=pl.col("p1").max().over(*g),
        _g2=pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over(*g),
    )
    return d.with_columns(
        s_hi_same=pl.col("_hs") - hi, s_hi_other=pl.col("_ha") - pl.col("_hs"),
        s_sum_other=pl.col("_sa") - pl.col("_ss"),
        sib_n=pl.when(pl.col("sig").is_not_null()).then(pl.col("_gn") - hi),
        sib_max=pl.when(pl.col("sig").is_not_null()).then(
            pl.when(pl.col("p1") >= pl.col("_g1")).then("_g2").otherwise("_g1")),
    ).drop("_hs", "_ha", "_ss", "_sa", "_gn", "_g1", "_g2", "sig")
