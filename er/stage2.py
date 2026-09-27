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
# TOKR: positive rate (train labels) of the name tokens the record adds / drops relative to S1. The generator
# swaps in decoy words from a fixed vocabulary ("holdings", "industries", "ventures": 0% positive over 40k+
# pairs each) and corruption words from another ("center", "services": 35-45%); OOV share could not see this.
TOKR = ["tr_min", "tr_max", "ts_min", "ts_max", "tr_n"]
TOKA = ["ta_min", "ta_max", "tas_min", "tas_max"]  # same statistic on street tokens the record adds / drops
TOKP = ["tp_rate", "tp_n"]  # rate of the exact substitution (S1 tokens dropped -> record tokens added)
TOK_MIN_N = 5
SIDES = {"r": ("uns_r", "tr"), "s": ("uns_s", "ts"), "ar": ("uns_ar", "ta"), "as": ("uns_as", "tas"), "p": ("pair", "tp")}


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


def _uns(d, nts, ntr):
    """pairs + token lists -> tokens the record adds (uns_r) and drops (uns_s) relative to S1; the same for
    street tokens (uns_ar / uns_as) when the lists carry `st`; and the substitution key (pair) when at most
    two tokens change on each side."""
    u = (d.select("rec_id", "s1_id").join(ntr, on="rec_id").join(nts, on="s1_id")
           .with_columns(uns_r=pl.col("nt").list.set_difference("nt1"), uns_s=pl.col("nt1").list.set_difference("nt")))
    if "st" in u.columns:
        u = u.with_columns(uns_ar=pl.col("st").list.set_difference("st1"), uns_as=pl.col("st1").list.set_difference("st"))
    small = (pl.col("uns_r").list.len() <= 2) & (pl.col("uns_s").list.len() <= 2) & (pl.col("uns_r").list.len() + pl.col("uns_s").list.len() > 0)
    u = u.with_columns(pair=pl.when(small).then(pl.concat_list([pl.concat_str(
        pl.col("uns_s").list.sort().list.join(" "), pl.lit(">"), pl.col("uns_r").list.sort().list.join(" "))])))
    return u.select("rec_id", "s1_id", *[c for c in ("uns_r", "uns_s", "uns_ar", "uns_as", "pair") if c in u.columns])


def token_table(d, nts, ntr, k=5):
    """Rate tables from labelled pairs: (side, t, n, s) totals and per-fold totals (fold = rec_id hash % k),
    so that training rows can be scored out of fold. Returns a DataFrame[side, t, fold, n, s]."""
    u = _uns(d, nts, ntr).join(d.select("rec_id", "s1_id", "y"), on=["rec_id", "s1_id"])
    u = u.with_columns(fold=(pl.col("rec_id").hash(seed=17) % k).cast(pl.Int32))
    parts = [u.select("fold", "y", side=pl.lit(side), t=pl.col(col)).explode("t").drop_nulls("t")
             for side, (col, _) in SIDES.items() if col in u.columns]
    return pl.concat(parts).group_by("side", "t", "fold").agg(n=pl.len(), s=pl.col("y").sum())


def token_rates(d, nts, ntr, table, oof=False, k=5):
    """pairs -> TOKR columns. oof: exclude the row's own fold from the rates (training rows)."""
    u = _uns(d, nts, ntr).with_columns(fold=(pl.col("rec_id").hash(seed=17) % k).cast(pl.Int32))
    tot = table.group_by("side", "t").agg(N=pl.col("n").sum(), S=pl.col("s").sum())
    out = u.select("rec_id", "s1_id")
    for side, (col, pre) in SIDES.items():
        if col not in u.columns or side not in table["side"].unique().to_list():
            continue
        e = u.select("rec_id", "s1_id", "fold", t=pl.col(col)).explode("t").drop_nulls("t")
        e = e.join(tot.filter(pl.col("side") == side).drop("side"), on="t", how="left")
        if oof:
            e = e.join(table.filter(pl.col("side") == side).drop("side"), on=["t", "fold"], how="left") \
                 .with_columns(N=pl.col("N") - pl.col("n").fill_null(0), S=pl.col("S") - pl.col("s").fill_null(0))
        e = e.with_columns(rate=pl.when(pl.col("N") >= TOK_MIN_N).then(pl.col("S") / pl.col("N")))
        if pre == "tp":
            g = e.group_by("rec_id", "s1_id").agg(tp_rate=pl.col("rate").first(), tp_n=pl.col("N").first().fill_null(0))
        else:
            g = e.group_by("rec_id", "s1_id").agg(**{f"{pre}_min": pl.col("rate").min(), f"{pre}_max": pl.col("rate").max(),
                                                     **({"tr_n": pl.col("rate").is_not_null().sum()} if pre == "tr" else {})})
        out = out.join(g, on=["rec_id", "s1_id"], how="left")
    return d.join(out, on=["rec_id", "s1_id"], how="left")
