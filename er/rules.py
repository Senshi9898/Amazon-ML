"""R-exact blocking + rule scorer + record->argmax assignment, parameterised by the address variant.

Rule-based on purpose: E-01..E-03 measure *representations*; the learned scorer is E-07.
"""
import polars as pl

KEY_CAP = 50  # drop blocking keys shared by more than this many S1s
MAX_NUMS = 4  # A2: numbers per record used as blocking keys


def keys(v, addr):
    k1 = v.select("id", "country", key=pl.concat_str(pl.lit("n|"), "nkey")).filter(pl.col("key") != "n|")
    if addr == "A1":
        h = v.filter(pl.col("house").is_not_null()).select("id", "country", "house", "st", "nt")
    else:  # every number (the first one may be an injected 'DOOR NO <rand>')
        h = (v.select("id", "country", "st", "nt", house=pl.col("nums").list.head(MAX_NUMS).cast(pl.List(pl.String)))
              .explode("house", empty_as_null=True).drop_nulls("house"))
    k4 = h.explode("st", empty_as_null=True).select(
        "id", "country", key=pl.concat_str(pl.lit("h|"), "house", pl.lit("|"), "st"))
    k5 = h.explode("nt", empty_as_null=True).select(
        "id", "country", key=pl.concat_str(pl.lit("hn|"), "house", pl.lit("|"), "nt"))
    return pl.concat([k1, k4, k5]).drop_nulls().unique()


def block(s1v, rv, addr, topk=None, n=8):
    """Candidates (rec_id, s1_id, hits = number of shared keys). topk: keep the k S1s with most
    shared keys per record (C6 budget). Joined in record chunks to bound memory."""
    ks = keys(s1v, addr).filter(pl.len().over("country", "key") <= KEY_CAP)
    out = []
    for _, part in rv.with_columns(_c=pl.col("id").hash(seed=5) % n).group_by("_c"):
        g = keys(part, addr)  # per chunk: the string key table for all records at once dominates RAM
        c = (g.join(ks, on=["country", "key"], suffix="_s1")
              .group_by(rec_id="id", s1_id="id_s1").agg(hits=pl.len()))
        if topk:
            c = c.sort("hits", "s1_id", descending=[True, False]).group_by("rec_id").head(topk)
        out.append(c)
    return pl.concat(out)


def jacc(a, b):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0)


def score(cands, s1v, rv, addr):
    cols = ["id", "nt", "at", "house"] + (["nums"] if addr == "A2" else [])
    p = (cands.join(rv.select(cols).rename({"id": "rec_id"}), on="rec_id")
              .join(s1v.select(cols).rename({"id": "s1_id"}), on="s1_id", suffix="_1"))
    p = p.with_columns(nj=jacc("nt", "nt_1"), aj=jacc("at", "at_1"), aempty=pl.col("at").list.len() == 0)
    strong_addr = (pl.col("nj") >= 0.4) | (pl.col("aj") >= 0.6)
    if addr == "A1":
        heq = (pl.col("house") == pl.col("house_1")).fill_null(False)
        accept = (heq & strong_addr) | (pl.col("aempty") & (pl.col("nj") >= 0.8))
    else:
        # every S1 number must survive in the record (near-misses alter one); records that lost all
        # numbers need a strong name + address
        n1, n = pl.col("nums_1").list.len(), pl.col("nums").list.len()
        cover = (pl.col("nums_1").list.set_intersection(pl.col("nums")).list.len() == n1)
        accept = (((n1 == 0) | ((n > 0) & cover)) & strong_addr) \
            | (pl.col("aempty") & (pl.col("nj") >= 0.8)) \
            | (~pl.col("aempty") & (n == 0) & (n1 > 0) & (pl.col("nj") >= 0.8) & (pl.col("aj") >= 0.5))
    return p.filter(accept).with_columns(score=pl.col("nj") + pl.col("aj"))


def score_chunked(cands, s1v, rv, addr, n=8):
    # chunk by record: the joined list columns for all candidates at once exceed the RAM budget
    part = cands.with_columns(_c=pl.col("rec_id").hash(seed=3) % n)
    return pl.concat([score(g.drop("_c"), s1v, rv, addr) for _, g in part.group_by("_c")])


def assign(scored):
    """Each record -> its single best S1 (ties dropped: ambiguity is a false-merge risk)."""
    top = scored.with_columns(pl.col("score").rank("min", descending=True).over("rec_id").alias("r"))
    top = top.filter(pl.col("r") == 1).filter(pl.len().over("rec_id") == 1)
    return top.select("s1_id", "rec_id")
