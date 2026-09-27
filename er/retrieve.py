"""C5 retrievers: record -> top-k S1 within a country (spec §2 C5 contract: rec_id, s1_id, score, rank).

R-sparse: TF-IDF over a bag of typed tokens (name, address word, number, house+street, house+name),
IDF from the S1 side, postings with df > cap dropped, score = sum(idf^2 over shared tokens) / |s1|.
Implemented as a chunked inverted-index join in polars (no dense or scipy intermediates).
"""
import polars as pl

MAX_NUMS = 4


def tokens(v):
    """-> (id, country, tok). Needs nt, st, nums (A2 view)."""
    nums = pl.col("nums").list.head(MAX_NUMS).cast(pl.List(pl.String))
    base = v.select("id", "country", "nt", "st", nums=nums)
    parts = [
        base.select("id", "country", tok=pl.col("nt").list.eval(pl.concat_str(pl.lit("n:"), pl.element()))).explode("tok", empty_as_null=True),
        base.select("id", "country", tok=pl.col("st").list.eval(pl.concat_str(pl.lit("a:"), pl.element()))).explode("tok", empty_as_null=True),
        base.select("id", "country", tok=pl.col("nums").list.eval(pl.concat_str(pl.lit("#:"), pl.element()))).explode("tok", empty_as_null=True),
    ]
    h = base.explode("nums", empty_as_null=True).drop_nulls("nums")
    parts += [
        h.explode("st", empty_as_null=True).select(
            "id", "country", tok=pl.concat_str(pl.lit("h:"), "nums", pl.lit("|"), "st")),
        h.explode("nt", empty_as_null=True).select(
            "id", "country", tok=pl.concat_str(pl.lit("hn:"), "nums", pl.lit("|"), "nt")),
    ]
    return pl.concat(parts).drop_nulls().unique()


def _int_id(col="id"):
    # 'S2-123' -> 2*10^10 + 123 (numeric parts of S2 and S3 ids overlap)
    return (pl.col(col).str.slice(1, 1).cast(pl.Int64) * 10**10 + pl.col(col).str.slice(3).cast(pl.Int64))


def chargrams(v, k=4, maxlen=48):
    """-> (id, country, tok): padded character k-grams of the sorted name key (R-char, spec C5).
    Fixed-offset slices (str.slice with a per-row offset allocates pathologically in polars 1.44)."""
    key = pl.concat_str(pl.lit("_"), pl.col("nkey").str.replace_all(" ", "_"), pl.lit("_")).str.slice(0, maxlen)
    base = v.select("id", "country", key=key)
    cols = [pl.col("key").str.slice(i, k).alias(str(i)) for i in range(maxlen - k + 1)]
    return (base.with_columns(cols).drop("key")
                .unpivot(index=["id", "country"], value_name="tok").drop("variable")
                .filter(pl.col("tok").str.len_chars() == k).unique())


NAME_CAP = 500  # df bound for the guaranteed name tokens (join volume: a generic name token is no key anyway)


def sparse(s1v, rv, topk=8, cap=100, n=16, tokens=tokens, q=None, qn=None):
    """q: keep only each record's q rarest tokens (by S1 df) as query terms; bounds join volume so a much
    higher df cap can be used (cap 3000 / q 6 = 408 postings per record vs 131 at cap 100 / all tokens).
    qn: guaranteed quota for the record's qn rarest *name* tokens. The compound house+street / house+name
    tokens are always the rarest, so without it a corrupted house number empties the whole query (6.7k
    DEV-VAL positives with an exact name match were never retrieved)."""
    ts = tokens(s1v)
    voc = (ts.group_by("country", "tok").agg(df=pl.len())
             .join(ts.group_by("country").agg(N=pl.col("id").n_unique()), on="country")
             .filter(pl.col("df") <= cap)
             .with_columns(w2=(pl.col("N") / pl.col("df")).log() ** 2)
             .with_row_index("tid").select("country", "tok", "tid", "w2"))
    post = ts.join(voc, on=["country", "tok"]).select("tid", "w2", s1=_int_id())
    norm = post.group_by("s1").agg(ns=pl.col("w2").sum().sqrt())
    post = post.select("tid", "s1")
    w = voc.select("tid", "w2")
    vk = voc.select("country", "tok", "tid").join(ts.group_by("country", "tok").agg(df=pl.len()), on=["country", "tok"])
    out = []
    n = max(n, -(-rv.height // 25_000))  # chunks of <= 25k records: the query x postings join is the peak
    for _, part in rv.with_columns(_c=pl.col("id").hash(seed=5) % n).group_by("_c"):
        # tokenise per chunk: the string token table for all records at once dominates peak RAM
        g = tokens(part).join(vk, on=["country", "tok"])
        if q:
            g = g.sort("df")
            g = pl.concat([g.group_by("id").head(q)] + ([g.filter(pl.col("tok").str.starts_with("n:") & (pl.col("df") <= NAME_CAP))
                                                        .group_by("id").head(qn)] if qn else [])).unique()
        g = g.select("tid", rec=_int_id())
        if g.height == 0:
            continue
        c = (g.join(post, on="tid").join(w, on="tid")
              .group_by("rec", "s1").agg(dot=pl.col("w2").sum())
              .join(norm, on="s1")
              .with_columns(score=(pl.col("dot") / pl.col("ns")).cast(pl.Float32))
              .sort("score", "s1", descending=[True, False]).group_by("rec").head(topk)
              .with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("rec").cast(pl.Int16))
              .select("rec", "s1", "score", "rank"))
        out.append(c)
    ids = s1v.select(s1_id="id", s1=_int_id())
    rids = rv.select(rec_id="id", rec=_int_id())
    return (pl.concat(out).join(rids, on="rec").join(ids, on="s1")
              .select("rec_id", "s1_id", "score", "rank"))
