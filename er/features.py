"""C6 candidate union + C7 pair representation.

Feature groups (spec §2 C7): NAME, ADDR, HOUSE (number relation: the near-miss signal), RET (retriever
scores/ranks), META, CTX (competition among a record's / an S1's candidates). Country is deliberately
NOT a feature (France is unseen; E-14 tests adding it).
"""
import polars as pl
from rapidfuzz import fuzz, process

from . import retrieve, rules

GROUPS = {
    "NAME": ["nj", "n_inter", "nlen_r", "nlen_s", "n_tsr", "n_ratio", "n_collide"],
    "ADDR": ["aj", "sj", "a_tsr", "aempty", "alen_r", "alen_s"],
    "HOUSE": ["n1", "nr", "ncover", "nmiss", "nextra", "first_eq", "ndiff", "ntrunc"],
    "RET": ["sp_score", "sp_rank", "ex_hits"],  # fv=3 adds ch_score, ch_rank (R-char)
    "META": ["src"],
    "CTX": ["ncand_r", "ncand_s", "sp_rel", "nj_gap", "aj_gap"],
}
FEATS = [f for g in GROUPS.values() for f in g]
# fv=2 (E-07b): legal-form relation + name out-of-vocabulary share (rebrand vs same-address decoy)
GROUPS2 = {"LEGAL": ["leg_rel", "leg_add", "leg_del", "grp_add"], "OOV": ["oov"]}
FEATS2 = FEATS + [f for g in GROUPS2.values() for f in g]
FEATS3 = FEATS2 + ["ch_score", "ch_rank"]
# fv=4: best number relation over ALL S1 x record number pairs (prefix/suffix truncation, one-digit
# substitution, min |delta|) and name-token alignment (char similarity and OOV share of the unshared tokens:
# a typo is OOV and char-close, a swapped-in decoy word is in-vocabulary and char-far)
GROUPS4 = {"NUM2": ["n_prefix", "n_suffix", "n_subst1", "ndiff_min"], "ALIGN": ["uns_ratio", "uns_oov", "n_uns_s", "n_uns_r"]}
FEATS4 = FEATS2 + [f for g in GROUPS4.values() for f in g]


def candidates(cfg, s1v, rv):
    """Union of the configured retrievers -> rec_id, s1_id, sp_score, sp_rank, ex_hits (nulls = not retrieved)."""
    c = None
    if "sparse" in cfg["ret"]:
        c = retrieve.sparse(s1v, rv, cfg["topk"], cfg["cap"], q=cfg.get("q"), qn=cfg.get("qn")).rename({"score": "sp_score", "rank": "sp_rank"})
    if "exact" in cfg["ret"]:
        e = rules.block(s1v, rv, cfg["addr"], cfg.get("ex_topk", cfg["topk"])).rename({"hits": "ex_hits"})
        c = e if c is None else c.join(e, on=["rec_id", "s1_id"], how="full", coalesce=True)
    if "char" in cfg["ret"]:
        # R-char only for the weak stratum: records with an empty address or no candidate so far
        weak = rv.filter(pl.col("at").list.len() == 0)
        if c is not None:
            weak = pl.concat([weak, rv.join(c.select(id="rec_id").unique(), on="id", how="anti")]).unique("id")
        ch = retrieve.sparse(s1v, weak, cfg.get("ch_topk", 4), cfg.get("ch_cap", 300), tokens=lambda v: retrieve.chargrams(v, 5)) \
                     .rename({"score": "ch_score", "rank": "ch_rank"})
        c = ch if c is None else c.join(ch, on=["rec_id", "s1_id"], how="full", coalesce=True)
    for col, dt in (("sp_score", pl.Float32), ("sp_rank", pl.Int16), ("ex_hits", pl.UInt32),
                    ("ch_score", pl.Float32), ("ch_rank", pl.Int16)):
        if col not in c.columns:
            c = c.with_columns(pl.lit(None, dt).alias(col))
    if cfg.get("prune"):
        # rule-based pruning (no model): keep a candidate only if its TF-IDF score is within `prune` of the
        # record's best, or it has the record's most shared exact keys. 3.3x fewer candidates per S1.
        c = c.filter((pl.col("sp_score") >= cfg["prune"] * pl.col("sp_score").max().over("rec_id"))
                     | (pl.col("ex_hits") == pl.col("ex_hits").max().over("rec_id")))
    if cfg.get("s1cap"):
        # generic-name S1s ("hubs") attract thousands of records; a true S1 has at most 11 matches.
        # Keep each S1's `s1cap` best candidates by (relative TF-IDF score, shared exact keys).
        rel = (pl.col("sp_score") / pl.col("sp_score").max().over("rec_id")).fill_null(0)
        c = c.with_columns(_r=pl.struct(rel, pl.col("ex_hits").fill_null(0)).rank("ordinal", descending=True)
                              .over("s1_id")).filter(pl.col("_r") <= cfg["s1cap"]).drop("_r")
    return c


def _jacc(a, b):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None)


def _pairs(c, s1v, rv):
    r = rv.select(rec_id="id", src="src", nt="nt", nkey="nkey", st="st", at="at", nums="nums", lt="lt", oov="oov",
                  stkey=pl.col("st").list.sort().list.join(" "))
    s = s1v.with_columns(n_collide=pl.len().over("country", "nkey")).select(
        s1_id="id", nt_1="nt", nkey_1="nkey", st_1="st", at_1="at", nums_1="nums", n_collide="n_collide",
        lt_1="lt", stkey_1=pl.col("st").list.sort().list.join(" "))
    p = c.join(r, on="rec_id").join(s, on="s1_id")
    miss = pl.col("nums_1").list.set_difference(pl.col("nums"))
    extra = pl.col("nums").list.set_difference(pl.col("nums_1"))
    p = p.with_columns(
        nj=_jacc("nt", "nt_1"), n_inter=pl.col("nt").list.set_intersection(pl.col("nt_1")).list.len(),
        nlen_r=pl.col("nt").list.len(), nlen_s=pl.col("nt_1").list.len(),
        aj=_jacc("at", "at_1"), sj=_jacc("st", "st_1"),
        aempty=pl.col("at").list.len() == 0, alen_r=pl.col("at").list.len(), alen_s=pl.col("at_1").list.len(),
        n1=pl.col("nums_1").list.len(), nr=pl.col("nums").list.len(),
        ninter=pl.col("nums_1").list.set_intersection(pl.col("nums")).list.len(),
        first_eq=pl.col("nums_1").list.first() == pl.col("nums").list.first(),
        _m=miss.list.first(), _x=extra.list.first(),
    ).with_columns(
        ncover=pl.when(pl.col("n1") > 0).then(pl.col("ninter") / pl.col("n1")),
        nmiss=pl.col("n1") - pl.col("ninter"), nextra=pl.col("nr") - pl.col("ninter"),
        ndiff=(pl.col("_m") - pl.col("_x")).abs(),
        ntrunc=pl.col("_m").cast(pl.String).str.ends_with(pl.col("_x").cast(pl.String)),
        leg_add=pl.col("lt").list.set_difference(pl.col("lt_1")).list.len(),
        leg_del=pl.col("lt_1").list.set_difference(pl.col("lt")).list.len(),
        grp_add=pl.col("lt").list.contains("group") & ~pl.col("lt_1").list.contains("group"),
    ).with_columns(  # 0 same, 1 record has none, 2 S1 has none, 3 record subset, 4 record superset, 5 changed
        leg_rel=pl.when(pl.col("lt") == pl.col("lt_1")).then(0).when(pl.col("lt").list.len() == 0).then(1)
                  .when(pl.col("lt_1").list.len() == 0).then(2).when(pl.col("leg_add") == 0).then(3)
                  .when(pl.col("leg_del") == 0).then(4).otherwise(5),
    )
    # fv=4 number relations over all pairs of numbers (small lists: cross them via explode + aggregate)
    x = (p.select("rec_id", "s1_id", a=pl.col("nums_1"), b=pl.col("nums"))
           .explode("a", empty_as_null=True).explode("b", empty_as_null=True).drop_nulls()
           .with_columns(sa=pl.col("a").cast(pl.String), sb=pl.col("b").cast(pl.String)))
    x = x.with_columns(
        pre=(pl.col("sa").str.starts_with(pl.col("sb")) | pl.col("sb").str.starts_with(pl.col("sa"))) & (pl.col("a") != pl.col("b")),
        suf=(pl.col("sa").str.ends_with(pl.col("sb")) | pl.col("sb").str.ends_with(pl.col("sa"))) & (pl.col("a") != pl.col("b")),
        sub1=(pl.col("sa").str.len_chars() == pl.col("sb").str.len_chars()) & (pl.col("a") != pl.col("b"))
             & ((pl.col("sa").str.split("").list.set_symmetric_difference(pl.col("sb").str.split(""))).list.len() <= 2),
        d=(pl.col("a") - pl.col("b")).abs(),
    ).group_by("rec_id", "s1_id").agg(n_prefix=pl.col("pre").any(), n_suffix=pl.col("suf").any(),
                                      n_subst1=pl.col("sub1").any(), ndiff_min=pl.col("d").min())
    p = p.join(x, on=["rec_id", "s1_id"], how="left")
    uns_s = pl.col("nt_1").list.set_difference(pl.col("nt")); uns_r = pl.col("nt").list.set_difference(pl.col("nt_1"))
    p = p.with_columns(n_uns_s=uns_s.list.len(), n_uns_r=uns_r.list.len(),
                       uns_oov=uns_r.list.eval(~pl.element().is_in(VOC.implode())).list.mean(),
                       _us=uns_s.list.sort().list.join(" "), _ur=uns_r.list.sort().list.join(" "))
    kw = dict(workers=-1, dtype="float32")
    p = p.with_columns(
        uns_ratio=pl.Series(process.cpdist(p["_us"].to_list(), p["_ur"].to_list(), scorer=fuzz.ratio, **kw)),
        n_tsr=pl.Series(process.cpdist(p["nkey"].to_list(), p["nkey_1"].to_list(), scorer=fuzz.token_set_ratio, **kw)),
        n_ratio=pl.Series(process.cpdist(p["nkey"].to_list(), p["nkey_1"].to_list(), scorer=fuzz.ratio, **kw)),
        a_tsr=pl.Series(process.cpdist(p["stkey"].to_list(), p["stkey_1"].to_list(), scorer=fuzz.token_set_ratio, **kw)),
    )
    return p


def build(c, s1v, rv, path, truth=None, n=16, feats=FEATS):
    """Pair feature table (rec_id, s1_id, feats float32, y if truth) streamed to path/part-*.parquet
    one record-chunk at a time (holding all chunks in memory exceeded the RAM budget)."""
    keep = ["id", "country", "nt", "nkey", "st", "at", "nums", "lt"]
    s1v, rv = s1v.select(keep), rv.select(keep + ["src"])  # raw strings are not needed any more
    voc = s1v.select(pl.col("nt").explode(empty_as_null=True)).drop_nulls().unique().to_series()
    global VOC
    VOC = voc
    rv = rv.with_columns(oov=pl.col("nt").list.eval(~pl.element().is_in(voc.implode())).list.mean())
    c = c.with_columns(ncand_r=pl.len().over("rec_id"), ncand_s=pl.len().over("s1_id"),
                       sp_rel=pl.col("sp_score") / pl.col("sp_score").max().over("rec_id"))
    path.mkdir(parents=True, exist_ok=True)
    c = c.with_columns(_c=pl.col("rec_id").hash(seed=9) % n)
    for i in range(n):
        p = _pairs(c.filter(pl.col("_c") == i).drop("_c"), s1v, rv).with_columns(
            nj_gap=pl.col("nj") - pl.col("nj").max().over("rec_id"),
            aj_gap=pl.col("aj") - pl.col("aj").max().over("rec_id"),
        ).select("rec_id", "s1_id", *[pl.col(f).cast(pl.Float32) for f in feats])
        if truth is not None:
            p = p.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_id", "rec_id"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
        p.write_parquet(path / f"part-{i:02d}.parquet")


def read(path):
    """Pairs written by build() (a part directory) or by older runs (a single file)."""
    return pl.read_parquet(path / "*.parquet" if path.is_dir() else path)
