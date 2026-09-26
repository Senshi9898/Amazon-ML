"""Reproducible forensic audits behind the whitepaper / RESEARCH_SPEC facts (F1-F12).

    python -m er.forensics            # all audits on the training labels (~5 min, < 4 GB)

Each function prints the numbers quoted in the documents. Sampling uses fixed hashes so
reruns are identical.
"""
import re
import unicodedata

import polars as pl

from .data import load, load_gt, load_recs
from .views import DROP, _fold

INDIC = r"[\x{0900}-\x{0DFF}]"


def _toks(col):
    return _fold(pl.col(col)).str.extract_all(r"[a-z0-9]+")


def _ntoks(col):
    return _toks(col).list.set_difference(pl.lit(DROP))


def _nums(col):
    return _fold(pl.col(col)).str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64)).list.unique()


def owner_audit():
    """F1: each S2/S3 record belongs to at most one S1; match-count distribution; singleton share."""
    gt = load_gt()
    multi = gt.group_by("rec_id").len().filter(pl.col("len") > 1).height
    s1 = load("train", 1).select(s1_id="id")
    k = s1.join(gt.group_by("s1_id").len("k"), on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    print(f"[F1] labels {gt.height:,}  distinct records {gt['rec_id'].n_unique():,}  records with >1 owner: {multi}")
    print(f"[F1] S1 {s1.height:,}  singletons {(k['k'] == 0).mean():.3%}  matches per S1: max {k['k'].max()}, "
          f"mode {k.group_by('k').len().sort('len', descending=True)['k'][0]}")
    return gt


def distractor_share(gt):
    """F2 (part): share of S2/S3 records that match no S1."""
    for src in (2, 3):
        r = load("train", src).select("id")
        un = r.join(gt.select(id="rec_id"), on="id", how="anti").height
        print(f"[F2] source {src}: {un:,} of {r.height:,} records unmatched ({un / r.height:.1%})")


def country_agreement(gt, frac=0.01):
    """F5: country of a record always equals the country of its S1."""
    s1 = load("train", 1).select(s1_id="id", c1="country")
    recs = load_recs("train").select(rec_id="id", c2="country")
    p = gt.filter(pl.col("s1_id").hash(seed=1) % int(1 / frac) == 0).join(s1, on="s1_id").join(recs, on="rec_id")
    print(f"[F5] sampled pairs {p.height:,}: country agrees in {(p['c1'] == p['c2']).mean():.2%}")


def _house(col):
    # leading house token as in the whitepaper taxonomy: first number (with suffix) near the start
    return _fold(pl.col(col)).str.extract(r"^\D{0,12}?(\d[\w/-]*)")


def house_relation(gt, country="US", src=2):
    """F2/F3: house-number relation for true matches vs near-miss decoys (same name key + same city).
    Same taxonomy as Figure 1 of the whitepaper."""
    def view(df):
        return df.select("id", nkey=_ntoks("name").list.sort().list.join(" "),
                         city=pl.col("addr").str.split(",").list.get(-2, null_on_oob=True).str.strip_chars().str.to_lowercase(),
                         h=_house("addr"))
    s1 = view(load("train", 1).filter(pl.col("country") == country))
    recs = view(load("train", src).filter(pl.col("country") == country))
    recs = recs.join(gt.rename({"rec_id": "id", "s1_id": "owner"}), on="id", how="left")
    pos = recs.drop_nulls("owner").join(s1.select(owner="id", h1="h"), on="owner")
    neg = recs.filter(pl.col("owner").is_null()).join(s1.select("nkey", "city", h1="h"), on=["nkey", "city"]).unique("id")

    def rel(df):
        df = df.drop_nulls(["h", "h1"])
        n1, n2 = (pl.col(c).str.extract(r"^(\d+)") for c in ("h1", "h"))
        d = (n1.cast(pl.Int64) - n2.cast(pl.Int64)).abs()
        return df.with_columns(
            rel=pl.when(pl.col("h") == pl.col("h1")).then(pl.lit("equal"))
                  .when(n1 == n2).then(pl.lit("suffix differs"))
                  .when(n1.str.strip_chars_start("0") == n2.str.strip_chars_start("0")).then(pl.lit("leading zero"))
                  .when(n1.str.ends_with(n2) | n2.str.ends_with(n1)).then(pl.lit("first digit truncated"))
                  .when(d <= 10).then(pl.lit("|delta| <= 10")).when(d <= 100).then(pl.lit("|delta| <= 100"))
                  .otherwise(pl.lit("far")))
    for name, df in (("true matches", rel(pos)), ("near-miss decoys", rel(neg))):
        dist = df.group_by("rel").len().with_columns(share=pl.col("len") / pl.col("len").sum()).sort("len", descending=True)
        print(f"[F2/F3] {country} S{src} {name} (n={df.height:,}): " + ", ".join(f"{r}={s:.1%}" for r, _, s in dist.rows()))


def sibling_agreement(gt, frac=0.05):
    """F4: two same-source siblings that both disagree with S1's house number agree with each other."""
    s1 = load("train", 1).select(s1_id="id", n1=_nums("addr").list.first())
    recs = load_recs("train").select(rec_id="id", src="src", n=_nums("addr").list.first())
    p = (gt.filter(pl.col("s1_id").hash(seed=2) % int(1 / frac) == 0).join(s1, on="s1_id").join(recs, on="rec_id")
           .drop_nulls(["n1", "n"]).filter(pl.col("n") != pl.col("n1")))
    g = p.group_by("s1_id", "src").agg(k=pl.len(), distinct=pl.col("n").n_unique()).filter(pl.col("k") >= 2)
    print(f"[F4] same-source sibling pairs where both differ from S1: {g.height:,}; "
          f"all siblings share one wrong number in {(g['distinct'] == 1).mean():.1%}")


def name_overlap(gt, frac=0.02):
    """F6: share of positives with no Latin name token shared with S1, by country."""
    s1 = load("train", 1).select(s1_id="id", country="country", t1=_ntoks("name"))
    recs = load_recs("train").select(rec_id="id", t=_ntoks("name"), addr="addr")
    p = gt.filter(pl.col("s1_id").hash(seed=3) % int(1 / frac) == 0).join(s1, on="s1_id").join(recs, on="rec_id")
    p = p.with_columns(none=pl.col("t").list.set_intersection(pl.col("t1")).list.len() == 0,
                       empty=pl.col("addr").str.strip_chars() == "")
    for (c,), g in p.group_by("country"):
        print(f"[F6] {c}: {g['none'].mean():.1%} of positives share no name token with S1; "
              f"[F8] {g['empty'].mean():.1%} have an empty address")


def empty_address(gt):
    """F8: records with an empty address are mostly true matches."""
    recs = load_recs("train").filter(pl.col("addr").str.strip_chars() == "").select(id="id")
    m = recs.join(gt.select(id="rec_id"), on="id", how="semi").height
    print(f"[F8] empty-address records {recs.height:,}: {m / recs.height:.1%} are true matches")


def indic_alignment(gt):
    """F6/C4: native-script names are token-aligned with the S1 name."""
    s1 = load("train", 1).select(s1_id="id", n1=pl.col("name").str.split(" ").list.len())
    recs = load_recs("train").filter(pl.col("name").str.contains(INDIC)).select(rec_id="id", n=pl.col("name").str.split(" ").list.len())
    p = gt.join(recs, on="rec_id").join(s1, on="s1_id")
    print(f"[C4] Indic-script positives {p.height:,}: token counts equal in {(p['n'] == p['n1']).mean():.2%}")


def postal_codes():
    """F9: no usable postal codes."""
    for c in ("US", "India"):
        a = load("train", 1).filter(pl.col("country") == c)["addr"]
        five, six = (a.str.contains(rf"\b\d{{{n}}}\b").mean() for n in (5, 6))
        print(f"[F9] {c} S1 addresses: 5-digit token in {five:.1%} (mostly house numbers), 6-digit token in {six:.2%}")


def id_leakage(gt):
    """F10: file position of an S1 and of its records are uncorrelated."""
    pos1 = load("train", 1).select(s1_id="id").with_row_index("p1")
    pos2 = load("train", 2).select(rec_id="id").with_row_index("p2")
    p = gt.join(pos1, on="s1_id").join(pos2, on="rec_id")
    print(f"[F10] corr(S1 row, S2 row) over {p.height:,} pairs = {p.select(pl.corr('p1', 'p2')).item():+.4f}")


if __name__ == "__main__":
    gt = owner_audit()
    distractor_share(gt)
    country_agreement(gt)
    house_relation(gt)
    sibling_agreement(gt)
    name_overlap(gt)
    empty_address(gt)
    indic_alignment(gt)
    postal_codes()
    id_leakage(gt)
