"""Representation variants (spec C2 name view, C3 address view, C4 transliteration).

Name:    V1 = E-00 tokens (unaccent, lowercase, alnum, legal/honorific dropped)
         V2 = V1 + DBA/formerly/t-a unwrap + id-tag strip + domain/handle segmentation
         V3 = V2 + Indic->Latin token dictionary (learned from ground truth, C4)
Address: A1 = token set + house = first number at the start
         A2 = token set ('null' placeholders dropped) + nums = set of all numbers (leading zeros stripped)
"""
import polars as pl

from .data import CACHE, load, load_gt

DROP = ("inc llc ltd limited pvt private corp corporation co company group llp pllc plc the "
        "mr sri shri ms m s and of").split()
DROP_FR = "sarl sas sasu eurl sci sa snc".split()  # V2+: French legal forms (test-only country)
LEGAL = [w for w in DROP + DROP_FR if w not in ("the", "mr", "sri", "shri", "ms", "m", "s", "and", "of")]
INDIC = r"[\x{0900}-\x{0DFF}]"
WRAPPER = r"(?:formerly known as|formerly|\bdba\b|\bd/b/a\b|\bt/a\b|trading as)\s*:?\s*(.+)$"


def _fold(e):
    return e.str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()


# ---------- C4: Indic -> Latin dictionary ----------
def indic_dict(exclude_s1=None, tag="all"):
    """Positional token alignment of native-script record names with their S1 names (train GT).
    exclude_s1: S1 ids whose pairs must not be used (the evaluation subset). -> (old, new) lists."""
    path = CACHE / f"indic_dict_{tag}.parquet"
    if path.exists():
        d = pl.read_parquet(path)
        return d["t"].to_list(), d["t1"].to_list()
    gt = load_gt()
    if exclude_s1 is not None:
        gt = gt.join(exclude_s1.to_frame("s1_id"), on="s1_id", how="anti")
    recs = pl.concat([load("train", s) for s in (2, 3)]).filter(pl.col("name").str.contains(INDIC))
    s1 = load("train", 1).select(s1_id="id", name1="name")
    p = (recs.select(rec_id="id", t=pl.col("name").str.split(" "))
             .join(gt, on="rec_id").join(s1, on="s1_id")
             .with_columns(pl.col("name1").str.split(" ").alias("t1"))
             .filter(pl.col("t").list.len() == pl.col("t1").list.len())
             .explode("t", "t1", empty_as_null=True)
             .filter(pl.col("t").str.contains(INDIC))
             .with_columns(pl.col("t1").str.to_lowercase().str.strip_chars(".,()"))
             .group_by("t", "t1").len()
             .sort("len", descending=True).unique("t", keep="first"))
    p.select("t", "t1").write_parquet(path)
    return p["t"].to_list(), p["t1"].to_list()


# ---------- domain / handle segmentation ----------
def _segment(s, vocab, maxlen=24):
    """Min-cost split of a concatenated string into S1-vocabulary words (unknown char cost 2)."""
    n = len(s)
    best = [(0, 0)] + [None] * n  # (cost, back-pointer)
    for i in range(1, n + 1):
        for j in range(max(0, i - maxlen), i):
            if best[j] is None:
                continue
            w = s[j:i]
            c = 1 if (len(w) >= 2 and w in vocab) else (2 if len(w) == 1 else None)
            if c is not None and (best[i] is None or best[j][0] + c < best[i][0]):
                best[i] = (best[j][0] + c, j)
    out, i = [], n
    while i > 0:
        j = best[i][1]
        out.append(s[j:i])
        i = j
    toks, buf = [], ""
    for w in reversed(out):  # glue runs of unknown single chars back together
        if len(w) == 1:
            buf += w
        else:
            if buf:
                toks.append(buf)
                buf = ""
            toks.append(w)
    return toks + ([buf] if buf else [])


def name_view(df, v, vocab=None, indic=None):
    """Adds nt (token list, deduped) and nkey (sorted join).
    vocab: S1 name tokens (V2+ segmentation); indic: (old, new) token lists from indic_dict() (V3)."""
    name = pl.col("name")
    if v == "V3":
        old, new = indic
        name = (pl.when(name.str.contains(INDIC))
                  .then(name.str.split(" ").list.eval(pl.element().replace(old, new)).list.join(" "))
                  .otherwise(name))
    s = _fold(name)
    drop = DROP
    if v in ("V2", "V3"):
        drop = DROP + DROP_FR
        s = pl.coalesce(s.str.extract(WRAPPER), s)
        s = s.str.replace_all(r"\(id:?\s*\d+\)|#\s*\d+", " ")
    df = df.with_columns(_n=s)
    if v in ("V2", "V3"):
        # domain / handle: one token, '@' prefix or a tld suffix -> segment with the S1 vocabulary
        dom = (pl.col("_n").str.strip_chars().str.contains(r"^@?[a-z0-9\-]+(\.?(com|net|org|in|co|fr))?$")
               & pl.col("_n").str.contains(r"^\s*@|\.(com|net|org|in|co|fr)\s*$|^\s*[a-z0-9]{7,}com\s*$"))
        df = df.with_columns(_dom=dom)
        d = df.filter("_dom").select("id", "_n")
        if d.height:
            seg = [" ".join(_segment(x.strip().lstrip("@").rsplit(".", 1)[0].removesuffix("com")
                                     .replace("-", ""), vocab)) for x in d["_n"]]
            d = d.with_columns(_seg=pl.Series(seg))
            df = df.join(d.select("id", "_seg"), on="id", how="left").with_columns(
                _n=pl.coalesce("_seg", "_n")).drop("_seg")
        df = df.drop("_dom")
    return df.with_columns(
        pl.col("_n").str.extract_all(r"[a-z0-9]+").list.set_difference(pl.lit(drop)).list.unique().alias("nt"),
        # legal-form tokens (dropped from nt) kept separately: legal changes are a decoy signature
        pl.col("_n").str.extract_all(r"[a-z]+").list.set_intersection(pl.lit(LEGAL)).list.sort().alias("lt"),
    ).with_columns(pl.col("nt").list.sort().list.join(" ").alias("nkey")).drop("_n")


def s1_vocab(s1):
    return set(s1.select(_fold(pl.col("name")).str.extract_all(r"[a-z0-9]+").explode(empty_as_null=True)
                           .drop_nulls().unique()).to_series().to_list())


def addr_view(df, v):
    """Adds at (token set), st (alphabetic street-ish tokens), house (str) and, for A2, nums (int set)."""
    a = _fold(pl.col("addr"))
    if v == "A2":
        a = a.str.replace_all(r"<null>|\bnull\b", " ")
    df = df.with_columns(_a=a).with_columns(
        pl.col("_a").str.extract_all(r"[a-z0-9]+").list.unique().alias("at"))
    if v == "A1":
        df = df.with_columns(pl.col("addr").str.to_lowercase().str.extract(r"^\D{0,12}?(\d[\w/-]*)").alias("house"))
    else:
        df = df.with_columns(pl.col("_a").str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64))
                               .list.unique(maintain_order=True).alias("nums")).with_columns(
            pl.col("nums").list.first().cast(pl.String).alias("house"))
    return df.with_columns(
        pl.col("at").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$"))).alias("st")
    ).drop("_a")
