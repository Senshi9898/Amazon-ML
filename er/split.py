"""DEV-TRAIN / DEV-VAL subsets partitioned by state (spec §4.1).

Near-miss distractors and same-name S1 collisions live within a city, so a state partition keeps
all of the hard competition inside a subset (random S1 sampling would drop competing S1s).

S1 state: component matching a fixed list of US state codes / Indian state names.
S2/S3 state: component -> state map learned from ground-truth co-occurrence (covers codes, full
names, native-script names and city names); unmapped unmatched records are hash-sampled.
"""
import polars as pl

from .data import CACHE, load, load_gt, load_recs

DEV = CACHE / "dev"
FRAC = 0.10  # target share of each country's S1 per subset

US = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
      "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").lower().split()
INDIA = ["andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
         "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
         "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
         "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
         "delhi", "chandigarh", "puducherry", "jammu and kashmir", "ladakh", "lakshadweep",
         "andaman and nicobar islands", "dadra and nagar haveli and daman and diu"]
# state ids are "<country>:<name>" so that e.g. a stray "LA" inside an Indian address is not Louisiana
STATES = pl.DataFrame({"country": ["US"] * len(US) + ["India"] * len(INDIA), "comp": US + INDIA}) \
           .with_columns(state=pl.concat_str("country", pl.lit(":"), "comp"))
GIANT = 0.06  # states above this share of a country's S1 stay out of the dev subsets (diversity)


def _components(df):
    """id, country, pos, comp (lowercased, stripped address components)."""
    return (df.select("id", "country", pl.col("addr").str.split(",").alias("comp"))
              .with_columns(pl.int_ranges(pl.col("comp").list.len()).alias("pos"))
              .explode("comp", "pos", empty_as_null=True).drop_nulls("comp")
              .with_columns(pl.col("comp").str.strip_chars().str.to_lowercase()))


def _last_match(comps, mapping):
    """Per id: state of the right-most component found in mapping[country, comp, state]."""
    return (comps.join(mapping, on=["country", "comp"]).sort("pos")
                 .group_by("id").agg(pl.col("state").last()))


def s1_states():
    s1 = load("train", 1)
    st = _last_match(_components(s1), STATES)
    return s1.join(st, on="id", how="left")


def learn_comp_map(s1, recs, gt, min_n=20, purity=0.95):
    comps = _components(recs.join(gt.select(id="rec_id"), on="id", how="semi"))
    pairs = (comps.join(gt.rename({"rec_id": "id"}), on="id")
                  .join(s1.select(pl.col("id").alias("s1_id"), "state").drop_nulls(), on="s1_id")
                  .group_by("country", "comp", "state").len())
    k = ["country", "comp"]
    return (pairs.with_columns((pl.col("len") / pl.col("len").sum().over(k)).alias("share"),
                               pl.col("len").sum().over(k).alias("n"))
                 .filter((pl.col("n") >= min_n) & (pl.col("share") >= purity))
                 .select("country", "comp", "state"))


def build():
    DEV.mkdir(parents=True, exist_ok=True)
    s1 = s1_states()
    print("S1 without state:", s1["state"].null_count(), "of", s1.height, flush=True)
    gt = load_gt()
    recs = load_recs("train")
    cmap = learn_comp_map(s1, recs, gt)
    print("learned component->state entries:", cmap.height, flush=True)
    recs = recs.join(_last_match(_components(recs), cmap), on="id", how="left")
    owner = gt.rename({"rec_id": "id", "s1_id": "owner"})
    recs = recs.join(owner, on="id", how="left")
    um = recs.filter(pl.col("owner").is_null())
    print("unmatched records without state:", um["state"].null_count(), "of", um.height, flush=True)

    # Assign whole states to subsets: seeded shuffle, fill DEV-TRAIN to FRAC, then DEV-VAL to FRAC.
    counts = s1.drop_nulls("state").group_by("country", "state").len().sort("country", "state")
    subsets = {"devtrain": [], "devval": []}
    for (country,), g in counts.group_by("country", maintain_order=True):
        total = g["len"].sum()
        g = g.sample(fraction=1.0, shuffle=True, seed=7)
        acc, cur = 0, "devtrain"
        for st, n in zip(g["state"], g["len"]):
            if n > GIANT * total:
                continue
            if cur == "devtrain" and acc >= FRAC * total:
                cur, acc = "devval", 0
            elif cur == "devval" and acc >= FRAC * total:
                break
            subsets[cur].append(st)
            acc += n

    for name, states in subsets.items():
        s1_sub = s1.filter(pl.col("state").is_in(states))
        frac = s1_sub.height / s1.height
        truth = gt.join(s1_sub.select(s1_id="id"), on="s1_id", how="semi")
        rec_sub = recs.filter(
            pl.col("owner").is_in(s1_sub["id"].implode())
            | (pl.col("owner").is_null() & pl.col("state").is_in(states))
            | (pl.col("owner").is_null() & pl.col("state").is_null()
               & (pl.col("id").hash(seed=11) % 1000 < int(frac * 1000))))
        s1_sub.write_parquet(DEV / f"{name}_s1.parquet")
        rec_sub.drop("owner").write_parquet(DEV / f"{name}_recs.parquet")
        truth.write_parquet(DEV / f"{name}_truth.parquet")
        print(name, "states:", sorted(states))
        print(f"  S1 {s1_sub.height} ({frac:.3f}), recs {rec_sub.height}, truth pairs {truth.height}, "
              f"singletons {s1_sub.height - truth['s1_id'].n_unique()}, "
              f"unmatched recs {rec_sub.filter(pl.col('owner').is_null()).height}, "
              f"by country {dict(s1_sub.group_by('country').len().iter_rows())}", flush=True)


def strata(s1, recs, truth):
    """rec_id -> stratum of each true pair (spec §5): name_only (empty address), indic (native-script
    name), addr_only (no Latin name token shared with S1), normal."""
    def toks(c):
        return pl.col(c).str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase() \
                 .str.extract_all(r"[a-z0-9]+")
    t = (truth.join(recs.select(rec_id="id", name="name", addr="addr"), on="rec_id")
              .join(s1.select(s1_id="id", name1="name"), on="s1_id"))
    return t.select("rec_id", stratum=pl.when(pl.col("addr").str.strip_chars() == "").then(pl.lit("name_only"))
                    .when(pl.col("name").str.contains(r"[\x{0900}-\x{0DFF}]")).then(pl.lit("indic"))
                    .when(toks("name").list.set_intersection(toks("name1")).list.len() == 0).then(pl.lit("addr_only"))
                    .otherwise(pl.lit("normal")))


def load_subset(name):
    """-> (s1, recs, truth) for 'devtrain' / 'devval'."""
    return tuple(pl.read_parquet(DEV / f"{name}_{t}.parquet") for t in ("s1", "recs", "truth"))


if __name__ == "__main__":
    build()
