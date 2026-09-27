"""G3: frozen E-11 configuration applied at full scale (spec §8 gate G3).

The scorer was trained on dev subsets of whole states (~100-180k S1 per country). Retrieval IDF,
posting caps and several features (n_collide, ncand_s, oov) depend on how many S1s share a partition,
so full-size data is processed in *state blocks* of the same scale:
  S1      -> block of its state (France, unlabelled and ~dev-sized: one block)
  record  -> block of its learned state; no state (empty/unparsed address) -> every block of its country
A record scored in several blocks keeps its best S1 overall.

  python -m er.pipeline train              # fit + save stage-1 / stage-2 models on DEV-TRAIN
  python -m er.pipeline run  test|train    # score every block (one subprocess per block)
  python -m er.pipeline finish test|train  # assign + outputs (test) or full-scale evaluation (train)
"""
import json
import subprocess
import sys

import lightgbm as lgb
import polars as pl

from . import exp, features, learn, stage2
from .data import CACHE, ROOT, load, load_gt, load_recs, write_id_lists
from .split import STATES, _components, _last_match, learn_comp_map, load_subset, s1_states
from .views import addr_view, indic_dict, name_view, s1_vocab

CFG = exp.config("E-11t")  # + token-rate stage-2 features; pruned candidates (organisers: smaller candidate sets rank higher)
F1 = exp._feats(CFG)
F2 = F1 + [f for g in CFG["stage2"] for f in getattr(stage2, g)]
TOKR = "TOKR" in CFG["stage2"]
MODELS = CACHE / "models"
BLOCK = 150_000  # target S1 per block (dev subsets held 76k-180k S1 per country)


def _dir(split):
    return CACHE / "full" / split


# ---------------- models ----------------
def train():
    MODELS.mkdir(parents=True, exist_ok=True)
    tr = features.read(exp._pairs_path(CFG, "devtrain"))
    s1t, _, trutht = load_subset("devtrain")
    s1te = s1t.select(s1_id="id", country="country")
    m1 = learn.fit(tr, F1)
    m1.save_model(MODELS / "stage1.txt")
    del tr
    oof1 = pl.read_parquet(CACHE / "preds" / f"{CFG['preds']}_oof_devtrain.parquet")  # same config, saved by the parent run
    t2, table = exp._stage2_table(CFG, "devtrain", oof1)
    if TOKR:
        table.write_parquet(MODELS / "token_rates.parquet")
    tau, curve = learn.choose_tau(learn.oof(t2, F2), s1te, trutht)
    learn.fit(t2, F2).save_model(MODELS / "stage2.txt")
    (MODELS / "meta.json").write_text(json.dumps({"tau": tau, "oof_f05": curve[tau], "F1": F1, "F2": F2}))
    print("tau", tau, "oof", curve[tau])


# ---------------- blocks ----------------
def _record_state(recs):
    """Train-GT-learned component -> state map (0.04% wrong on train; see G3 notes)."""
    cmap_path = CACHE / "comp_state_map.parquet"
    if not cmap_path.exists():
        learn_comp_map(s1_states(), load_recs("train"), load_gt()).write_parquet(cmap_path)
    return recs.join(_last_match(_components(recs), pl.read_parquet(cmap_path)), on="id", how="left")


def plan(split):
    """-> s1 (id, country, block), recs (id, country, block | null = all blocks of the country)."""
    s1 = load(split, 1)
    st = _last_match(_components(s1), STATES)
    # fallback for S1s whose state is not spelled as in STATES: the learned component -> state map
    # (without it those S1s sat in a misc block their records never reach: F0.5 0.37 there)
    fb = _record_state(s1.select("id", "country", "addr")).select("id", fb="state")
    s1 = s1.join(st, on="id", how="left").join(fb, on="id", how="left").with_columns(
        pl.when(pl.col("country").is_in(["US", "India"])).then(pl.coalesce("state", "fb"))
          .otherwise(pl.col("country")).alias("state"))
    # greedy: states in size order fill blocks of ~BLOCK S1 per country; unknown-state S1 -> country misc block
    sizes = s1.drop_nulls("state").group_by("country", "state").len().sort("country", "len", descending=[False, True])
    assign, fill = {}, {}
    for c, stt, n in sizes.iter_rows():
        b = fill.setdefault(c, [0, 0])
        if b[1] and b[1] + n > BLOCK:
            b[0], b[1] = b[0] + 1, 0
        assign[(c, stt)] = f"{c}_{b[0]}"
        b[1] += n
    amap = pl.DataFrame({"country": [k[0] for k in assign], "state": [k[1] for k in assign],
                         "block": list(assign.values())})
    # S1 with no parsable state: "*" = scored in every block of its country (like stateless records)
    s1 = s1.join(amap, on=["country", "state"], how="left").with_columns(pl.col("block").fill_null("*"))
    recs = _record_state(load_recs(split)).join(amap, on=["country", "state"], how="left")
    return s1.select("id", "country", "block"), recs.select("id", "country", "block")


def run(split):
    out = _dir(split)
    out.mkdir(parents=True, exist_ok=True)
    s1, recs = plan(split)
    s1.write_parquet(out / "s1_blocks.parquet")
    recs.write_parquet(out / "rec_blocks.parquet")
    blocks = s1.filter(pl.col("block") != "*").group_by("block", "country").len().sort("block")
    print(blocks.rows(), "unrouted records:", recs["block"].null_count(), flush=True)
    for b, c, n in blocks.iter_rows():
        if (out / "scores" / f"{b}.parquet").exists():
            continue
        subprocess.run([sys.executable, "-m", "er.pipeline", "block", split, b], check=True)


def block(split, b):
    """Score one block: views -> candidates -> pair features -> stage 1 -> stage 2."""
    out = _dir(split)
    s1b = pl.read_parquet(out / "s1_blocks.parquet")
    country = s1b.filter(pl.col("block") == b)["country"][0]
    s1b = s1b.filter((pl.col("block") == b) | ((pl.col("block") == "*") & (pl.col("country") == country)))
    rb = pl.read_parquet(out / "rec_blocks.parquet").filter(
        (pl.col("block") == b) | (pl.col("block").is_null() & (pl.col("country") == country)))
    s1 = load(split, 1).join(s1b.select("id"), on="id", how="semi")
    recs = load_recs(split).join(rb.select("id"), on="id", how="semi")
    # train split = full-scale validation: the Indic dictionary must not see the evaluated S1s' pairs
    indic = indic_dict() if split == "test" else indic_dict(
        exclude_s1=load("train", 1).join(load_subset("devtrain")[0].select("id"), on="id", how="anti")["id"],
        tag="devtrain_only")
    vocab = s1_vocab(s1)
    s1v = addr_view(name_view(s1, CFG["name"], vocab, indic), CFG["addr"])
    rv = addr_view(name_view(recs, CFG["name"], vocab, indic), CFG["addr"])
    ppath = out / "pairs" / b
    if not (ppath / "part-15.parquet").exists():  # stage-2-only reruns reuse the block's pair features
        c = features.candidates(CFG, s1v, rv)
        features.build(c, s1v, rv, ppath, feats=F1)
        del c
    nts, ntr = s1v.select(s1_id="id", nt1="nt"), rv.select(rec_id="id", nt="nt")
    del s1v, rv
    m1, m2 = (lgb.Booster(model_file=str(MODELS / f)) for f in ("stage1.txt", "stage2.txt"))
    pairs = features.read(ppath)
    p1 = learn.predict(m1, pairs, F1)
    (out / "cands").mkdir(exist_ok=True)
    p1.select("rec_id", "s1_id").write_parquet(out / "cands" / f"{b}.parquet")
    del pairs
    keep = p1.filter(pl.col("p") >= stage2.FLOOR)
    pairs = pl.scan_parquet(ppath / "*.parquet").join(keep.lazy().select("rec_id", "s1_id"),
                                                      on=["rec_id", "s1_id"], how="semi").collect()
    t2 = stage2.context(pairs, keep, stage2.addr_sig(recs))
    if TOKR:
        t2 = stage2.token_rates(t2, nts, ntr, pl.read_parquet(MODELS / "token_rates.parquet"))
    (out / "scores").mkdir(exist_ok=True)
    learn.predict(m2, t2, F2).write_parquet(out / "scores" / f"{b}.parquet")
    print(b, country, "S1", s1.height, "recs", recs.height, "cands", p1.height, "stage2", t2.height, flush=True)


# ---------------- outputs ----------------
XMARGIN = 0.1  # stateless records: abstain if another block's best is within this (+0.0025 at full scale)
# The test set holds 5.75 records per S1 against 4.68 in train (1.9x the unmatched-record density);
# re-weighting DEV-VAL decoys by 1.9 moves the best tau from 0.65 to 0.75 (+0.0006 at that density).
TAU_SHIFT_TEST = 0.10


def assign(scores, tau, out):
    """Record -> best S1 over all blocks if p >= tau. A stateless record is scored in every block of its
    country, where same-name S1s of other states compete: it abstains unless its best block wins by XMARGIN."""
    unr = pl.read_parquet(out / "rec_blocks.parquet").filter(pl.col("block").is_null()).select(rec_id="id")
    # a "*" S1 (no parsable state) is scored in every block of its country: keep one score per pair
    # (duplicates made the same pair its own runner-up under XMARGIN: +0.0009 at full scale)
    scores = scores.group_by("s1_id", "rec_id").agg(pl.col("p").max())
    s = scores.filter(pl.col("p") >= tau)
    bad = (s.join(unr, on="rec_id", how="semi")
            .group_by("rec_id").agg(p1=pl.col("p").max(), p2=pl.col("p").sort(descending=True).get(1, null_on_oob=True))
            .filter(pl.col("p2").is_not_null() & (pl.col("p1") - pl.col("p2") < XMARGIN)))
    return (s.join(bad.select("rec_id"), on="rec_id", how="anti").sort("p", descending=True)
             .unique("rec_id", keep="first").select("s1_id", "rec_id"))
def finish(split):
    out = _dir(split)
    tau = json.loads((MODELS / "meta.json").read_text())["tau"] + (TAU_SHIFT_TEST if split == "test" else 0)
    scores = pl.read_parquet(out / "scores" / "*.parquet")
    pred = assign(scores, tau, out)
    s1 = pl.read_parquet(out / "s1_blocks.parquet")
    print("pred pairs", pred.height, "S1 with >=1 match", pred["s1_id"].n_unique(), "of", s1.height, flush=True)
    if split == "test":
        sub = ROOT / "submission" / "output"
        sub.mkdir(parents=True, exist_ok=True)
        write_id_lists(s1["id"], pred, sub / "matching_results.tsv", "matched_entity_ids")
        # every S1 lives in exactly one block, so its candidate list comes from that block's file alone;
        # written block by block (all ~115M candidate pairs at once exceed RAM)
        with open(sub / "candidate_pairs.tsv", "w") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for (b,), g in s1.group_by("block"):
                part = sub / f".cand_{'star' if b == '*' else b}.tsv"
                if b == "*":  # S1s scored in every block of their country: union over those blocks
                    c = pl.concat([pl.read_parquet(f).join(g.select(s1_id="id"), on="s1_id", how="semi")
                                   for f in (out / "cands").glob("*.parquet")])
                else:
                    c = pl.read_parquet(out / "cands" / f"{b}.parquet")
                write_id_lists(g["id"], c, part, "candidate_entity_ids")
                with open(part) as pf:
                    next(pf)
                    f.writelines(pf)
                part.unlink()
        by = s1.join(pred.group_by("s1_id").len("k"), left_on="id", right_on="s1_id", how="left") \
               .with_columns(pl.col("k").fill_null(0))
        print(by.group_by("country").agg(empty=(pl.col("k") == 0).mean(), mean_k=pl.col("k").mean()).rows())
        return
    # full-scale validation on train: evaluate S1s outside DEV-TRAIN states (never trained on)
    from .log import log_experiment
    from .metric import blocking, evaluate
    dev = load_subset("devtrain")[0].select("id")
    s1e = s1.join(dev, on="id", how="anti").select(s1_id="id", country="country")
    truth = load_gt().join(s1e.select("s1_id"), on="s1_id", how="semi")
    keep = lambda d: d.join(s1e.select("s1_id"), on="s1_id", how="semi")
    # blocking metrics block by block (all candidate pairs at once exceed RAM)
    hits, n_c = [], 0
    for f in sorted((out / "cands").glob("*.parquet")):
        c = keep(pl.read_parquet(f))
        n_c += c.height
        hits.append(truth.join(c, on=["s1_id", "rec_id"], how="semi"))
    hits = pl.concat(hits).unique()
    bm = {"n_cands": n_c, "cands_per_s1": n_c / s1e.height, "blocking_recall": hits.height / truth.height,
          "oracle_f05": evaluate(s1e, truth, hits, ci=False)["macro_f05"]}
    m = evaluate(s1e, truth, keep(pred))
    log_experiment("G3-full", "dev-scale blocks reproduce DEV-VAL F0.5 at full scale",
                   "frozen E-11, state blocks, train split minus DEV-TRAIN states", "train_full", {**bm, **m},
                   parent_id="E-11", notes=f"tau={tau}")


if __name__ == "__main__":
    {"train": train, "run": run, "block": block, "finish": finish}[sys.argv[1]](*sys.argv[2:])
