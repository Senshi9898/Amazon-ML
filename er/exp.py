"""Experiment registry + runner (spec §6, §8). Usage: python -m er.exp E-01 [devval|devtrain]

Each experiment = parent config + one change. Rules pipeline until the learned scorer (E-07).
"""
import hashlib
import subprocess
import sys

import polars as pl

from . import decide, features, learn, retrieve, rules, stage2
from .data import CACHE
from .log import log_experiment
from .metric import blocking, evaluate
from .split import load_subset, strata
from .views import addr_view, indic_dict, name_view, s1_vocab

BASE = {"name": "V1", "addr": "A1", "topk": None, "ret": "exact", "scorer": "rules", "fv": 1}
EXPS = {  # id: (parent, hypothesis, change dict)
    "E-00": ("", "rules give a usable floor + harness sanity check", {}),
    "E-01": ("E-00", "inverting name noise (DBA/formerly, id tags, domains) helps", {"name": "V2"}),
    "E-02": ("E-00", "all-number set beats first-number house (India DOOR NO, reordered S1)", {"addr": "A2"}),
    "E-03": ("E-01", "Indic->Latin dictionary recovers India positives", {"name": "V3"}),
    "E-03x": ("E-03", "E-01+E-02+E-03 gains stack", {"addr": "A2"}),
    "E-04a": ("E-03x", "R-exact within the C6 budget (top-8 S1 per record by shared keys) keeps the gain",
              {"topk": 8}),
    "E-04": ("E-04a", "TF-IDF typed-token retrieval beats exact keys at equal budget (k=8)",
             {"ret": "sparse", "cap": 100}),
    "E-04b": ("E-04", "looser posting cap (300) buys recall worth its cost", {"cap": 300}),
    "E-05": ("E-04", "union with R-exact (C6) adds >=0.3pt recall (name-only records)", {"ret": "sparse+exact"}),
    "E-07": ("E-05", "a learned pair scorer (LightGBM, C7 features) beats rules", {"scorer": "gbdt"}),
    "E-07b": ("E-07", "legal-form relation + name OOV share separate decoys from positives", {"fv": 2}),
    "E-07p": ("E-07b", "relative-score candidate pruning (0.5 of record best | best exact) keeps F0.5 at 3.3x fewer candidates",
              {"prune": 0.5}),
    "E-11p": ("E-07p", "stage 2 on pruned candidates", {"stage2": ["CTX2", "SIB", "XSRC"], "preds": "E-07p"}),
    "E-07q": ("E-07p", "cap each S1 at its 50 best candidates (removes generic-name hubs)", {"s1cap": 50}),
    "E-11q": ("E-07q", "stage 2 on pruned + capped candidates", {"stage2": ["CTX2", "SIB", "XSRC"], "preds": "E-07q"}),
    "E-07r": ("E-07q", "recall fix: rarest-6 query tokens at cap 3000, abbreviation/state/ordinal expansion, "
                       "dotted legal forms, all-pairs number relations, token-alignment features",
              {"name": "V4", "addr": "A3", "cap": 3000, "q": 6, "fv": 4}),
    "E-11r": ("E-07r", "stage 2 on E-07r", {"stage2": ["CTX2", "SIB", "XSRC"], "preds": "E-07r"}),
    "E-07s": ("E-07r", "guaranteed query quota for the rarest name tokens: a corrupted house number no longer "
                       "empties the query (6.7k exact-name DEV-VAL positives were never retrieved)", {"qn": 3}),
    "E-11s": ("E-07s", "stage 2 on E-07s", {"stage2": ["CTX2", "SIB", "XSRC"], "preds": "E-07s"}),
    "E-11t": ("E-11s", "positive rate of the swapped-in / dropped name tokens (train labels) separates decoy "
                       "vocabulary from corruption vocabulary", {"stage2": ["CTX2", "SIB", "XSRC", "TOKR"]}),
    "E-11u": ("E-11t", "the same statistic on street tokens and on the exact substitution pair adds precision",
              {"stage2": ["CTX2", "SIB", "XSRC", "TOKR", "TOKA", "TOKP"]}),
    "E-11v": ("E-11t", "token rates from all 1.94M non-DEV-VAL training S1s (9x the label mass) cover the rare tokens",
              {"stage2": ["CTX2", "SIB", "XSRC", "TOKR"], "tokfile": "token_rates_full.parquet"}),
    "E-08": ("E-07b", "up-weighting OOF hard negatives (p>=0.1, x5) raises precision at equal recall",
             {"hardneg": 5.0}),
    "E-11h": ("E-08", "stage 2 on the hard-negative stage 1", {"stage2": ["CTX2", "SIB", "XSRC"], "preds": "E-08"}),
    "E-05c": ("E-05", "R-char (name 4-grams) on empty-address / uncovered records lifts name-only recall",
              {"ret": "sparse+exact+char"}),
    "E-07c": ("E-07b", "R-char candidates (+ch_score/ch_rank features) raise macro F0.5 >= +0.002",
              {"ret": "sparse+exact+char", "fv": 3}),
    # stage 2 on E-07b scores (OOF on DEV-TRAIN, full model on DEV-VAL)
    "E-10a": ("E-07b", "record-competition context (runner-up, margin, rank) as stage-2 features helps",
              {"stage2": ["CTX2"], "preds": "E-07b"}),
    "E-10": ("E-10a", "within-source sibling consensus (shared base address) helps", {"stage2": ["CTX2", "SIB"]}),
    "E-11": ("E-10", "cross-source S1 support (S2<->S3) helps", {"stage2": ["CTX2", "SIB", "XSRC"]}),
    # France proxy: stage-1 E-07b model trained on one country, scored on the other
    "E-14": ("E-07b", "features transfer across countries (France proxy): train one country, test the other",
             {"xcountry": True}),
    # decision layer: reuse E-07's saved scores (OOF on DEV-TRAIN for tuning, DEV-VAL for evaluation)
    "E-09": ("E-07", "a top1-top2 margin removes S1<->S1 confusions", {"decision": "margin"}),
    "E-12": ("E-07", "per-S1 expected-F0.5 prefix beats a global tau", {"decision": "ef05"}),
    "E-13": ("E-12", "isotonic calibration (OOF) improves the expected-F0.5 decision", {"calib": "isotonic"}),
}
TAUS = [round(0.4 + 0.05 * i, 2) for i in range(10)]
DECISIONS = {
    "margin": (decide.margin, [{"tau": t, "m": m} for t in TAUS for m in (0.0, 0.05, 0.1, 0.2, 0.3)]),
    "ef05": (decide.ef05, [{"floor": f} for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]),
}


def config(exp_id):
    parent, _, change = EXPS[exp_id]
    return {**(config(parent) if parent else BASE), **change}


def _views(cfg, subset):
    s1, recs, truth = load_subset(subset)
    vocab = s1_vocab(s1) if cfg["name"] != "V1" else None
    indic = indic_dict(exclude_s1=s1["id"], tag=f"ex_{subset}") if cfg["name"] in ("V3", "V4") else None
    s1v = addr_view(name_view(s1, cfg["name"], vocab, indic), cfg["addr"])
    rv = addr_view(name_view(recs, cfg["name"], vocab, indic), cfg["addr"])
    return s1, recs, truth, s1v, rv


def _pairs_path(cfg, subset):
    key = {k: cfg[k] for k in ("name", "addr", "topk", "ret", "cap")}
    if cfg.get("prune"):
        key["prune"] = cfg["prune"]
    if cfg.get("s1cap"):
        key["s1cap"] = cfg["s1cap"]
    if cfg.get("q"):
        key["q"] = cfg["q"]
    if cfg.get("qn"):
        key["qn"] = cfg["qn"]
    if cfg["fv"] > 1:
        key["fv"] = cfg["fv"]
    return CACHE / "pairs" / f"{subset}_{hashlib.md5(str(sorted(key.items())).encode()).hexdigest()[:8]}.parquet"


def build_pairs(exp_id, subset):
    """Candidate union + pair features for one subset (run in its own process to bound peak RAM)."""
    cfg = config(exp_id)
    s1, recs, truth, s1v, rv = _views(cfg, subset)
    del s1, recs
    c = features.candidates(cfg, s1v, rv)
    features.build(c, s1v, rv, _pairs_path(cfg, subset), truth, feats=_feats(cfg))


def _feats(cfg):
    return {1: features.FEATS, 2: features.FEATS2, 3: features.FEATS3, 4: features.FEATS4}[cfg["fv"]]


def run_learned(exp_id, subset):
    cfg = config(exp_id)
    for sub in ("devtrain", subset):
        if not _pairs_path(cfg, sub).exists():
            subprocess.run([sys.executable, "-m", "er.exp", "--build", exp_id, sub], check=True)
    feats = _feats(cfg)
    tr = features.read(_pairs_path(cfg, "devtrain"))
    if cfg.get("hardneg"):  # weights from the parent's OOF scores on the same training subset (DEV-VAL untouched)
        base = pl.read_parquet(CACHE / "preds" / f"{EXPS[exp_id][0]}_oof_devtrain.parquet").rename({"p": "p0"})
        tr = tr.join(base, on=["rec_id", "s1_id"], how="left").with_columns(
            w=pl.when((pl.col("y") == 0) & (pl.col("p0") >= 0.1)).then(cfg["hardneg"]).otherwise(1.0).cast(pl.Float32)).drop("p0")
    s1t, _, trutht = load_subset("devtrain")
    oof = learn.oof(tr, feats)
    tau, curve = learn.choose_tau(oof, s1t.select(s1_id="id", country="country"), trutht)
    model = learn.fit(tr, feats)
    del tr
    s1, recs, truth = load_subset(subset)
    va = features.read(_pairs_path(cfg, subset))
    s1e = s1.select(s1_id="id", country="country")
    bm = blocking(
        s1e, truth, va.select("rec_id", "s1_id"), strata(s1, recs, truth))
    scores = learn.predict(model, va, feats)
    (CACHE / "preds").mkdir(exist_ok=True)  # saved for decision-layer experiments (E-09, E-12, E-13)
    oof.write_parquet(CACHE / "preds" / f"{exp_id}_oof_devtrain.parquet")
    scores.write_parquet(CACHE / "preds" / f"{exp_id}_{subset}.parquet")
    pred = learn.assign(scores, tau)
    m = evaluate(s1e, truth, pred)
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])[:8]
    notes = _fp_notes(pred, truth, bm) + f" tau={tau} oof_curve={ {k: round(v, 4) for k, v in curve.items()} }" \
        + " top_gain=" + ",".join(f"{f}:{g:.0f}" for f, g in imp)
    parent, hyp, change = EXPS[exp_id]
    log_experiment(exp_id, hyp, str(change), subset, {**bm, **m}, parent_id=parent, notes=notes)


def _fp_notes(pred, truth, bm):
    # false-positive anatomy: distractor accepted vs record given to the wrong S1
    wrong = pred.join(truth, on=["s1_id", "rec_id"], how="anti")
    fp_dis = wrong.join(truth.select("rec_id"), on="rec_id", how="anti").height
    notes = " ".join(f"{k}={v:.4f}" for k, v in bm.items() if k.startswith("recall_"))
    return notes + f" fp_distractor={fp_dis} fp_wrong_s1={wrong.height - fp_dis} n_pred={pred.height}"


def run_decision(exp_id, subset, preds_from="E-07"):
    cfg = config(exp_id)
    s1t, _, trutht = load_subset("devtrain")
    s1, recs, truth = load_subset(subset)
    s1te, s1e = (x.select(s1_id="id", country="country") for x in (s1t, s1))
    oof = pl.read_parquet(CACHE / "preds" / f"{preds_from}_oof_devtrain.parquet")
    va = pl.read_parquet(CACHE / "preds" / f"{preds_from}_{subset}.parquet")
    if cfg.get("calib") == "isotonic":
        oof, va = decide.calibrate(oof, trutht, oof), decide.calibrate(oof, trutht, va)
    fn, grid = DECISIONS[cfg["decision"]]
    params, res = decide.tune(fn, grid, oof, s1te, trutht)
    pred = fn(va, **params)
    bm = blocking(s1e, truth, va.select("rec_id", "s1_id"))
    m = evaluate(s1e, truth, pred)
    top = sorted(res.items(), key=lambda x: -x[1])[:4]
    notes = _fp_notes(pred, truth, bm) + f" params={params} oof_top={[(dict(k), round(v, 4)) for k, v in top]}"
    parent, hyp, change = EXPS[exp_id]
    log_experiment(exp_id, hyp, str(change), subset, {**bm, **m}, parent_id=parent, notes=notes)


def _stage2_table(cfg, subset, p, table=None):
    """Stage 2 only sees pairs stage 1 did not confidently reject (p >= FLOOR: 8% of pairs, 99.94% of
    retrieved positives); everything below the floor stays rejected.
    TOKR: table = token rate table (from DEV-TRAIN labels); None -> build it from this subset's labels and
    score its rows out of fold. Returns (table, rate_table)."""
    _, recs, _ = load_subset(subset)
    p = p.filter(pl.col("p") >= stage2.FLOOR)
    path = _pairs_path(cfg, subset)
    pairs = pl.scan_parquet(path / "*.parquet" if path.is_dir() else path).join(
        p.lazy().select("rec_id", "s1_id"), on=["rec_id", "s1_id"], how="semi").collect()
    t = stage2.context(pairs, p, stage2.addr_sig(recs))
    if "TOKR" in cfg.get("stage2", []):
        _, _, _, s1v, rv = _views(cfg, subset)
        if "TOKA" in cfg["stage2"] or "TOKP" in cfg["stage2"]:
            nts, ntr = s1v.select(s1_id="id", nt1="nt", st1="st"), rv.select(rec_id="id", nt="nt", st="st")
        else:
            nts, ntr = s1v.select(s1_id="id", nt1="nt"), rv.select(rec_id="id", nt="nt")
        del s1v, rv
        oof = table is None
        if oof and cfg.get("tokfile"):  # rates built from the full training set by er.tokrates (DEV-VAL excluded)
            table = pl.read_parquet(CACHE / cfg["tokfile"])
        elif oof:
            table = stage2.token_table(t.select("rec_id", "s1_id", "y"), nts, ntr)
        t = stage2.token_rates(t, nts, ntr, table, oof=oof)
    return t, table


def run_stage2(exp_id, subset):
    cfg = config(exp_id)
    feats = _feats(cfg) + [f for g in cfg["stage2"] for f in getattr(stage2, g)]
    tr, table = _stage2_table(cfg, "devtrain", pl.read_parquet(CACHE / "preds" / f"{cfg['preds']}_oof_devtrain.parquet"))
    s1t, _, trutht = load_subset("devtrain")
    oof = learn.oof(tr, feats)
    tau, curve = learn.choose_tau(oof, s1t.select(s1_id="id", country="country"), trutht)
    model = learn.fit(tr, feats)
    del tr
    va, _ = _stage2_table(cfg, subset, pl.read_parquet(CACHE / "preds" / f"{cfg['preds']}_{subset}.parquet"), table)
    s1, recs, truth = load_subset(subset)
    s1e = s1.select(s1_id="id", country="country")
    scores = learn.predict(model, va, feats)
    oof.write_parquet(CACHE / "preds" / f"{exp_id}_oof_devtrain.parquet")
    scores.write_parquet(CACHE / "preds" / f"{exp_id}_{subset}.parquet")
    pred = learn.assign(scores, tau)
    bm = blocking(s1e, truth, va.select("rec_id", "s1_id"))
    m = evaluate(s1e, truth, pred)
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])[:8]
    notes = _fp_notes(pred, truth, bm) + f" tau={tau} oof_best={curve[tau]:.4f} top_gain=" \
        + ",".join(f"{f}:{g:.0f}" for f, g in imp)
    parent, hyp, change = EXPS[exp_id]
    log_experiment(exp_id, hyp, str(change), subset, {**bm, **m}, parent_id=parent, notes=notes)


def run_xcountry(exp_id, subset):
    """For each country C: train on DEV-TRAIN of the other country only, score DEV-VAL of C, and compare
    with the model trained on both countries (tau picked on the training data's OOF in both cases)."""
    cfg = config(exp_id)
    feats = _feats(cfg)
    s1t, _, trutht = load_subset("devtrain")
    s1, _, truth = load_subset(subset)
    ct = s1t.select(s1_id="id", country="country")
    cv = s1.select(s1_id="id", country="country")
    tr = features.read(_pairs_path(cfg, "devtrain")).join(ct, on="s1_id")
    va = features.read(_pairs_path(cfg, subset)).join(cv, on="s1_id")
    res, notes = {}, []
    for c, other in (("US", "India"), ("India", "US")):
        trc = tr.filter(pl.col("country") == other)
        tau, _ = learn.choose_tau(learn.oof(trc, feats), ct.filter(pl.col("country") == other), trutht)
        model = learn.fit(trc, feats)
        del trc
        vac = va.filter(pl.col("country") == c)
        m = evaluate(cv.filter(pl.col("country") == c), truth, learn.assign(learn.predict(model, vac, feats), tau), ci=False)
        res[f"f05_{c.lower()}"] = m["macro_f05"]
        notes.append(f"{other}->{c}: {m['macro_f05']:.4f} (tau={tau})")
    ref = pl.read_csv(__import__("er.log", fromlist=["LOG"]).LOG).filter(
        (pl.col("exp_id") == "E-07b") & (pl.col("subset") == subset)).tail(1)
    for c in ("us", "india"):
        notes.append(f"{c}: both-countries model {ref[f'f05_{c}'][0]:.4f}, transfer gap {res[f'f05_{c}'] - ref[f'f05_{c}'][0]:+.4f}")
    parent, hyp, change = EXPS[exp_id]
    log_experiment(exp_id, hyp, str(change), subset, res, parent_id=parent, notes="; ".join(notes))


def run(exp_id, subset="devval"):
    cfg = config(exp_id)
    if cfg.get("xcountry"):
        return run_xcountry(exp_id, subset)
    if cfg.get("stage2"):
        return run_stage2(exp_id, subset)
    if cfg.get("decision"):
        return run_decision(exp_id, subset)
    if cfg["scorer"] == "gbdt":
        return run_learned(exp_id, subset)
    s1, recs, truth, s1v, rv = _views(cfg, subset)
    s1e = s1.select(s1_id="id", country="country")
    cands = features.candidates(cfg, s1v, rv).select("rec_id", "s1_id")
    bm = blocking(s1e, truth, cands, strata(s1, recs, truth))
    pred = rules.assign(rules.score_chunked(cands, s1v, rv, cfg["addr"]))
    m = evaluate(s1e, truth, pred)
    notes = _fp_notes(pred, truth, bm) + f" cfg={cfg}"
    parent, hyp, change = EXPS[exp_id]
    log_experiment(exp_id, hyp, str(change or cfg), subset, {**bm, **m}, parent_id=parent, notes=notes)


if __name__ == "__main__":
    if sys.argv[1] == "--build":
        build_pairs(*sys.argv[2:])
    else:
        run(*sys.argv[1:])
