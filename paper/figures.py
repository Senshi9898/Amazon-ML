"""Figures + typeset equations for the white paper. Every number comes from paper/data.json
(built from experiments/experiments.csv, saved scores, models and ground truth) or from the
forensic measurements quoted in docs/RESEARCH_SPEC.md."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
D = json.load(open(HERE / "data.json"))

import os
_ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf")
for f in ("cmr10.ttf", "cmb10.ttf", "cmmi10.ttf", "DejaVuSerif.ttf"):
    font_manager.fontManager.addfont(os.path.join(_ttf, f))
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, MUTED, GRID, SURF = "#000000", "#333333", "#777777", "#e3e3e3", "#ffffff"
plt.rcParams.update({
    "font.family": ["cmr10", "DejaVu Serif"], "font.size": 9, "axes.formatter.use_mathtext": True, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 220, "savefig.bbox": "tight",
    "mathtext.fontset": "cm", "text.color": INK,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor=SURF)
    plt.close(fig)


def logrow(exp, subset="devval"):
    rows = [r for r in D["log"] if r["exp_id"] == exp and r["subset"] == subset and r["macro_f05"] not in (None, "", "None")]
    return rows[-1]


def notes_recall(r):
    return {k: float(v) for k, v in re.findall(r"recall_([a-z_]+)=([0-9.]+)", r["notes"] or "")}


# ---------- 1. architecture ----------
def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_axis_off(); ax.set_xlim(-2, 102); ax.set_ylim(-1.5, 47); ax.grid(False)
    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#fdf1ea" if color == ORANGE else "#f4f4f4",
                                    ec=color if color == ORANGE else "#555555", lw=1.0))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK, family=["cmb10", "DejaVu Serif"])
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.35)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    box(0, 25, 17, 20, "Ingest", "3 sources, TSV\n→ Parquet\n12.5M train\n11.7M test", INK)
    box(21, 25, 18, 20, "Views", "name: unwrap, IDs,\ndomains, Indic→Latin\naddress: all-number\nset, street tokens", BLUE)
    box(43, 25, 18, 20, "State blocks", "≈150k S1 per block\nrecords routed by\nlearned state map\nstateless → all", BLUE)
    box(65, 25, 16, 20, "Retrieval", "TF-IDF typed tokens\n∪ exact keys\ntop-8 S1 / record\nrecall 0.971", BLUE)
    box(84, 25, 16, 20, "Features", "35 pair features\nname · address\nhouse relation\nlegal form · OOV", BLUE)
    box(84, 1, 16, 19, "Stage 1", "LightGBM\n127 leaves × 400\n$p_1$ per pair\nfloor 0.01", INK)
    box(61, 1, 19, 19, "Stage 2", "+10 entity features\ncompetition\nsiblings\ncross-source", ORANGE)
    box(38, 1, 19, 19, "Assignment", "record → argmax S1\n$p \\geq \\tau = 0.65$\ncross-block\nmargin 0.1", AQUA)
    box(15, 1, 19, 19, "Outputs", r"$\mathtt{matching\_results}$" "\n" r"$\mathtt{candidate\_pairs}$" "\nvalidator PASS", AQUA)
    for a, b in ((17, 21), (39, 43), (61, 65), (81, 84)):
        arrow(a, 35, b, 35)
    arrow(92, 25, 92, 20); arrow(84, 10.5, 80, 10.5); arrow(61, 10.5, 57, 10.5); arrow(38, 10.5, 34, 10.5)
    save(fig, "architecture")


# ---------- 2. ablation ladder ----------
def ablation():
    steps = [("E-00", "Rules baseline"), ("E-01", "+ name noise inversion"), ("E-03", "+ Indic→Latin dictionary"),
             ("E-02", "+ all-number address set"), ("E-04a", "+ combined views, top-8 exact keys"),
             ("E-05", "+ TF-IDF ∪ exact retrieval (rules)"), ("E-07", "+ LightGBM pair scorer"),
             ("E-07b", "+ legal-form & OOV features"), ("E-11", "+ stage 2 (entity evidence)")]
    vals = [float(logrow(e)["macro_f05"]) for e, _ in steps]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    y = list(range(len(steps)))[::-1]
    ax.barh(y, vals, color=BLUE, height=0.56)
    ax.barh(y[-1], vals[-1], color=BLUE, height=0.56)
    for yi, v, (e, lab) in zip(y, vals, steps):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{lab}  ({e})" for e, lab in steps], fontsize=7.5)
    ax.set_xlim(0.55, 1.02); ax.set_xlabel("Macro F0.5 on held-out states (DEV-VAL, 255,858 S1)")
    ax.grid(axis="y", visible=False)
    save(fig, "ablation")


# ---------- 3. house-number relation (the decoy signal) ----------
def house():
    cats = ["equal", "suffix\ndiffers", "leading\nzero", "first digit\ntruncated", r"$|\Delta| \leq 10$", r"$|\Delta| \leq 100$", "far"]
    pos = [0.746, 0.091, 0.050, 0.025, 0.014, 0.010, 0.063]   # US S2 positives (forensics, §4 of plan)
    neg = [0.005, 0.004, 0.000, 0.001, 0.637, 0.274, 0.079]   # US S2 near-miss decoys
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    x = range(len(cats)); w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], pos, w, color=BLUE, label="true matches")
    ax.bar([i + w / 2 + 0.01 for i in x], neg, w, color=ORANGE, label="near-miss decoys (same name + city)")
    ax.set_xticks(list(x), cats, fontsize=7.5); ax.set_ylabel("share of pairs")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for i, v in ((0, pos[0]), (4, neg[4])):
        ax.text(i + (-w / 2 if i == 0 else w / 2), v + 0.02, f"{v:.1%}", ha="center", fontsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0), fontsize=7.5); ax.grid(axis="x", visible=False); ax.set_ylim(0, 0.85)
    save(fig, "house")


# ---------- 4. candidate recall by stratum ----------
def recall():
    exps = [("E-00", "exact keys, V1/A1"), ("E-04a", "exact keys, V3/A2, top-8"), ("E-04", "TF-IDF, top-8"), ("E-05", "TF-IDF ∪ exact")]
    strata = [("normal", "normal\n(730k)"), ("addr_only", "address-only\n(68k)"), ("indic", "Indic script\n(47k)"), ("name_only", "empty address\n(40k)")]
    cols = [MUTED, "#86b6ef", BLUE, "#1c5cab"]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.2
    for j, ((e, lab), c) in enumerate(zip(exps, cols)):
        r = notes_recall(logrow(e))
        xs = [i + (j - 1.5) * (w + 0.01) for i in range(len(strata))]
        ax.bar(xs, [r[s] for s, _ in strata], w, color=c, label=f"{e}: {lab}")
    ax.set_xticks(range(len(strata)), [l for _, l in strata], fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of true pairs retrieved")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(ncol=2, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0)); ax.grid(axis="x", visible=False)
    save(fig, "recall")


# ---------- 5. threshold curve ----------
def tau():
    c = D["tau_curve"]
    t = [r["tau"] for r in c]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot(t, [r["P"] for r in c], color=BLUE, lw=1.6, label="pair precision")
    ax.plot(t, [r["R"] for r in c], color=ORANGE, lw=1.6, label="pair recall")
    ax.plot(t, [r["f"] for r in c], color=INK, lw=2, label="macro F0.5")
    best = max(c, key=lambda r: r["f"])
    ax.axvline(0.65, color=MUTED, lw=0.8)
    ax.text(0.655, 0.905, "τ = 0.65\n(chosen on OOF\ntraining scores)", fontsize=6.8, color=INK2, va="bottom")
    ax.scatter([best["tau"]], [best["f"]], s=18, color=INK, zorder=3)
    ax.set_xlabel("acceptance threshold τ"); ax.set_ylim(0.88, 1.0)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "tau")


# ---------- 6. score separation ----------
def hist():
    b = D["bins"]; mids = [(b[i] + b[i + 1]) / 2 for i in range(len(b) - 1)]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.bar(mids, D["hist_0"], width=0.0235, color=ORANGE, label="non-matching candidates")
    ax.bar(mids, D["hist_1"], width=0.0235, color=BLUE, alpha=0.9, label="true matches")
    ax.set_yscale("log"); ax.set_xlabel("stage-2 score p"); ax.set_ylabel("pairs (log scale)")
    ax.axvline(0.65, color=MUTED, lw=0.8); ax.legend(fontsize=7, loc="upper center")
    save(fig, "hist")


# ---------- 7. feature importance ----------
def importance():
    names = {"sp_rel": "retrieval score / record best", "sp_rank": "retrieval rank", "sp_score": "retrieval score",
             "aj_gap": "address Jaccard gap to best", "n_ratio": "name edit ratio", "ndiff": r"house-number $|\Delta|$",
             "nj_gap": "name Jaccard gap to best", "a_tsr": "street token-set ratio", "n_tsr": "name token-set ratio",
             "ncand_s": "candidates per S1", "oov": "name OOV share", "nextra": "extra numbers", "p_margin": "margin to rival S1",
             "p1": "stage-1 probability", "p_rank": "rank among rivals", "sib_max": "best sibling score",
             "p_r2": "runner-up score", "ncover": "S1-number coverage", "leg_rel": "legal-form relation", "aj": "address Jaccard",
             "nj": "name Jaccard", "s_hi_other": "other-source support", "s_sum_other": "other-source score sum",
             "n1": "numbers in S1", "nmiss": "missing numbers", "ncand_r": "candidates per record", "alen_s": "S1 address tokens",
             "sj": "street Jaccard", "n_collide": "S1 name collisions", "grp_add": "'Group' added", "s_hi_same": "same-source support",
             "s_max_other": "other-source max", "sib_n": "sibling count", "first_eq": "first number equal", "leg_add": "legal tokens added", "leg_del": "legal tokens dropped"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, st, title in zip(axes, ("stage1", "stage2"), ("Stage 1 (35 features)", "Stage 2 (45 features)")):
        imp = D[f"imp_{st}"][:10]; tot = sum(g for _, g in D[f"imp_{st}"])
        ax.barh(range(10)[::-1], [g / tot for _, g in imp], color=BLUE if st == "stage1" else ORANGE, height=0.6)
        ax.set_yticks(range(10)[::-1], [names.get(n, n) for n, _ in imp], fontsize=7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=8, loc="left", color=INK); ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of total split gain", fontsize=7.5)
    fig.tight_layout(w_pad=2)
    save(fig, "importance")


# ---------- 8. match-count distribution: truth vs test predictions ----------
def kdist():
    def share(rows):
        tot = sum(n for _, n in rows); d = dict((int(k), n / tot) for k, n in rows)
        return [d.get(k, 0) for k in range(11)]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ks = list(range(11))
    ax.plot(ks, share(D["k_train"]), color=INK, lw=2, marker="o", ms=4, label="training ground truth")
    for c, col in (("US", BLUE), ("India", ORANGE), ("France", AQUA)):
        ax.plot(ks, share(D["k_test"][c]), color=col, lw=1.4, marker="o", ms=3.5, label=f"test predictions: {c}")
    ax.set_xticks(ks, [str(k) if k < 10 else "10+" for k in ks]); ax.set_xlabel("matches per S1 entity")
    ax.set_ylabel("share of S1"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=7)
    save(fig, "kdist")


# ---------- 9. full-scale per block ----------
def blocks():
    rows = sorted([r for r in D["blocks"] if r[0] != "*"], key=lambda r: (r[1], int(r[0].split("_")[1])))
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = range(len(rows))
    ax.scatter(x, [r[3] for r in rows], s=[max(10, r[2] / 2500) for r in rows],
               c=[BLUE if r[1] == "US" else ORANGE for r in rows], edgecolors=SURF, linewidths=1.2, zorder=3)
    ax.set_xticks(list(x), [r[0].replace("_", " ") for r in rows], rotation=45, ha="right", fontsize=6.8)
    ax.axhline(0.9612, color=INK2, lw=0.8); ax.text(len(rows) - 0.5, 0.9625, "overall 0.961", ha="right", fontsize=7, color=INK2)
    ax.set_ylim(0.86, 0.985); ax.set_ylabel("macro F0.5"); ax.grid(axis="x", visible=False)
    ax.scatter([], [], c=BLUE, s=25, label="US block"); ax.scatter([], [], c=ORANGE, s=25, label="India block")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "blocks")


# ---------- 10. F by number of true matches ----------
def fbyk():
    rows = D["f_by_k"]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=BLUE, width=0.6)
    for k, f, n in rows:
        ax.text(k, f + 0.004, f"{f:.3f}", ha="center", fontsize=6.5)
    ax.set_ylim(0.85, 1.0); ax.set_xlabel("true matches of the S1 entity"); ax.set_ylabel("mean F0.5")
    ax.set_xticks([r[0] for r in rows], [str(r[0]) if r[0] < 8 else "8+" for r in rows]); ax.grid(axis="x", visible=False)
    save(fig, "fbyk")


# ---------- 11. transfer (France proxy) ----------
def transfer():
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    lab = ["US test", "India test"]
    both, cross = [0.9720, 0.9650], [0.9579, 0.9122]
    ax.bar([0 - 0.2, 1 - 0.2], both, 0.38, color=BLUE, label="trained on US + India")
    ax.bar([0 + 0.2, 1 + 0.2], cross, 0.38, color=ORANGE, label="trained on the other country only")
    for i, (a, b) in enumerate(zip(both, cross)):
        ax.text(i - 0.2, a + 0.004, f"{a:.3f}", ha="center", fontsize=7); ax.text(i + 0.2, b + 0.004, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], lab); ax.set_ylim(0.85, 1.0); ax.set_ylabel("macro F0.5")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "transfer")


# ---------- equations ----------
EQ = {
    "f05": r"$F_{0.5}(e)=\dfrac{(1+\beta^2)\,P_e R_e}{\beta^2 P_e+R_e}=\dfrac{1.25\,|\hat M_e\cap M_e|}{0.25\,|M_e|+|\hat M_e|},\qquad \beta=0.5$",
    "macro": r"$\mathrm{Score}=\dfrac{1}{|S_1|}\sum_{e\in S_1}F_{0.5}(e),\qquad F_{0.5}(e)=1\ \ \mathrm{if}\ \ M_e=\hat M_e=\emptyset$",
    "idf": r"$w_t=\left(\log\dfrac{N_c}{\mathrm{df}_t}\right)^{2}\ \ (\mathrm{df}_t\leq 100),\qquad s(r,e)=\dfrac{\sum_{t\in T(r)\cap T(e)}w_t}{\sqrt{\sum_{t\in T(e)}w_t}}$",
    "jacc": r"$J(A,B)=\dfrac{|A\cap B|}{|A\cup B|},\qquad \mathrm{cover}(r,e)=\dfrac{|\mathcal{N}(e)\cap\mathcal{N}(r)|}{|\mathcal{N}(e)|}$",
    "delta": r"$\Delta(r,e)=\left|\min\,(\mathcal{N}(e)\setminus\mathcal{N}(r))-\min\,(\mathcal{N}(r)\setminus\mathcal{N}(e))\right|$",
    "stage2": r"$p_2(r,e)=g\left(\mathbf{x}(r,e),\ p_1(r,e),\ p_1(r,e)-\max_{e'\neq e}p_1(r,e'),\ \max_{r'\in\mathrm{sib}(r)}p_1(r',e),\ \sum_{r''\in\bar{s}(r)}p_1(r'',e)\right)$",
    "assign": r"$a(r)=\arg\max_{e}\,p_2(r,e)\ \ \mathrm{if}\ \ \max_e p_2(r,e)\geq\tau,\ \ \mathrm{else}\ \emptyset;\qquad \tau^{*}=\arg\max_{\tau}\ \mathrm{Score}_{\mathrm{OOF}}(\tau)$",
    "margin": r"$\mathrm{stateless}\ r:\ \ a(r)=\emptyset\ \ \mathrm{if}\ \ p_2^{(1)}(r)-p_2^{(2)}(r)<0.1\ \ \mathrm{across\ blocks}$",
}


def equations():
    for k, tex in EQ.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, tex, fontsize=12, color=INK)
        fig.savefig(FIG / f"eq_{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    for f in (architecture, ablation, house, recall, tau, hist, importance, kdist, blocks, fbyk, transfer, equations):
        f()
    print(sorted(p.name for p in FIG.iterdir()))


# ---------- 12. error anatomy (final model, DEV-VAL) ----------
def anatomy():
    A = json.load(open(HERE / "anatomy.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    fn = A["fn"]; lab = ["never retrieved", "retrieved,\nscore below τ", "assigned to\nanother S1"]
    v = [fn["not_retrieved"], fn["below_tau"], fn["given_to_other_s1"]]
    axes[0].barh(range(3)[::-1], v, color=BLUE, height=0.6)
    for i, x in zip(range(3)[::-1], v):
        axes[0].text(x + 400, i, f"{x:,} ({x / fn['total']:.0%})", va="center", fontsize=7.5)
    axes[0].set_yticks(range(3)[::-1], lab, fontsize=7.5); axes[0].set_xlim(0, max(v) * 1.45)
    axes[0].set_title(f"False negatives ({fn['total']:,})", fontsize=8.5, loc="left"); axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel("true pairs missed", fontsize=7.5)
    fp = sorted(A["fp"].items(), key=lambda x: -x[1])
    axes[1].barh(range(len(fp))[::-1], [x for _, x in fp], color=ORANGE, height=0.6)
    for i, (k, x) in zip(range(len(fp))[::-1], fp):
        axes[1].text(x + 60, i, f"{x:,} ({x / A['fp_total']:.0%})", va="center", fontsize=7.5)
    axes[1].set_yticks(range(len(fp))[::-1], [k.replace(" (", "\n(") for k, _ in fp], fontsize=7)
    axes[1].set_xlim(0, fp[0][1] * 1.5); axes[1].set_title(f"False positives ({A['fp_total']:,})", fontsize=8.5, loc="left")
    axes[1].grid(axis="y", visible=False); axes[1].set_xlabel("wrong pairs accepted", fontsize=7.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "anatomy")


if __name__ == "__main__":
    anatomy()
