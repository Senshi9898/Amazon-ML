"""Experiment log: one row per run in experiments/experiments.csv (schema = spec §6)."""
import csv
import datetime
import resource
import subprocess
import time

from .data import ROOT

LOG = ROOT / "experiments/experiments.csv"
COLS = ["exp_id", "parent_id", "timestamp", "git_rev", "hypothesis", "change", "subset", "n_s1",
        "n_cands", "cands_per_s1", "blocking_recall", "oracle_f05", "precision", "recall",
        "macro_f05", "f05_ci_lo", "f05_ci_hi", "f05_us", "f05_india", "f05_singletons",
        "runtime_s", "peak_rss_mb", "conclusion", "notes"]
_T0 = time.time()


def _git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nocommit"
    except OSError:
        return "nogit"


def log_experiment(exp_id, hypothesis, change, subset, metrics, parent_id="", conclusion="INFO", notes=""):
    row = {c: "" for c in COLS}
    row.update({k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items() if k in COLS})
    row.update(exp_id=exp_id, parent_id=parent_id, hypothesis=hypothesis, change=change, subset=subset,
               conclusion=conclusion, notes=notes, git_rev=_git_rev(),
               timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
               runtime_s=round(time.time() - _T0, 1),
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print({k: row[k] for k in COLS if row[k] != ""})


def conclude(exp_id, conclusion, note="", subset="devval"):
    """Set the conclusion (KEEP/KILL/INFO) of the latest run of exp_id on subset; append note."""
    with open(LOG, newline="") as f:
        rows = list(csv.DictReader(f))
    i = max(k for k, r in enumerate(rows) if r["exp_id"] == exp_id and r["subset"] == subset)
    rows[i]["conclusion"] = conclusion
    rows[i]["notes"] = (note + " | " + rows[i]["notes"]).strip(" |")
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
