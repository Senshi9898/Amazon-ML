"""Token positive-rate table from the whole training set (E-11v): every stage-2 candidate pair of the
full-scale train run (1.94M S1 after excluding the DEV-VAL states, so DEV-VAL stays a clean test) labelled
with the ground truth. Folds by rec_id let DEV-TRAIN rows be scored out of fold.  python -m er.tokrates"""
import polars as pl

from . import stage2
from .data import CACHE, load, load_gt, load_recs
from .split import load_subset
from .views import indic_dict, name_view, s1_vocab

if __name__ == "__main__":
    dv = load_subset("devval")[0].select("id")
    pairs = pl.read_parquet(CACHE / "full" / "train" / "scores" / "*.parquet").select("rec_id", "s1_id")
    pairs = pairs.join(dv.rename({"id": "s1_id"}), on="s1_id", how="anti").unique()
    pairs = pairs.join(load_gt().with_columns(y=pl.lit(1, pl.Int8)), on=["s1_id", "rec_id"], how="left") \
                 .with_columns(pl.col("y").fill_null(0))
    print("pairs", pairs.height, "pos", pairs["y"].mean(), flush=True)
    s1 = load("train", 1).join(dv, on="id", how="anti")
    recs = load_recs("train").join(pairs.select(id="rec_id").unique(), on="id", how="semi")
    vocab = s1_vocab(s1)
    indic = indic_dict(exclude_s1=dv["id"], tag="ex_devval")
    nts = name_view(s1, "V4", vocab, indic).select(s1_id="id", nt1="nt")
    ntr = name_view(recs, "V4", vocab, indic).select(rec_id="id", nt="nt")
    del s1, recs
    t = stage2.token_table(pairs, nts, ntr)
    t.write_parquet(CACHE / "token_rates_full.parquet")
    print("table rows", t.height, "tokens", t["t"].n_unique(), flush=True)
