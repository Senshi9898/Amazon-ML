"""Paths, TSV -> Parquet ingest (C1), loaders."""
import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get(
    "ER_DATA", ROOT / "ref/6ab10eb3b23ba_student_resource/student_resource/dataset"))
CACHE = Path(os.environ.get("ER_CACHE", ROOT / "cache"))
RAW = CACHE / "raw"


def _read_tsv(path):
    # quote_char=None: names/addresses contain ' and " literally; files have no embedded tabs.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def ingest():
    RAW.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            out = RAW / f"{split}_s{s}.parquet"
            if out.exists():
                continue
            df = _read_tsv(DATA / split / f"{split}_source{s}.tsv")
            df = df.rename({"entity_id": "id", "business_name": "name",
                            "business_address": "addr"}).with_columns(pl.col("addr").fill_null(""))
            df.write_parquet(out)
            print(out.name, df.height, flush=True)
    out = RAW / "train_gt.parquet"
    if not out.exists():
        gt = _read_tsv(DATA / "train/train_ground_truth.tsv")
        gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
                .explode("matched_entity_ids", empty_as_null=True)
                .drop_nulls("matched_entity_ids")
                .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "rec_id"}))
        gt.write_parquet(out)
        print(out.name, gt.height, flush=True)


def load(split, s):
    """Raw records: id, name, addr, country."""
    return pl.read_parquet(RAW / f"{split}_s{s}.parquet")


def load_recs(split):
    """S2 and S3 stacked, with a `src` column (2/3)."""
    return pl.concat([load(split, s).with_columns(pl.lit(s, pl.Int8).alias("src")) for s in (2, 3)])


def load_gt():
    return pl.read_parquet(RAW / "train_gt.parquet")


def write_id_lists(s1_ids, pairs, path, col):
    """Official output format: one row per S1 (empty list allowed), comma-joined S2/S3 ids, TSV.
    col = 'matched_entity_ids' (matching_results.tsv) or 'candidate_entity_ids' (candidate_pairs.tsv)."""
    lists = pairs.unique(["s1_id", "rec_id"]).sort("rec_id").group_by("s1_id").agg(pl.col("rec_id").str.join(","))
    out = (pl.DataFrame({"source1_entity_id": s1_ids}).join(
        lists.rename({"s1_id": "source1_entity_id", "rec_id": col}), on="source1_entity_id", how="left")
        .with_columns(pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")


if __name__ == "__main__":
    ingest()
