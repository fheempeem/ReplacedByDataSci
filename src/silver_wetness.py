"""Silver stage: image-based wetness score -> silver.wetness.

p_wet = CLIP zero-shot probability that the image shows a wet road / rain (see wetness_eval.py for the prompts).
It is a RANK-LIKE score, not a calibrated probability. Validation against 46 daytime hand labels (docs/wetness_validation.md):
AUC 0.78 (95% interval 0.58-0.95), better than the Open-Meteo rain value (0.62, 0.47-0.78). Night/dusk scores are NOT
validated (street lights make dry asphalt look wet), so analyses should use daytime images only.

Needs the embeddings from silver_embed.py. Idempotent: only images missing from silver.wetness are scored.
"""
from __future__ import annotations

from datetime import datetime

import pyarrow as pa
from pyducklake import FloatType, Schema, StringType, TimestampTZType, required

from common import BKK, open_catalog
from wetness_eval import MODEL_ID, PROMPTS_DRY, PROMPTS_WET, load_embeddings, text_embeddings, zero_shot_scores

TABLE = ("silver", "wetness")
VERSION = "zeroshot-v1"
SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("p_wet", FloatType()),
    required("method", StringType()),
    required("model", StringType()),
    required("scored_at", TimestampTZType()),
)


def run() -> dict:
    catalog = open_catalog()
    if "silver" not in catalog.list_namespaces():
        catalog.create_namespace("silver")
    try:
        table = catalog.load_table(TABLE)
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)
    con = catalog.connection

    todo = con.sql(
        """
        SELECT f.image_id FROM bkk.silver.image_features f
        WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.wetness w WHERE w.image_id = f.image_id)
        """
    ).df()["image_id"].tolist()
    if not todo:
        summary = {"scored": 0, "table_rows": table.scan().count()}
        print(summary)
        return summary

    ids, emb = load_embeddings(todo)
    missing = len(todo) - len(ids)
    if missing:
        raise SystemExit(f"{missing} images have no embedding: run silver_embed.py first")
    scores = zero_shot_scores(emb, text_embeddings(PROMPTS_WET + PROMPTS_DRY))

    tz = pa.timestamp("us", tz="Asia/Bangkok")
    now = datetime.now(BKK)
    batch = pa.table(
        {
            "image_id": pa.array(ids, pa.string()),
            "p_wet": pa.array(scores, pa.float32()),
            "method": pa.array([VERSION] * len(ids), pa.string()),
            "model": pa.array([MODEL_ID] * len(ids), pa.string()),
            "scored_at": pa.array([now] * len(ids), tz),
        }
    )
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"wetness: {len(ids)} images", author="silver_wetness.py")
        txn.load_table(TABLE).append(batch)
    summary = {"scored": len(ids), "table_rows": table.scan().count()}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
