"""Pretraitement. action=plan recense et nomme, action=resize redimensionne un lot."""

from __future__ import annotations

import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from PIL import Image

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "64"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")

# le pool doit couvrir le parallelisme sinon les threads attendent une connexion
S3 = boto3.client("s3", config=Config(max_pool_connections=MAX_WORKERS + 4, retries={"max_attempts": 5}))


def _list_images(bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or item["Size"] == 0:
                continue
            if key.lower().endswith(IMAGE_SUFFIXES):
                keys.append(key)
    return sorted(keys)


def _plan(event):
    bucket = event["bucket"]
    splits = event.get("splits", ["train", "test"])
    classes = event.get("classes", ["comics", "manga"])

    run_id = event.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # un nom de ressource sagemaker n'accepte qu'alphanumerique et tirets
    if not re.fullmatch(r"[A-Za-z0-9-]{1,40}", run_id):
        raise ValueError(f"run_id invalide pour une ressource SageMaker : {run_id!r}")

    shards, totals = [], {}
    for split in splits:
        for label in classes:
            source_prefix = f"{split}/{label}/"
            keys = _list_images(bucket, source_prefix)
            if not keys:
                raise ValueError(f"Aucune image sous s3://{bucket}/{source_prefix}")
            shards.append(
                {
                    "bucket": bucket,
                    "run_id": run_id,
                    "split": split,
                    "label": label,
                    "source_prefix": source_prefix,
                    "target_prefix": f"processed/{run_id}/{split}/{label}/",
                    "count": len(keys),
                }
            )
            totals[f"{split}/{label}"] = len(keys)
            LOGGER.info("%s : %d images", source_prefix, len(keys))

    return {
        "run_id": run_id,
        "bucket": bucket,
        "classes": classes,
        "shards": shards,
        "counts": totals,
        "total_images": sum(totals.values()),
        "train_s3_uri": f"s3://{bucket}/processed/{run_id}/train/",
        "test_prefix": f"processed/{run_id}/test/",
        "code_s3_uri": f"s3://{bucket}/scripts/",
        "model_output_s3_uri": f"s3://{bucket}/models/{run_id}/",
        "model_artifact_s3_uri": f"s3://{bucket}/models/{run_id}/model.tar.gz",
        "output_prefix": f"output/test_jury/{run_id}/",
        "processing_job_name": f"comics-manga-train-{run_id}",
        "model_name": f"comics-manga-{run_id}",
        "endpoint_config_name": f"comics-manga-cfg-{run_id}",
        "endpoint_name": f"comics-manga-ep-{run_id}",
    }


def _resize_one(bucket: str, key: str, target_prefix: str) -> bool:
    body = S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with Image.open(io.BytesIO(body)) as img:
        resized = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # png sans perte : le jpeg ajouterait des artefacts qui faussent les couleurs
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    # « a.jpg » -> « a.jpg.png » : pas de collision, nom d'origine recuperable
    name = key.rsplit("/", 1)[-1]
    S3.put_object(
        Bucket=bucket,
        Key=f"{target_prefix}{name}.png",
        Body=buffer.getvalue(),
        ContentType="image/png",
        Metadata={"source-key": key},
    )
    return True


def _resize(event):
    bucket, source_prefix = event["bucket"], event["source_prefix"]
    target_prefix = event["target_prefix"]
    keys = _list_images(bucket, source_prefix)

    def work(key):
        try:
            _resize_one(bucket, key, target_prefix)
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("image ignoree %s : %r", key, exc)
            return {"key": key, "erreur": repr(exc)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        failures = [item for item in pool.map(work, keys) if item is not None]

    processed = len(keys) - len(failures)
    LOGGER.info("%s -> %s : %d traitees, %d ignorees", source_prefix, target_prefix, processed, len(failures))

    if processed == 0:
        raise RuntimeError(f"Aucune image n'a pu etre traitee pour {source_prefix}")

    return {
        "split": event["split"],
        "label": event["label"],
        "target_prefix": target_prefix,
        "processed": processed,
        "skipped": len(failures),
        "failures": failures[:20],
    }


def lambda_handler(event, context):
    action = event.get("action", "plan")
    LOGGER.info("action=%s", action)

    if action == "plan":
        return _plan(event)
    if action == "resize":
        return _resize(event)
    raise ValueError(f"Action inconnue : {action!r} (attendu 'plan' ou 'resize')")
