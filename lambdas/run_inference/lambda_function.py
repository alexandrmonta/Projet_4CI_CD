"""Interroge l'endpoint pour chaque image de test, ecrit output/<run_id>/."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.config import Config

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))

S3 = boto3.client("s3", config=Config(max_pool_connections=MAX_WORKERS + 4))
RUNTIME = boto3.client(
    "sagemaker-runtime",
    config=Config(max_pool_connections=MAX_WORKERS + 4, retries={"max_attempts": 5, "mode": "standard"}),
)


def _list_keys(bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if not item["Key"].endswith("/") and item["Size"] > 0:
                keys.append(item["Key"])
    return sorted(keys)


def _predict(endpoint: str, bucket: str, key: str, true_label: str, classes: list[str]) -> dict:
    body = S3.get_object(Bucket=bucket, Key=key)["Body"].read()
    response = RUNTIME.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/x-image",
        Accept="application/json",
        Body=body,
    )
    payload = json.loads(response["Body"].read())
    probabilities = payload.get("probabilities", {})

    # on retire le « .png » ajoute par le pretraitement
    name = key.rsplit("/", 1)[-1]
    original = name[:-4] if name.lower().endswith(".png") and name.count(".") > 1 else name

    row = {
        "image_key": key,
        "image_name": original,
        "true_label": true_label,
        "predicted_label": payload["predicted_label"],
        "confidence": float(payload["confidence"]),
        "is_correct": payload["predicted_label"] == true_label,
        "inferred_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for label in classes:
        row[f"proba_{label}"] = float(probabilities.get(label, float("nan")))
    return row


def _metrics(frame: pd.DataFrame, classes: list[str]) -> dict:
    total = len(frame)
    accuracy = float(frame["is_correct"].mean())

    per_class, matrix = {}, []
    for actual in classes:
        subset = frame[frame["true_label"] == actual]
        predicted_as = frame[frame["predicted_label"] == actual]
        true_positive = int((subset["predicted_label"] == actual).sum())

        recall = true_positive / len(subset) if len(subset) else 0.0
        precision = true_positive / len(predicted_as) if len(predicted_as) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[actual] = {
            "support": int(len(subset)),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }
        matrix.append([int((subset["predicted_label"] == other).sum()) for other in classes])

    balanced = sum(v["recall"] for v in per_class.values()) / len(classes)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": total,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced, 4),
        "random_baseline": round(1.0 / len(classes), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "rows_are_true_labels": True, "matrix": matrix},
    }


def _write_excel(frame: pd.DataFrame, metrics: dict, path: str) -> bool:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        LOGGER.warning("openpyxl absent : le fichier XLSX n'est pas produit")
        return False

    summary = pd.DataFrame(
        [
            {"metrique": "images", "valeur": metrics["images"]},
            {"metrique": "accuracy", "valeur": metrics["accuracy"]},
            {"metrique": "accuracy equilibree", "valeur": metrics["balanced_accuracy"]},
            {"metrique": "reference aleatoire", "valeur": metrics["random_baseline"]},
        ]
        + [
            {"metrique": f"{name} / {key}", "valeur": value}
            for name, block in metrics["per_class"].items()
            for key, value in block.items()
        ]
    )
    labels = metrics["confusion_matrix"]["labels"]
    confusion = pd.DataFrame(
        metrics["confusion_matrix"]["matrix"],
        index=[f"reel {label}" for label in labels],
        columns=[f"predit {label}" for label in labels],
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="predictions", index=False)
        summary.to_excel(writer, sheet_name="synthese", index=False)
        confusion.to_excel(writer, sheet_name="matrice_confusion")
    return True


def lambda_handler(event, context):
    bucket = event["bucket"]
    endpoint = event["endpoint_name"]
    test_prefix = event["test_prefix"]
    output_prefix = event["output_prefix"]
    classes = event.get("classes", ["comics", "manga"])

    targets = []
    for label in classes:
        keys = _list_keys(bucket, f"{test_prefix}{label}/")
        LOGGER.info("%s%s/ : %d images", test_prefix, label, len(keys))
        targets.extend((key, label) for key in keys)

    if not targets:
        raise RuntimeError(f"Aucune image de test sous s3://{bucket}/{test_prefix}")

    def work(item):
        key, label = item
        try:
            return _predict(endpoint, bucket, key, label, classes)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("echec sur %s : %r", key, exc)
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = [row for row in pool.map(work, targets) if row is not None]

    failed = len(targets) - len(rows)
    if not rows:
        raise RuntimeError("Toutes les inferences ont echoue")
    LOGGER.info("%d predictions obtenues, %d echecs", len(rows), failed)

    columns = (
        ["image_key", "image_name", "true_label", "predicted_label", "confidence"]
        + [f"proba_{label}" for label in classes]
        + ["is_correct", "inferred_at"]
    )
    frame = pd.DataFrame(rows)[columns].sort_values("image_key").reset_index(drop=True)

    metrics = _metrics(frame, classes)
    metrics["failed_inferences"] = failed
    metrics["run_id"] = event.get("run_id")
    metrics["endpoint_name"] = endpoint
    LOGGER.info("accuracy=%.4f sur %d images", metrics["accuracy"], metrics["images"])

    written = {}

    parquet_path = "/tmp/predictions.parquet"
    frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    S3.upload_file(parquet_path, bucket, f"{output_prefix}predictions.parquet")
    written["parquet"] = f"s3://{bucket}/{output_prefix}predictions.parquet"

    csv_path = "/tmp/predictions.csv"
    frame.to_csv(csv_path, index=False)
    S3.upload_file(csv_path, bucket, f"{output_prefix}predictions.csv")
    written["csv"] = f"s3://{bucket}/{output_prefix}predictions.csv"

    if _write_excel(frame, metrics, "/tmp/predictions.xlsx"):
        S3.upload_file("/tmp/predictions.xlsx", bucket, f"{output_prefix}predictions.xlsx")
        written["xlsx"] = f"s3://{bucket}/{output_prefix}predictions.xlsx"

    S3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}metrics.json",
        Body=json.dumps(metrics, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    written["metrics"] = f"s3://{bucket}/{output_prefix}metrics.json"

    return {
        "run_id": event.get("run_id"),
        "images": metrics["images"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "failed_inferences": failed,
        "outputs": written,
    }
