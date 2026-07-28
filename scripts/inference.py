"""Handlers SageMaker. Prediction reimplementee en numpy : aucun pickle sklearn."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import extract_features  # noqa: E402

LOGGER = logging.getLogger(__name__)

# la variable vaut "20", que setLevel refuse : il lui faut l'entier 20 ou "INFO"
_LOG_LEVEL = os.environ.get("SAGEMAKER_CONTAINER_LOG_LEVEL", "INFO")
LOGGER.setLevel(int(_LOG_LEVEL) if str(_LOG_LEVEL).lstrip("-").isdigit() else _LOG_LEVEL)

JSON_CONTENT_TYPE = "application/json"
IMAGE_CONTENT_TYPES = {
    "application/x-image",
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}


def model_fn(model_dir):
    artifact_path = Path(model_dir) / "model.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    if payload.get("format_version") != 1:
        raise ValueError(f"Version d'artefact non supportee : {payload.get('format_version')!r}")

    model = {
        "classes": [str(label) for label in payload["classes"]],
        "feature_names": payload["feature_names"],
        "mean": np.asarray(payload["scaler_mean"], dtype=np.float64),
        "scale": np.asarray(payload["scaler_scale"], dtype=np.float64),
        "coefficients": np.asarray(payload["coefficients"], dtype=np.float64),
        "intercept": np.asarray(payload["intercept"], dtype=np.float64),
        "metadata": payload.get("metadata", {}),
    }

    LOGGER.info(
        "modele charge : %d classes, %d descripteurs, entraine le %s",
        len(model["classes"]),
        model["mean"].shape[0],
        model["metadata"].get("trained_at", "date inconnue"),
    )
    return model


def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    """Octets bruts de l'image, ou {"image_base64": "..."} pour les tests manuels."""
    media_type = (content_type or "").split(";")[0].strip().lower()

    if media_type in IMAGE_CONTENT_TYPES:
        raw = request_body if isinstance(request_body, (bytes, bytearray)) else bytes(request_body)
        return extract_features(raw).reshape(1, -1)

    if media_type == JSON_CONTENT_TYPE:
        document = json.loads(request_body)
        if "image_base64" not in document:
            raise ValueError("Le corps JSON doit contenir la cle 'image_base64'.")
        return extract_features(base64.b64decode(document["image_base64"])).reshape(1, -1)

    raise ValueError(
        f"Type de contenu non supporte : {content_type!r}. "
        f"Types acceptes : {sorted(IMAGE_CONTENT_TYPES)} ou {JSON_CONTENT_TYPE}."
    )


def _probabilities(scores):
    # en binaire sklearn n'a qu'une ligne de coefficients : sigmoide, sinon softmax
    if scores.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        return np.column_stack([1.0 - positive, positive])

    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def predict_fn(features, model):
    standardized = (features - model["mean"]) / model["scale"]
    scores = standardized @ model["coefficients"].T + model["intercept"]
    probabilities = _probabilities(scores)

    classes = model["classes"]
    predictions = []
    for row in probabilities:
        best = int(np.argmax(row))
        predictions.append(
            {
                "predicted_label": classes[best],
                "confidence": float(row[best]),
                "probabilities": {label: float(value) for label, value in zip(classes, row)},
            }
        )

    return predictions[0] if len(predictions) == 1 else predictions


def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    media_type = (accept or "").split(";")[0].strip().lower()
    if media_type in ("", "*/*", JSON_CONTENT_TYPE):
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Type de reponse non supporte : {accept!r}")


if __name__ == "__main__":
    loaded = model_fn(sys.argv[1])
    for image_path in sys.argv[2:]:
        with open(image_path, "rb") as handle:
            vector = input_fn(handle.read(), "application/x-image")
        body, _ = output_fn(predict_fn(vector, loaded))
        print(f"{image_path} -> {body}")
