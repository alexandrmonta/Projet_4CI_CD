"""Entrainement du classifieur comics / manga.

Tourne tel quel en Processing Job, en Training Job et en local :

    python3 train.py --train /opt/ml/processing/input/train \
                     --archive-dir /opt/ml/processing/model \
                     --output-data-dir /opt/ml/processing/output
    python3 train.py --train ./data/train --model-dir ./build/model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import IMAGE_SIZE, extract_features, feature_names  # noqa: E402

LOGGER = logging.getLogger("train")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

# pas de requirements.txt : sa presence declencherait un pip install au demarrage
# de l'endpoint alors que le conteneur a deja Pillow
ARTIFACT_FILES = ("inference.py", "features.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    # le processing job n'archive pas /opt/ml/model, contrairement au training job
    parser.add_argument("--archive-dir", default=None)
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--code-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--classes", default="comics,manga")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args(argv)


def list_dataset(root: Path, classes: list[str]) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for label in classes:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Repertoire de classe introuvable : {class_dir}")
        found = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            raise ValueError(f"Aucune image exploitable dans {class_dir}")
        LOGGER.info("classe %-8s : %4d images", label, len(found))
        samples.extend((path, label) for path in found)
    return samples


def build_matrix(samples, max_workers: int):
    def safe_extract(item):
        path, label = item
        try:
            return extract_features(str(path)), label, path, None
        except Exception as exc:  # noqa: BLE001
            # quelques fichiers corrompus sont attendus, on ne fait pas echouer le job
            return None, label, path, exc

    started = time.time()
    rows, labels, skipped = [], [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for vector, label, path, error in pool.map(safe_extract, samples):
            if error is not None:
                skipped.append({"image": str(path), "erreur": repr(error)})
                LOGGER.warning("image ignoree %s : %s", path, error)
                continue
            rows.append(vector)
            labels.append(label)

    elapsed = time.time() - started
    LOGGER.info(
        "descripteurs extraits pour %d images en %.1f s (%.0f images/s), %d ignorees",
        len(rows),
        elapsed,
        len(rows) / max(elapsed, 1e-9),
        len(skipped),
    )
    return np.vstack(rows), np.asarray(labels), skipped


def build_pipeline(args):
    # le scaler est dans le pipeline pour etre reajuste a chaque pli de la CV
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=args.regularization,
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )


def export_model(model_dir: Path, scaler, classifier, classes, names, metadata, code_dir: Path):
    payload = {
        "format_version": 1,
        "model_type": "logistic_regression",
        "image_size": IMAGE_SIZE,
        "classes": list(classes),
        "feature_names": names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "metadata": metadata,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # retrouve par SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code
    code_target = model_dir / "code"
    code_target.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACT_FILES:
        origin = code_dir / filename
        if origin.is_file():
            shutil.copy2(origin, code_target / filename)
        else:
            LOGGER.warning("fichier absent du repertoire de code, non embarque : %s", origin)


def archive_model(model_dir: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in sorted(model_dir.iterdir()):
            tar.add(item, arcname=item.name)
    LOGGER.info("archive ecrite : %s (%d octets)", archive_path, archive_path.stat().st_size)
    return archive_path


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout)
    args = parse_args(argv)
    classes = [label.strip() for label in args.classes.split(",") if label.strip()]

    LOGGER.info("scikit-learn %s | numpy %s | python %s", sklearn.__version__, np.__version__, sys.version.split()[0])
    LOGGER.info("lecture du jeu d'entrainement depuis %s", args.train)

    samples = list_dataset(Path(args.train), classes)
    features, labels, skipped = build_matrix(samples, args.max_workers)
    names = feature_names()
    LOGGER.info("matrice de descripteurs : %d observations x %d variables", *features.shape)

    # CV stratifiee plutot qu'un holdout unique, qui donnait 4 a 5 points de trop
    pipeline = build_pipeline(args)
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    scores = cross_val_score(pipeline, features, labels, cv=folds, scoring="accuracy")
    LOGGER.info("accuracy validation croisee = %.4f (+/- %.4f)", scores.mean(), scores.std())

    predicted = cross_val_predict(pipeline, features, labels, cv=folds)
    report = classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=classes)

    metrics = {
        "cross_validation_accuracy_mean": float(scores.mean()),
        "cross_validation_accuracy_std": float(scores.std()),
        "cross_validation_scores": [float(s) for s in scores],
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in classes
        },
        "confusion_matrix": {"labels": classes, "rows_are_true_labels": True, "matrix": matrix.tolist()},
    }

    # scaler et classifieur ajustes separement pour pouvoir exporter leurs coefficients
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=args.regularization, max_iter=args.max_iter, class_weight="balanced", random_state=args.seed
    ).fit(scaler.transform(features), labels)

    ordered_classes = [str(label) for label in classifier.classes_]

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "training_samples": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "class_distribution": {label: int((labels == label).sum()) for label in classes},
        "skipped_images": skipped,
        "hyperparameters": {
            "C": args.regularization,
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "seed": args.seed,
            "image_size": IMAGE_SIZE,
        },
        "metrics": metrics,
    }

    model_dir = Path(args.model_dir)
    export_model(model_dir, scaler, classifier, ordered_classes, names, metadata, Path(args.code_dir))
    LOGGER.info("artefact ecrit dans %s", model_dir / "model.json")

    if args.archive_dir:
        archive_model(model_dir, Path(args.archive_dir))

    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("rapport d'entrainement ecrit dans %s", output_dir / "training_report.json")

    print(f"final_cv_accuracy={metrics['cross_validation_accuracy_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
