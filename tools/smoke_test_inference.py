"""Rejoue le demarrage du conteneur de service hors AWS.

    python tools/smoke_test_inference.py <model.tar.gz|repertoire> <image...>
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_artifact(source: Path) -> Path:
    """Retourne un repertoire contenant model.json et code/."""
    if source.is_dir():
        return source

    target = Path(tempfile.mkdtemp(prefix="artefact-"))
    with tarfile.open(source, "r:gz") as tar:
        tar.extractall(target)
    return target


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    artifact = load_artifact(Path(argv[0]))
    images = argv[1:]

    code_dir = artifact / "code"
    if not code_dir.is_dir():
        print(f"ECHEC : {artifact} ne contient pas de repertoire code/")
        return 1
    for expected in ("inference.py", "features.py"):
        if not (code_dir / expected).is_file():
            print(f"ECHEC : {expected} absent de l'artefact")
            return 1
    if not (artifact / "model.json").is_file():
        print(f"ECHEC : model.json absent de {artifact}")
        return 1
    print(f"artefact  : {artifact}")

    # Les variables telles que SageMaker les injecte dans le conteneur.
    os.environ["SAGEMAKER_CONTAINER_LOG_LEVEL"] = "20"
    os.environ["SAGEMAKER_PROGRAM"] = "inference.py"
    os.environ["SAGEMAKER_REGION"] = os.environ.get("AWS_REGION", "eu-central-1")

    sys.path.insert(0, str(code_dir))
    import inference  # noqa: PLC0415 - l'import est justement ce qu'on teste

    print("import    : OK (SAGEMAKER_CONTAINER_LOG_LEVEL='20' accepte)")

    model = inference.model_fn(str(artifact))
    print(f"model_fn  : OK, classes={model['classes']}, {model['mean'].shape[0]} descripteurs")

    if not images:
        print("\nAucune image fournie : la partie prediction est ignoree.")
        return 0

    failures = 0
    for path in images:
        raw = Path(path).read_bytes()

        vector = inference.input_fn(raw, "application/x-image")
        prediction = inference.predict_fn(vector, model)
        body, content_type = inference.output_fn(prediction)

        parsed = json.loads(body)
        total = sum(parsed["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            print(f"ECHEC : les probabilites de {path} somment a {total}")
            failures += 1
        if content_type != "application/json":
            print(f"ECHEC : type de reponse inattendu {content_type}")
            failures += 1

        print(
            f"  {Path(path).name:38s} -> {parsed['predicted_label']:7s} "
            f"(confiance {parsed['confidence']:.3f})"
        )

    # Le chemin JSON/base64, utile depuis la console SageMaker.
    import base64

    encoded = base64.b64encode(Path(images[0]).read_bytes()).decode()
    vector = inference.input_fn(json.dumps({"image_base64": encoded}), "application/json")
    print(f"input_fn  : OK sur application/json ({vector.shape[1]} descripteurs)")

    # Une image tronquee doit produire une erreur nette, pas un plantage du serveur.
    try:
        inference.input_fn(b"pas une image", "application/x-image")
    except Exception as exc:  # noqa: BLE001
        print(f"robustesse: OK, entree invalide rejetee ({type(exc).__name__})")
    else:
        print("ECHEC : une entree invalide a ete acceptee")
        failures += 1

    print("\n" + ("ECHEC" if failures else "Tous les controles passent."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
