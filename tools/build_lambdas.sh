#!/usr/bin/env bash
#
# Construit build/lambda/<nom>/, source des archives Terraform.
# Les dependances sont installees pour Amazon Linux x86_64, pas pour la machine locale.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/lambda"
PYTHON_VERSION="3.13"
PLATFORM="manylinux2014_x86_64"

# tableau : le chemin du projet contient des espaces (iCloud)
if [[ -n "${PIP:-}" ]]; then
  read -r -a PIP_CMD <<<"${PIP}"
elif [[ -x "${ROOT}/.venv/bin/pip" ]]; then
  PIP_CMD=("${ROOT}/.venv/bin/pip")
else
  PIP_CMD=(python3 -m pip)
fi

rm -rf "${BUILD}"

for source_dir in "${ROOT}"/lambdas/*/; do
  name="$(basename "${source_dir}")"
  target="${BUILD}/${name}"
  echo "==> ${name}"
  mkdir -p "${target}"

  cp "${source_dir}lambda_function.py" "${target}/"

  requirements="${source_dir}requirements.txt"
  if [[ -s "${requirements}" ]]; then
    # --only-binary evite une compilation locale qui produirait un binaire macOS
    "${PIP_CMD[@]}" install \
      --quiet \
      --disable-pip-version-check \
      --target "${target}" \
      --platform "${PLATFORM}" \
      --python-version "${PYTHON_VERSION}" \
      --implementation cp \
      --only-binary=:all: \
      --upgrade \
      -r "${requirements}"
  fi

  find "${target}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  find "${target}" -type d -name "*.dist-info" -prune -exec rm -rf {} + 2>/dev/null || true
  find "${target}" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true

  echo "    $(du -sh "${target}" | cut -f1) dans ${target#"${ROOT}/"}"
done

echo "Termine."
