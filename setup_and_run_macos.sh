#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_NAME="${SCREENLENS_CONDA_ENV:-screenlens}"
readonly PYTHON_VERSION="3.11"

if [[ "$(uname -s)" == "Linux" \
    && ( "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ) ]]; then
    printf '%s\n' \
        '[ScreenLens] DGX Spark detected. Use ./setup_and_run_dgx.sh setup, then' \
        '[ScreenLens] ./setup_and_run_dgx.sh run [COMMAND ...] so CUDA 13 wheels are pinned.' >&2
    exit 2
fi

log() {
    printf '[ScreenLens] %s\n' "$*"
}

die() {
    printf '[ScreenLens] Error: %s\n' "$*" >&2
    exit 1
}

find_conda() {
    local candidate

    if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        printf '%s\n' "$CONDA_EXE"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    for candidate in \
        "${HOME}/miniforge3/bin/conda" \
        "${HOME}/mambaforge/bin/conda" \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/anaconda3/bin/conda"
    do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

conda_env_exists() {
    "$CONDA_BIN" env list | awk -v environment="$ENV_NAME" '
        $1 == environment { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

CONDA_BIN="$(find_conda)" || die \
    "Conda was not found. Install Miniforge or Miniconda, then run this script again."
readonly CONDA_BIN

cd -- "$REPO_ROOT"

if conda_env_exists; then
    log "Using existing Conda environment: $ENV_NAME"
else
    log "Creating Conda environment: $ENV_NAME"
    "$CONDA_BIN" create --yes --name "$ENV_NAME" --channel conda-forge \
        "python=$PYTHON_VERSION" pip setuptools wheel ffmpeg
fi

if ! "$CONDA_BIN" run --name "$ENV_NAME" python -c \
    'import pip, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'
then
    log "Ensuring $ENV_NAME has Python $PYTHON_VERSION and pip"
    "$CONDA_BIN" install --yes --name "$ENV_NAME" --channel conda-forge \
        "python=$PYTHON_VERSION" pip setuptools wheel
fi

if ! "$CONDA_BIN" run --name "$ENV_NAME" ffprobe -version >/dev/null 2>&1; then
    log "Installing ffmpeg in $ENV_NAME"
    "$CONDA_BIN" install --yes --name "$ENV_NAME" --channel conda-forge ffmpeg
fi

if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    log "Created .env from .env.example; review its Apple Silicon/oMLX settings."
fi

log "Installing ScreenLens and TUI support in editable mode"
"$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" \
    python -m pip install --editable "${REPO_ROOT}[all]"

if [[ $# -eq 0 ]]; then
    set -- tui
fi

log "Launching ScreenLens"
exec "$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" \
    python -m src.cli "$@"
