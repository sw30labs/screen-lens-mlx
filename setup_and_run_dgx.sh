#!/usr/bin/env bash
# Bootstrap and run ScreenLens on NVIDIA DGX Spark.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"

# Load only supported literal KEY=value settings. The repository .env is never
# executed as shell code, and an already-exported nonempty value wins.
load_repo_dotenv() {
  local env_file="${REPO_ROOT}/.env"
  local line key value first last
  local -A supported=(
    [HF_TOKEN]=1
    [HF_HUB_DISABLE_XET]=1
    [DGX_VENV_DIR]=1
    [DGX_PYTHON_BIN]=1
    [DGX_HF_CACHE]=1
    [DGX_VLLM_CACHE]=1
    [PYTORCH_DGX_INDEX_URL]=1
    [VLLM_IMAGE]=1
    [VLLM_MODEL]=1
    [VLLM_MODEL_REVISION]=1
    [VLLM_QUANTIZATION]=1
    [VLLM_KV_CACHE_DTYPE]=1
    [VLLM_BASE_URL]=1
    [VLLM_API_KEY]=1
    [VLLM_GPU_MEMORY_UTILIZATION]=1
    [VLLM_MAX_MODEL_LEN]=1
    [VLLM_START_TIMEOUT]=1
    [VLLM_LOG_TAIL]=1
  )
  local -A inherited=()

  [[ -f "${env_file}" ]] || return 0

  for key in "${!supported[@]}"; do
    if [[ -n "$(printenv "${key}" 2>/dev/null)" ]]; then
      inherited["${key}"]=1
    fi
  done

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    if [[ ! "${line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      continue
    fi

    key="${BASH_REMATCH[2]}"
    value="${BASH_REMATCH[3]}"
    [[ -n "${supported[${key}]:-}" ]] || continue
    [[ -z "${inherited[${key}]:-}" ]] || continue

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ -n "${value}" ]]; then
      first="${value:0:1}"
      last="${value: -1}"
      if [[ "${first}" == '"' || "${first}" == "'" ]]; then
        if [[ "${#value}" -lt 2 || "${last}" != "${first}" ]]; then
          echo "error: .env has an unmatched quote for ${key}" >&2
          exit 1
        fi
        value="${value:1:${#value}-2}"
      elif [[ "${last}" == '"' || "${last}" == "'" ]]; then
        echo "error: .env has an unmatched quote for ${key}" >&2
        exit 1
      elif [[ "${value}" =~ [[:space:]]# ]]; then
        echo "error: .env uses an inline comment for ${key}; use a separate line" >&2
        exit 1
      fi
    fi

    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done < "${env_file}"
}

load_repo_dotenv

COMPOSE_FILE="${REPO_ROOT}/compose.dgx-spark.yaml"
VENV_DIR="${DGX_VENV_DIR:-${REPO_ROOT}/.venv-dgx}"
PYTHON_BIN="${DGX_PYTHON_BIN:-python3.12}"
DGX_HF_CACHE="${DGX_HF_CACHE:-${REPO_ROOT}/.local-models/huggingface}"
DGX_VLLM_CACHE="${DGX_VLLM_CACHE:-${REPO_ROOT}/.local-models/vllm}"
VLLM_MODEL="${VLLM_MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
VLLM_MODEL_REVISION="${VLLM_MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-local}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-1800}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
TORCH_VERSION="2.11.0+cu130"
TORCHVISION_VERSION="0.26.0+cu130"
TORCH_INDEX_URL="${PYTORCH_DGX_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${REPO_ROOT}/${VENV_DIR}"
fi
if [[ "${DGX_HF_CACHE}" != /* ]]; then
  DGX_HF_CACHE="${REPO_ROOT}/${DGX_HF_CACHE}"
fi
if [[ "${DGX_VLLM_CACHE}" != /* ]]; then
  DGX_VLLM_CACHE="${REPO_ROOT}/${DGX_VLLM_CACHE}"
fi

export VLLM_MODEL VLLM_MODEL_REVISION VLLM_QUANTIZATION VLLM_KV_CACHE_DTYPE
export VLLM_BASE_URL VLLM_API_KEY
export VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN
export DGX_HF_CACHE DGX_VLLM_CACHE HF_HUB_DISABLE_XET

usage() {
  cat <<'EOF'
Usage: ./setup_and_run_dgx.sh COMMAND [ARGS...]

Commands:
  doctor    Check DGX Spark, Python, Docker/Compose, GPU runtime, disk, and token.
  setup     Create .venv-dgx and install the checked CUDA 13 ScreenLens stack.
  llm-up    Reuse an exact-model endpoint, or start the repository vLLM service.
  llm-wait  Wait for VLLM_BASE_URL to serve the exact configured model.
  llm-logs  Follow this repository's vLLM container logs.
  llm-down  Stop this repository's vLLM container; shared services are untouched.
  smoke     Read "test.mov" from assets/ingest-demo.png through ScreenLens.
  run       Ensure vLLM, export DGX defaults, and invoke python -m src.cli.
  help      Show this help.

Examples:
  (umask 077; touch .env)
  chmod 600 .env
  ${EDITOR:-nano} .env            # add HF_TOKEN=hf_...
  ./setup_and_run_dgx.sh doctor
  ./setup_and_run_dgx.sh setup
  ./setup_and_run_dgx.sh llm-up
  ./setup_and_run_dgx.sh smoke
  ./setup_and_run_dgx.sh run        # launches the TUI
  ./setup_and_run_dgx.sh run ingest input-videos/demo.mov

Supported .env/export overrides:
  HF_TOKEN, HF_HUB_DISABLE_XET, DGX_VENV_DIR, DGX_PYTHON_BIN,
  DGX_HF_CACHE, DGX_VLLM_CACHE, PYTORCH_DGX_INDEX_URL,
  VLLM_IMAGE, VLLM_MODEL, VLLM_MODEL_REVISION, VLLM_QUANTIZATION,
  VLLM_KV_CACHE_DTYPE, VLLM_BASE_URL,
  VLLM_API_KEY, VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN,
  VLLM_START_TIMEOUT, and VLLM_LOG_TAIL.

The helper parses these values literally and never sources .env. Exported
nonempty values take precedence. Put comments on their own lines.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

note() {
  echo "[dgx-spark] $*"
}

is_python_312() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
    >/dev/null 2>&1
}

require_python_312() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die \
    "Python 3.12 is required; install python3.12 or set DGX_PYTHON_BIN."
  is_python_312 "${PYTHON_BIN}" || die \
    "${PYTHON_BIN} is not Python 3.12; set DGX_PYTHON_BIN appropriately."
}

require_venv() {
  [[ -x "${VENV_DIR}/bin/python" ]] || die \
    "the DGX environment is missing; run './setup_and_run_dgx.sh setup'."
  is_python_312 "${VENV_DIR}/bin/python" || die \
    "${VENV_DIR} is not a Python 3.12 environment; move it aside and rerun setup."
}

docker_access_error() {
  if [[ -S /var/run/docker.sock && ! -w /var/run/docker.sock ]]; then
    echo "error: the current user cannot write to /var/run/docker.sock." >&2
    echo "Add the user to Docker's group, then start a fresh login session:" >&2
    echo "  sudo usermod -aG docker \"${USER:-$(id -un)}\"" >&2
  else
    echo "error: Docker is unavailable to the current user." >&2
    echo "Start Docker and verify 'docker info'." >&2
  fi
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is required on DGX Spark."
  docker compose version >/dev/null 2>&1 || die \
    "the Docker Compose plugin is required."
  if ! docker info >/dev/null 2>&1; then
    docker_access_error
    exit 1
  fi
}

valid_hf_token() {
  [[ -n "${HF_TOKEN:-}" \
    && "${HF_TOKEN}" != "hf_replace_me" \
    && "${HF_TOKEN}" != "your-hugging-face-token" ]]
}

require_hf_token() {
  valid_hf_token || die \
    "HF_TOKEN is missing or still a placeholder; add a Hugging Face read token to .env."
}

# Compose interpolates HF_TOKEN for every command. Control operations receive a
# harmless sentinel, while operations that can start the service require the
# real token. Implicit Compose .env loading stays disabled.
compose_control() {
  COMPOSE_DISABLE_ENV_FILE=1 \
    HF_TOKEN="${HF_TOKEN:-compose-control-only}" \
    docker compose --project-directory "${REPO_ROOT}" -f "${COMPOSE_FILE}" "$@"
}

compose_with_token() {
  require_hf_token
  COMPOSE_DISABLE_ENV_FILE=1 \
    docker compose --project-directory "${REPO_ROOT}" -f "${COMPOSE_FILE}" "$@"
}

api_python() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    printf '%s\n' "${VENV_DIR}/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    return 1
  fi
}

api_ready() {
  local python
  python="$(api_python)" || return 1
  "${python}" - "${VLLM_BASE_URL%/}/models" "${VLLM_MODEL}" \
    "${VLLM_MAX_MODEL_LEN}" <<'PY' >/dev/null 2>&1
import json
import os
import sys
import urllib.request

url, expected, required_context = sys.argv[1:]
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'local')}"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=3) as response:
    payload = json.load(response)
for row in payload.get("data", []):
    if not isinstance(row, dict) or str(row.get("id", "")) != expected:
        continue
    try:
        advertised_context = int(row.get("max_model_len"))
    except (TypeError, ValueError):
        raise SystemExit(1)
    raise SystemExit(0 if advertised_context >= int(required_context) else 1)
raise SystemExit(1)
PY
}

api_models_reachable() {
  local python
  python="$(api_python)" || return 1
  "${python}" - "${VLLM_BASE_URL%/}/models" <<'PY' >/dev/null 2>&1
import json
import os
import sys
import urllib.request

request = urllib.request.Request(
    sys.argv[1],
    headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'local')}"},
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=3) as response:
    payload = json.load(response)
raise SystemExit(0 if isinstance(payload.get("data"), list) else 1)
PY
}

managed_endpoint() {
  [[ "${VLLM_BASE_URL%/}" == "http://127.0.0.1:8000/v1" ]]
}

default_port_busy() {
  local python
  python="$(api_python)" || return 1
  "${python}" - <<'PY' >/dev/null 2>&1
import socket

with socket.create_connection(("127.0.0.1", 8000), timeout=1):
    pass
PY
}

local_container_id() {
  compose_control ps --all -q llm 2>/dev/null
}

local_container_running() {
  [[ -n "$(compose_control ps --status running -q llm 2>/dev/null || true)" ]]
}

validate_timeout() {
  [[ "${VLLM_START_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die \
    "VLLM_START_TIMEOUT must be a positive integer number of seconds."
}

cmd_doctor() {
  local failures=0 output runtimes architecture available_kb available_gib
  local shared_ready=0

  if api_ready; then
    shared_ready=1
    note "exact model is already ready at ${VLLM_BASE_URL}; it can be reused"
  fi

  if [[ "$(uname -s)" == "Linux" ]]; then
    note "Linux detected"
  else
    echo "[error] DGX Spark requires Linux (found $(uname -s))" >&2
    failures=$((failures + 1))
  fi

  architecture="$(uname -m)"
  if [[ "${architecture}" == "aarch64" || "${architecture}" == "arm64" ]]; then
    note "ARM64 host detected (${architecture})"
  else
    echo "[error] DGX Spark requires ARM64 (found ${architecture})" >&2
    failures=$((failures + 1))
  fi

  if command -v "${PYTHON_BIN}" >/dev/null 2>&1 && is_python_312 "${PYTHON_BIN}"; then
    note "Python 3.12 detected at $(command -v "${PYTHON_BIN}")"
    if "${PYTHON_BIN}" -c 'import ensurepip, venv' >/dev/null 2>&1; then
      note "Python venv/ensurepip support detected"
    else
      echo "[error] ${PYTHON_BIN} lacks venv/ensurepip support" >&2
      failures=$((failures + 1))
    fi
  else
    echo "[error] Python 3.12 was not found (checked ${PYTHON_BIN})" >&2
    failures=$((failures + 1))
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && output="$(nvidia-smi -L 2>/dev/null)"; then
    if [[ "${output}" == *GB10* ]]; then
      note "NVIDIA GB10 GPU detected"
    else
      echo "[error] NVIDIA is visible, but a GB10 GPU was not detected" >&2
      failures=$((failures + 1))
    fi
  else
    echo "[error] nvidia-smi cannot communicate with the DGX Spark GPU" >&2
    failures=$((failures + 1))
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "[error] Docker is not installed" >&2
    failures=$((failures + 1))
  elif ! docker compose version >/dev/null 2>&1; then
    echo "[error] the Docker Compose plugin is not installed" >&2
    failures=$((failures + 1))
  elif ! docker info >/dev/null 2>&1; then
    docker_access_error
    failures=$((failures + 1))
  else
    note "Docker daemon and Compose are accessible"
    runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    if [[ "${runtimes}" == *nvidia* ]]; then
      note "NVIDIA Container Runtime detected"
    else
      echo "[error] Docker does not report an NVIDIA runtime" >&2
      echo "  sudo nvidia-ctk runtime configure --runtime=docker" >&2
      echo "  sudo systemctl restart docker" >&2
      failures=$((failures + 1))
    fi
    if compose_control config --quiet >/dev/null 2>&1; then
      note "DGX Compose configuration is valid"
    else
      echo "[error] ${COMPOSE_FILE} is not accepted by this Compose version" >&2
      failures=$((failures + 1))
    fi
  fi

  if valid_hf_token; then
    note "HF_TOKEN is set (value hidden)"
  elif (( shared_ready == 1 )); then
    note "HF_TOKEN is not needed while reusing the ready external service"
  else
    echo "[error] HF_TOKEN is missing or still a placeholder" >&2
    failures=$((failures + 1))
  fi

  if (( shared_ready == 0 )); then
    if api_models_reachable; then
      echo "[error] ${VLLM_BASE_URL} serves a model/context that does not match " \
        "${VLLM_MODEL} (${VLLM_MAX_MODEL_LEN} tokens required)" >&2
      failures=$((failures + 1))
    elif managed_endpoint && default_port_busy; then
      echo "[error] port 8000 is occupied but its model API is not usable" >&2
      failures=$((failures + 1))
    elif ! managed_endpoint; then
      echo "[error] configured external endpoint ${VLLM_BASE_URL} is not ready" >&2
      failures=$((failures + 1))
    fi
  fi

  available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print $4}')"
  if [[ "${available_kb}" =~ ^[0-9]+$ ]]; then
    available_gib=$((available_kb / 1024 / 1024))
    note "${available_gib} GiB free in the project filesystem"
    if (( available_gib < 70 )); then
      echo "[warning] less than 70 GiB is free; initial image/model downloads may fail" >&2
    fi
  fi

  if (( failures > 0 )); then
    echo "DGX Spark doctor found ${failures} blocking issue(s)." >&2
    return 1
  fi
  note "doctor passed"
}

cmd_setup() {
  require_python_312
  mkdir -p "${DGX_HF_CACHE}" "${DGX_VLLM_CACHE}"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    note "creating ${VENV_DIR} with Python 3.12"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  elif ! is_python_312 "${VENV_DIR}/bin/python"; then
    die "${VENV_DIR} uses another Python; move it aside and rerun setup."
  else
    note "reusing Python 3.12 environment at ${VENV_DIR}"
  fi

  local python="${VENV_DIR}/bin/python"
  note "updating packaging tools"
  "${python}" -m pip install --upgrade pip "setuptools<82" wheel
  note "installing PyTorch ${TORCH_VERSION} and torchvision ${TORCHVISION_VERSION}"
  "${python}" -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"
  note "installing ScreenLens development and TUI dependencies"
  "${python}" -m pip install -e "${REPO_ROOT}[dev,tui]"

  note "running CUDA, torchvision, and OpenCLIP preflight"
  "${python}" - "${TORCH_VERSION}" "${TORCHVISION_VERSION}" <<'PY'
import importlib.metadata as metadata
import sys

import open_clip
import torch
import torchvision

expected_torch, expected_torchvision = sys.argv[1:]
if torch.__version__ != expected_torch:
    raise SystemExit(f"expected torch {expected_torch}, found {torch.__version__}")
if torchvision.__version__ != expected_torchvision:
    raise SystemExit(
        f"expected torchvision {expected_torchvision}, found {torchvision.__version__}"
    )
if torch.version.cuda != "13.0":
    raise SystemExit(f"expected a CUDA 13.0 torch build, found CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to PyTorch; verify the DGX driver")

device = torch.device("cuda:0")
left = torch.ones((64, 64), device=device, dtype=torch.float16)
right = torch.ones((64, 64), device=device, dtype=torch.float16)
product = left @ right
torch.cuda.synchronize(device)
if product[0, 0].item() != 64.0:
    raise SystemExit("CUDA matrix preflight returned an invalid result")

# Construct the same library family used by ScreenLens embeddings without
# downloading weights, then execute a real image encoder pass on CUDA.
model = open_clip.create_model("ViT-B-32", pretrained=None).eval().to(device)
image = torch.zeros((1, 3, 224, 224), device=device)
with torch.inference_mode():
    features = model.encode_image(image)
torch.cuda.synchronize(device)
if features.ndim != 2 or features.shape[0] != 1 or not torch.isfinite(features).all():
    raise SystemExit("OpenCLIP CUDA image-encoder preflight returned invalid features")

name = torch.cuda.get_device_name(device)
major, minor = torch.cuda.get_device_capability(device)
print(
    f"[dgx-spark] CUDA/OpenCLIP preflight passed: {name}, "
    f"capability {major}.{minor}, torch {torch.__version__}, "
    f"torchvision {torchvision.__version__}, "
    f"open-clip-torch {metadata.version('open-clip-torch')}"
)
PY

  local check_output check_status=0
  check_output="$("${python}" -m pip check 2>&1)" || check_status=$?
  if (( check_status != 0 )); then
    # The official CUDA 13 ARM64 index currently embeds an SBSA wheel tag that
    # pip check misreports. Accept only this exact warning after the real CUDA
    # and OpenCLIP operations above; all other dependency failures remain fatal.
    local known_sbsa_warning
    known_sbsa_warning="nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
    if [[ "${check_output}" == "${known_sbsa_warning}" ]]; then
      note "ignoring the known cuSPARSELt SBSA tag warning after CUDA preflight"
    else
      printf '%s\n' "${check_output}" >&2
      return "${check_status}"
    fi
  else
    printf '%s\n' "${check_output}"
  fi
  note "setup complete"
}

cmd_llm_wait() {
  validate_timeout
  local started elapsed=0 local_id=""
  started="${SECONDS}"
  note "waiting up to ${VLLM_START_TIMEOUT}s for ${VLLM_MODEL} at ${VLLM_BASE_URL}"

  while (( elapsed < VLLM_START_TIMEOUT )); do
    if api_ready; then
      note "exact-model vLLM endpoint is ready"
      return 0
    fi
    if api_models_reachable; then
      echo "error: ${VLLM_BASE_URL} is serving models, but not ${VLLM_MODEL}." >&2
      echo "Refusing to treat a different model as a ready ScreenLens backend." >&2
      return 1
    fi

    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
      local_id="$(local_container_id || true)"
      if [[ -n "${local_id}" ]] && ! local_container_running; then
        echo "error: this repository's vLLM container exited while loading." >&2
        echo "Inspect it with './setup_and_run_dgx.sh llm-logs'." >&2
        return 1
      fi
    fi

    if managed_endpoint && [[ -z "${local_id}" ]]; then
      if default_port_busy; then
        echo "error: port 8000 is owned by an external service whose model API " \
          "is not usable." >&2
        echo "Wait for or fix that service at its owner, then rerun this command." >&2
      else
        echo "error: no vLLM service is listening at ${VLLM_BASE_URL}." >&2
        echo "Start it with './setup_and_run_dgx.sh llm-up'." >&2
      fi
      return 1
    fi

    sleep 5
    elapsed=$((SECONDS - started))
    if (( elapsed > 0 && elapsed % 30 < 5 )); then
      note "vLLM is still loading (${elapsed}s elapsed)"
    fi
  done

  echo "error: ${VLLM_MODEL} was not ready within ${VLLM_START_TIMEOUT}s." >&2
  return 1
}

cmd_llm_up() {
  if api_ready; then
    note "reusing ready exact-model service at ${VLLM_BASE_URL}"
    return 0
  fi
  if api_models_reachable; then
    die "${VLLM_BASE_URL} serves a different model or a smaller context; refusing reuse."
  fi
  if ! managed_endpoint; then
    die "configured external endpoint ${VLLM_BASE_URL} is not ready; start it at its owner."
  fi
  if default_port_busy; then
    die "port 8000 is occupied but its model API is not usable; fix it at its owner."
  fi

  require_docker
  require_hf_token
  mkdir -p "${DGX_HF_CACHE}" "${DGX_VLLM_CACHE}"
  note "starting ${VLLM_MODEL} on loopback port 8000"
  compose_with_token up --detach --remove-orphans llm
  cmd_llm_wait
}

cmd_llm_down() {
  require_docker
  local container_id
  container_id="$(local_container_id || true)"
  if [[ -z "${container_id}" ]]; then
    note "no ScreenLens-owned vLLM container exists; shared services are untouched"
    return 0
  fi
  note "stopping the ScreenLens vLLM stack (host caches are preserved)"
  compose_control down --remove-orphans
}

cmd_llm_logs() {
  require_docker
  local tail_lines="${VLLM_LOG_TAIL:-200}" container_id
  [[ "${tail_lines}" =~ ^[1-9][0-9]*$ ]] || die \
    "VLLM_LOG_TAIL must be a positive integer."
  container_id="$(local_container_id || true)"
  [[ -n "${container_id}" ]] || die \
    "no ScreenLens-owned container exists; inspect the shared service at its owner."
  compose_control logs --follow --tail "${tail_lines}" llm
}

cmd_smoke() {
  require_venv
  cmd_llm_wait
  local image_path="${REPO_ROOT}/assets/ingest-demo.png"
  [[ -f "${image_path}" ]] || die "vision fixture is missing: ${image_path}"

  note "running a real ScreenLens vision request against ingest-demo.png"
  (
    cd "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" - \
      "${image_path}" "${VLLM_BASE_URL}" "${VLLM_MODEL}" "${VLLM_API_KEY}" <<'PY'
import sys

from src.omlx_client import InferenceClient

image_path, base_url, model, api_key = sys.argv[1:]
client = InferenceClient.from_endpoint(
    base_url=base_url,
    model=model,
    api_key=api_key,
    backend="vllm",
    timeout=180.0,
    default_max_tokens=128,
    default_temperature=0.0,
)
answer = client.chat(
    "You are a precise screen OCR system. Return only text visible in the image.",
    "Read the video filename shown in the terminal command and configuration panel.",
    images=[image_path],
    max_tokens=128,
    temperature=0.0,
    extra={"chat_template_kwargs": {"enable_thinking": False}},
)
if "test.mov" not in answer.casefold():
    raise SystemExit(
        "vision smoke failed: expected the image response to contain 'test.mov'; "
        f"received {answer[:240]!r}"
    )
print(f"[dgx-spark] ScreenLens vision smoke passed ({model} read test.mov)")
PY
  )
}

cmd_run() {
  require_venv
  cmd_llm_up
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ $# -eq 0 ]]; then
    set -- tui
  fi

  # Force the bounded DGX execution profile. The vLLM server admits at most two
  # sequences, so ScreenLens should not fan out more concurrent image requests.
  export VLLM_MODEL VLLM_BASE_URL VLLM_API_KEY
  export HF_HOME="${DGX_HF_CACHE}"
  if ! valid_hf_token; then
    unset HF_TOKEN
  fi
  export SCREENLENS_BACKEND="vllm"
  export SCREENLENS_DEVICE="cuda"
  export SCREENLENS_BATCH_SIZE="2"

  note "running ScreenLens with vLLM vision, CUDA OpenCLIP, and concurrency 2"
  cd "${REPO_ROOT}"
  exec "${VENV_DIR}/bin/python" -m src.cli "$@"
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  doctor) cmd_doctor "$@" ;;
  setup) cmd_setup "$@" ;;
  llm-up) cmd_llm_up "$@" ;;
  llm-wait) cmd_llm_wait "$@" ;;
  llm-logs) cmd_llm_logs "$@" ;;
  llm-down) cmd_llm_down "$@" ;;
  smoke) cmd_smoke "$@" ;;
  run) cmd_run "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "error: unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
