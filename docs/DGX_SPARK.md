# NVIDIA DGX Spark deployment

ScreenLens supports DGX Spark through a local, OpenAI-compatible vLLM service.
It is the native default on Linux/ARM64 and does not replace the existing Apple
Silicon/oMLX workflow.

## Architecture

- vLLM runs in an ARM64 NVIDIA container and binds only to
  `http://127.0.0.1:8000/v1`.
- ScreenLens runs in `.venv-dgx` with Python 3.12, CUDA 13 PyTorch, and CUDA
  OpenCLIP embeddings.
- The default `nvidia/Qwen3.6-27B-NVFP4` checkpoint is a dense multimodal model:
  one served model handles frame captioning, verbatim OCR, and text reconstruction.
- Hugging Face downloads and vLLM compilation caches persist outside the
  container and can be shared with another project.

When ScreenLens starts its own service, the model revision is pinned to
`0893e1606ff3d5f97a441f405d5fc541a6bdf404` and the validated vLLM image is
digest-pinned. The server exposes the model's native 262,144-token context,
admits at most two sequences, and targets 45% GPU memory so the long FP8 KV
cache fits while leaving unified memory for image inference, OpenCLIP, the
operating system, and other local workloads.

ScreenLens requests captions with a 32K output ceiling. Prompt, chat-template,
image, and completion tokens share the server's 262K context, so the caption
limit leaves substantial input headroom. If `VLLM_MAX_MODEL_LEN` is deliberately
reduced to the same 32K value, the client omits `max_tokens`; vLLM then assigns
the exact context remaining after the input instead of making an impossible
zero-input reservation.
Direct captions also use light repetition controls so a malformed generation
cannot consume that entire ceiling by looping. Later reconstruction stages
greedily pack captions by serialized size and split an individually oversized
caption; they do not assume every frame caption is near the global average.
Caption failures are isolated per frame, so a timed-out request cannot discard
the successful result beside it in the same two-request chunk. ScreenLens
retries only the failed frame once with a 2,048-token ceiling (configurable as
`captioning.retry_attempts` and `captioning.retry_max_tokens`) before recording
an error marker for that frame and continuing. Retries use deterministic
decoding and must terminate naturally within the bounded ceiling; a truncated
loop is not accepted as a caption.
The shared extraction pass requests at most 1,400 output tokens while retaining
the server's entire context as completion headroom. Recursive synthesis filters
notes to the current file or artifact and uses the same full ceiling. Long
reconstruction calls have an independent 1,800-second HTTP timeout, configurable
as `reconstruction.timeout_seconds`, so they do not inherit the shorter caption
request budget. A group
that still ends with `finish_reason=length` is discarded and retried with less
input; incomplete prefixes never flow into later passes. If the endpoint and
`VLLM_MAX_MODEL_LEN` are upgraded together, reconstruction automatically uses a
larger served context such as Qwen3.6's native 262K window.

The service uses Qwen's built-in `mtp` speculative method with two draft
tokens. MTP is lossless speculative decoding: it changes throughput, not the
model's answer or context limit. DFlash is intentionally absent because it
requires an additional draft checkpoint and does not solve completion-length
failures.

## Sharing vLLM with DigitalTwin

DigitalTwin uses the same model and loopback port. Only one Compose project can
own port 8000, so `./setup_and_run_dgx.sh llm-up` checks `/v1/models` first. If an
already-running service exposes the configured model id and at least the
configured context length, ScreenLens reuses it and does not start or recreate
a container. The subsequent image smoke proves that the shared endpoint really
processes vision input.

The standard models response does not expose a Hugging Face revision or every
vLLM launch flag. When reusing a service, its owner remains responsible for the
revision and runtime settings; ScreenLens verifies the observable model id,
context length, and multimodal behavior. The pinned revision and image apply
directly when this repository owns the service.

Likewise, `run` reuses a ready service. `llm-down` and `llm-logs` operate only on
the container created by this repository; they never stop or attach to a
DigitalTwin-owned service. If port 8000 serves a different model, the helper
fails instead of replacing it.

To avoid duplicate model storage when ScreenLens owns the service, point its
caches at an existing compatible cache before startup:

```bash
export DGX_HF_CACHE="$HOME/Desktop/projects/digitalTwin/.local-models/huggingface"
export DGX_VLLM_CACHE="$HOME/Desktop/projects/digitalTwin/.local-models/vllm"
```

These paths are overrides; the repository defaults are
`.local-models/huggingface` and `.local-models/vllm`.

## Prerequisites

The checked configuration expects:

- NVIDIA DGX Spark / GB10 running a current DGX OS on ARM64
- CUDA 13 and a working `nvidia-smi`
- Docker Engine, the Compose plugin, and NVIDIA Container Toolkit
- Python 3.12 with `venv` support
- ffmpeg/ffprobe recommended for complete video metadata (a missing binary uses the quiet OpenCV fallback)
- approximately 70 GiB of free disk for a first container/model download
- outbound HTTPS to Docker Hub, Hugging Face, and Python package indexes
- a Hugging Face read token when this repository must start vLLM

Verify Docker GPU passthrough independently if the NVIDIA runtime was recently
installed:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If Docker does not list an NVIDIA runtime, perform the one-time administrator
configuration and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If the user cannot access `/var/run/docker.sock`, add that user to the Docker
group and start a fresh login session. Do not run ScreenLens itself as root.

## One-time setup

From the repository root:

```bash
(umask 077; touch .env)
chmod 600 .env
${EDITOR:-nano} .env
```

Add `HF_TOKEN=hf_...` to `.env` if ScreenLens may need to start its own server.
The helper reads only its documented variable whitelist as literal values; it
does not source or execute `.env`.

Then run:

```bash
./setup_and_run_dgx.sh doctor
./setup_and_run_dgx.sh setup
./setup_and_run_dgx.sh llm-up
./setup_and_run_dgx.sh llm-wait
./setup_and_run_dgx.sh smoke
```

`setup` creates `.venv-dgx`, installs `torch==2.11.0+cu130` and
`torchvision==0.26.0+cu130` from PyTorch's CUDA 13 index, installs ScreenLens,
and executes real CUDA matrix and OpenCLIP image-encoder operations. It does not
install vLLM into the application environment; the server's CUDA/Triton stack
stays isolated in its container.

The first server start may download roughly 24 GB of model data and build
FlashInfer/Torch caches. The default readiness timeout is 1,800 seconds.

## Validation

Readiness is not based only on container state. `llm-wait` requires
`/v1/models` to contain the configured model id and advertise at least
`VLLM_MAX_MODEL_LEN`. `smoke` then uses ScreenLens's own inference client to
send `assets/ingest-demo.png` as an image and passes only when the model reads
the visible filename `test.mov`.

```bash
./setup_and_run_dgx.sh llm-wait
./setup_and_run_dgx.sh smoke
```

This catches text-only deployments and OpenAI-compatible servers that are alive
but not actually processing vision content.

## Running ScreenLens

With no additional arguments, `run` launches the TUI:

```bash
./setup_and_run_dgx.sh run
```

CLI arguments pass through to `python -m src.cli`:

```bash
./setup_and_run_dgx.sh run ingest input-videos/demo.mov
./setup_and_run_dgx.sh run transcribe input-videos/demo.mov
./setup_and_run_dgx.sh run models
```

The helper exports the selected `VLLM_*` connection, sets
`SCREENLENS_BACKEND=vllm` and `SCREENLENS_DEVICE=cuda`, and bounds ScreenLens
image-request concurrency to two to match the server's sequence limit. It also
points `HF_HOME` at `DGX_HF_CACHE` so OpenCLIP weights share the persistent
application/model cache; placeholder tokens are removed before launching the
application.

## Commands

| Command | Effect |
|---|---|
| `doctor` | Read-only host, GPU, Docker, Compose, disk, Python, token, and reuse checks |
| `setup` | Build or repair `.venv-dgx` and run CUDA/OpenCLIP preflight |
| `llm-up` | Reuse an exact-model service or start this repository's Compose service |
| `llm-wait` | Wait for exact model discovery through `/v1/models` |
| `llm-logs` | Follow only the ScreenLens-owned container logs |
| `llm-down` | Stop only the ScreenLens-owned stack and preserve caches |
| `smoke` | Run the real `ingest-demo.png` vision assertion |
| `run` | Ensure the endpoint, export DGX defaults, and invoke ScreenLens |
| `help` | Show command and environment help |

## Configuration

Settings may be exported or stored in the private `.env`; exported nonempty
values take precedence.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | required for a new local start | Hugging Face read token; never printed |
| `HF_HUB_DISABLE_XET` | `1` | Use resumable standard Hub HTTP and avoid observed Xet CAS 401 failures |
| `DGX_VENV_DIR` | `.venv-dgx` | Python 3.12 application environment |
| `DGX_PYTHON_BIN` | `python3.12` | Interpreter used to create the environment |
| `DGX_HF_CACHE` | `.local-models/huggingface` | Persistent vLLM and OpenCLIP Hugging Face cache |
| `DGX_VLLM_CACHE` | `.local-models/vllm` | Persistent compilation/runtime cache |
| `VLLM_IMAGE` | validated `vllm/vllm-openai@sha256:…` digest | ARM64 vLLM image; override deliberately when validating an upgrade |
| `VLLM_MODEL` | `nvidia/Qwen3.6-27B-NVFP4` | Served and requested dense multimodal model id |
| `VLLM_MODEL_REVISION` | pinned SHA above | Reproducible model contents |
| `VLLM_QUANTIZATION` | `modelopt` | vLLM quantization backend; use `fp8` with `Qwen/Qwen3.6-27B-FP8` |
| `VLLM_KV_CACHE_DTYPE` | `fp8` | KV-cache precision; use `auto` when a checkpoint lacks calibrated FP8 KV scales |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible API root |
| `VLLM_API_KEY` | `local` | Placeholder for loopback, or bearer token for an externally authenticated endpoint |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.45` | vLLM allocator target sized for the long FP8 KV cache |
| `VLLM_MAX_MODEL_LEN` | `262144` | Native serving limit and Python prompt-planning context |
| `VLLM_START_TIMEOUT` | `1800` | Readiness timeout in seconds |
| `VLLM_LOG_TAIL` | `200` | Initial line count for `llm-logs` |

The included loopback service does not enable API authentication. `local` only
satisfies clients that require a nonempty key. Do not expose port 8000 to the
LAN or Internet; use an authenticated TLS reverse proxy for remote access.

## Memory and performance notes

DGX Spark memory is unified, not 128 GB of VRAM plus separate system RAM. In the
The checked recipe runs one dense NVFP4 model rather than keeping a separate
draft or second large model resident. Its `0.45` allocator target is deliberate:
the 262K window needs a substantially larger FP8 KV cache than the former 32K
service. Keep concurrency at two, and lower the allocator or context together
if other sustained GPU workloads must share the host.

Integrated-GPU memory fields may appear unavailable in `nvidia-smi`. Use
`free -h`, the DGX Dashboard, and per-process measurements for the shared pool.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Docker permission denied | Add the user to the Docker group, log out/in, and rerun `doctor` |
| NVIDIA runtime missing | Configure it with `nvidia-ctk`, restart Docker, and validate GPU passthrough |
| `HF_TOKEN` missing | Add a read token to mode-600 `.env`; it is unnecessary only when reusing a ready service |
| Public OpenCLIP weights load without `HF_TOKEN` | Supported; ScreenLens hides only the unauthenticated-download advisory while surfacing real Hub failures |
| Reconstruction exhausts the served context | The incomplete prefix is discarded automatically and the input group is split; a final failure means one minimum-size group still cannot fit even with the full served context |
| Hugging Face Xet/CAS returns 401 | Keep `HF_HUB_DISABLE_XET=1`; standard Hub HTTP resumes partial downloads |
| Port 8000 has another model | Stop/reconfigure its owner or set `VLLM_BASE_URL`; ScreenLens will not replace it |
| First startup appears stalled | Follow the owning stack's logs and allow the 1,800-second model/compile timeout |
| PyTorch reports no CUDA | Rerun `setup`; do not replace the `+cu130` ARM64 wheels with PyPI CPU wheels |
| `pip check` reports the known cuSPARSELt SBSA tag | The helper accepts only that exact warning after successful real CUDA/OpenCLIP operations |
| Vision smoke omits `test.mov` | Confirm the exact multimodal model and inspect vLLM logs; a text-only or mismatched service is not valid |
| Out of memory or heavy swap | Stop competing workloads first; otherwise reduce `VLLM_GPU_MEMORY_UTILIZATION` and `VLLM_MAX_MODEL_LEN` together while retaining concurrency two |

## Apple Silicon remains supported

The existing oMLX/MPS settings, CLI flags, and environment aliases remain the
native macOS path. None of the DGX commands run automatically, and `.venv-dgx`
is separate from Apple development environments. Use the ordinary README and
oMLX settings on Apple Silicon; use this helper only on Linux/ARM64 DGX Spark.
