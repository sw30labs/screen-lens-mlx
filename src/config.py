"""
Configuration for the ScreenLens pipeline.
All settings are centralized here for easy tuning.
"""
from enum import Enum
import os
import platform
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


_DOTENV_LOADED = False


def load_dotenv_if_present() -> None:
    """Load literal ``KEY=value`` entries without overriding shell exports."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / ".env", repo_root / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        env_path = env_path.resolve()
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            os.environ.setdefault(key, value)


# ── Frame Extraction ────────────────────────────────────────────────────────

class ExtractionStrategy(str, Enum):
    """How to decide which frames to extract."""
    fixed_fps = "fixed_fps"       # Simple: 1 frame every N seconds
    keyframe = "keyframe"         # Smart: hybrid change detection (SSIM + pHash + HSV)


class FrameExtractionConfig(BaseModel):
    """Settings for video frame extraction."""
    strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.keyframe,
        description="Extraction strategy: 'keyframe' (smart, recommended) or 'fixed_fps'"
    )
    # Fixed FPS settings
    fps: float = Field(default=1.0, description="Frames per second (only for fixed_fps strategy)")
    # Keyframe detection settings (hybrid change detector)
    ssim_threshold: float = Field(default=0.97, description="SSIM below this = scene change")
    phash_threshold: int = Field(default=8, description="Perceptual hash hamming distance threshold")
    hist_corr_threshold: float = Field(default=0.90, description="HSV histogram correlation threshold")
    min_interval_seconds: float = Field(default=0.5, description="Min seconds between keyframes")
    max_interval_seconds: float = Field(default=4.0, description="Force a keyframe at least this often")
    min_changed_area: float = Field(default=0.02, description="Min fraction of pixels that must change")
    # Shared settings
    max_dimension: int = Field(default=1280, description="Max width or height for extracted frames")
    output_format: str = Field(default="jpg", description="Frame image format (jpg, png)")
    quality: int = Field(default=85, description="JPEG quality (1-100)")


# ── Captioning ──────────────────────────────────────────────────────────────

def is_dgx_spark_host() -> bool:
    """Return whether platform defaults should target NVIDIA DGX Spark."""
    return platform.system() == "Linux" and platform.machine().lower() in {
        "aarch64",
        "arm64",
    }


class InferenceBackend(str, Enum):
    """Direct OpenAI-compatible inference servers supported by ScreenLens."""
    vllm = "vllm"
    omlx = "omlx"


class CaptionBackend(str, Enum):
    """Which vision model backend to use for captioning."""
    vllm = "vllm"           # vLLM OpenAI-compatible server (DGX Spark)
    omlx = "omlx"           # oMLX OpenAI-compatible server (Apple Silicon native)
    ollama = "ollama"       # Any Ollama vision model (llama3.2-vision, etc.)


def default_inference_backend() -> InferenceBackend:
    """Choose the native direct server, with an explicit env override."""
    load_dotenv_if_present()
    configured = os.getenv("SCREENLENS_BACKEND", "").strip().lower()
    if configured and configured != CaptionBackend.ollama.value:
        return InferenceBackend(configured)
    return InferenceBackend.vllm if is_dgx_spark_host() else InferenceBackend.omlx


def default_caption_backend() -> CaptionBackend:
    """Choose vLLM on DGX Spark and oMLX elsewhere unless overridden."""
    load_dotenv_if_present()
    configured = os.getenv("SCREENLENS_BACKEND", "").strip().lower()
    if configured:
        return CaptionBackend(configured)
    return CaptionBackend.vllm if is_dgx_spark_host() else CaptionBackend.omlx


def default_inference_concurrency() -> int:
    """Match the conservative two-sequence DGX Spark serving recipe."""
    load_dotenv_if_present()
    configured = os.getenv("SCREENLENS_BATCH_SIZE", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return 2 if is_dgx_spark_host() else 4


def default_embedding_device() -> str:
    """Choose the native accelerator without importing torch at config time."""
    load_dotenv_if_present()
    configured = os.getenv("SCREENLENS_DEVICE", "").strip().lower()
    if configured:
        return configured
    if is_dgx_spark_host():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mps"
    return "cpu"


class CaptioningConfig(BaseModel):
    """Settings for frame captioning."""
    backend: CaptionBackend = Field(
        default_factory=default_caption_backend,
        description=(
            "Vision backend: 'vllm' (DGX Spark), 'omlx' (Apple Silicon), or 'ollama'"
        ),
    )
    # vLLM settings (OpenAI-compatible DGX Spark server)
    vllm_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="vLLM OpenAI-compatible API base URL",
    )
    vllm_model: Optional[str] = Field(
        default=None,
        description="vLLM model id; defaults to VLLM_MODEL then the DGX Spark model",
    )
    vllm_api_key: Optional[str] = Field(
        default=None,
        description="vLLM API key; defaults to VLLM_API_KEY or a local placeholder",
    )
    vllm_timeout_seconds: float = Field(
        default=600.0,
        description="HTTP timeout for vLLM generation requests",
    )
    vllm_model_context: int = Field(
        default=32768,
        description="Configured vLLM context window used for chunk planning",
    )
    # oMLX settings (OpenAI-compatible local server)
    omlx_base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="oMLX OpenAI-compatible API base URL"
    )
    omlx_model: Optional[str] = Field(
        default=None,
        description="Model ID served by oMLX. Defaults to MLX_MODEL, OMLX_MODEL, LLM_MODEL, then 'default'."
    )
    omlx_api_key: Optional[str] = Field(
        default=None,
        description="oMLX API key. If unset, MLX_API_KEY or OMLX_API_KEY is read from the environment."
    )
    omlx_timeout_seconds: float = Field(
        default=600.0,
        description="HTTP timeout for oMLX generation requests"
    )
    omlx_model_context: int = Field(
        default=32768,
        description="Assumed oMLX model context window for chunk planning"
    )
    # Ollama settings (fallback)
    ollama_model: str = Field(default="llama3.2-vision", description="Ollama vision model name")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", description="Ollama API endpoint")
    # Shared generation settings
    temperature: float = Field(default=0.1, description="LLM temperature for captions")
    max_tokens: int = Field(
        default=32768,
        description=(
            "Requested output-token ceiling per caption. When this reaches the "
            "served vLLM context size, ScreenLens lets vLLM use all context "
            "remaining after the prompt and image."
        ),
    )
    retry_attempts: int = Field(
        default=1,
        ge=0,
        description="Per-frame retries after a direct caption request fails",
    )
    retry_max_tokens: int = Field(
        default=2048,
        ge=1,
        description=(
            "Bounded output-token ceiling for a retried caption request, used to "
            "prevent a malformed generation from consuming the full normal budget"
        ),
    )
    repetition_penalty: float = Field(
        default=1.05,
        description="Lightly discourage degenerate long-form caption repetition",
    )
    no_repeat_ngram_size: int = Field(
        default=12,
        description="Block long repeated caption loops (0 disables)",
    )
    batch_size: int = Field(
        default_factory=default_inference_concurrency,
        description=(
            "Frames per chunk / concurrent OpenAI-compatible requests. Defaults "
            "to 2 on DGX Spark to match the bundled vLLM service."
        ),
    )
    disable_thinking: bool = Field(
        default=True,
        description=(
            "Disable model reasoning for OpenAI-compatible captions so Qwen models spend "
            "their token budget on the visible answer instead of hidden thinking."
        ),
    )
    system_prompt: str = Field(
        default=(
            "You are a meticulous video frame analyst. You respond ONLY with your analysis — "
            "no preamble, no thinking, no meta-commentary. Output raw Markdown directly."
        ),
        description="System prompt for the vision model"
    )
    user_prompt: str = Field(
        default=(
            "Analyze this video frame and describe everything visible. "
            "Respond directly with the analysis — no preamble or reasoning.\n\n"
            "1. **Text**: Reproduce all visible text exactly, preserving hierarchy.\n"
            "2. **UI Elements**: Describe buttons, menus, toolbars, dialogs, and their states.\n"
            "3. **Tables**: Render any tables as Markdown tables.\n"
            "4. **Diagrams**: Describe any diagrams, charts, or visual flows.\n"
            "5. **Actions**: Note what action or interaction appears to be happening.\n\n"
            "IMPORTANT: Ignore browser chrome and OS window decorations. "
            "Focus only on application content."
        ),
        description="User prompt sent with each frame"
    )


# ── Verbatim OCR (NEW: faithful transcription path) ──────────────────────────
#
# This is distinct from `CaptioningConfig`. Captioning *describes* a frame;
# OCR *transcribes* it character-for-character. Verbatim reconstruction needs
# the latter. Critically, the OCR model MUST be vision-capable — the original
# failure mode was pointing MLX_MODEL at a text-only model (MiniMax-M2), which
# silently answered every frame "no image provided".

# A transcription prompt, NOT a description prompt. Tuned to avoid the two
# documented VLM-OCR failure modes: (a) summarizing/"improving" instead of
# copying, and (b) repetition loops on dense symbol runs.
OCR_SYSTEM_PROMPT = (
    "You are a high-fidelity OCR engine. You transcribe the text visible in an "
    "image EXACTLY as it appears — you never summarize, paraphrase, translate, "
    "complete, correct, or explain. You output only the transcribed text."
)

OCR_USER_PROMPT = (
    "Transcribe ALL text visible in this image, verbatim, preserving reading "
    "order top-to-bottom.\n\n"
    "RULES:\n"
    "- Copy every character exactly as shown: spelling, casing, punctuation, "
    "numbers, symbols, and indentation. Do NOT fix typos or 'improve' anything.\n"
    "- Preserve line breaks. Keep code indentation and alignment using spaces.\n"
    "- For code, reproduce it inside a fenced ``` block with the visible language.\n"
    "- For tables, use Markdown table rows with the exact cell text.\n"
    "- Transcribe partially-visible lines at the top/bottom edge only if the "
    "text is fully legible; otherwise omit them.\n"
    "- Ignore OS/window chrome, scrollbars, the mouse cursor, and browser UI. "
    "Transcribe only document/editor/application content.\n"
    "- If the frame has no legible text, output exactly: [NO TEXT]\n"
    "- Do not add commentary, headers, or markdown fences around the whole "
    "response — output the transcription directly."
)


class OCRConfig(BaseModel):
    """Settings for the verbatim OCR pass (vision model required)."""
    backend: InferenceBackend = Field(
        default_factory=default_inference_backend,
        description="OpenAI-compatible vision server: vllm or omlx",
    )
    # OpenAI-compatible vision server
    base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="OpenAI-compatible API base URL (dashboard/root URLs are normalized)",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Vision model id for OCR. MUST be vision-capable (VL/vision/omni). "
            "Uses VLLM_OCR_MODEL/VLLM_MODEL on vLLM, or OCR_MODEL/MLX_VISION_MODEL "
            "on oMLX, before the provider default."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Inference API key (env: OCR_API_KEY/VLLM_API_KEY/MLX_API_KEY)",
    )
    timeout_seconds: float = Field(default=600.0, description="HTTP timeout per frame")
    # Generation — tuned for verbatim fidelity, not creativity
    temperature: float = Field(default=0.0, description="0 = deterministic; do not raise for OCR")
    max_tokens: int = Field(default=4096, description="Max tokens per frame transcription")
    repetition_penalty: float = Field(
        default=1.15,
        description="Guards against the documented VLM-OCR repetition-loop failure on symbol runs",
    )
    no_repeat_ngram_size: int = Field(
        default=6, description="Block verbatim n-gram loops (0 disables)"
    )
    concurrency: int = Field(
        default_factory=default_inference_concurrency,
        description="Concurrent OCR requests (2 by default on DGX Spark)",
    )
    system_prompt: str = Field(default=OCR_SYSTEM_PROMPT)
    user_prompt: str = Field(default=OCR_USER_PROMPT)
    # Deterministic cross-check (Apple Vision via ocrmac) — optional
    deterministic_backstop: bool = Field(
        default=False,
        description=(
            "If true, also run Apple Vision OCR (ocrmac, language-correction OFF) "
            "and flag character-level disagreements. Recommended for CODE recordings "
            "where VLMs hallucinate tokens."
        ),
    )
    require_vision_model: bool = Field(
        default=True,
        description="Abort if the served model is not vision-capable (prevents the blind-model bug)",
    )
    disable_thinking: bool = Field(
        default=True,
        description=(
            "Disable model 'thinking'/reasoning for OCR. Verbatim copy needs no "
            "chain-of-thought; a reasoning model (e.g. Qwen3.x) otherwise burns the "
            "whole token budget on reasoning and never emits the transcription. "
            "Sends chat_template_kwargs.enable_thinking=false (honored by Qwen3 / "
            "vLLM / SGLang / MLX-LM servers)."
        ),
    )


# ── Frame selection for transcription (NEW) ──────────────────────────────────
#
# Philosophy reversal vs. the old keyframe detector: for VERBATIM work we do NOT
# try to be clever with pixel metrics on scrolling text (they fail — proven
# empirically). We extract densely, drop only NEAR-EXACT duplicates (static
# pauses), and let the text-space stitcher do the real dedup.

class FrameSelectionConfig(BaseModel):
    """Settings for selecting frames to OCR (scroll-safe)."""
    sample_fps: float = Field(
        default=2.0,
        description="Sample this many frames/sec before dedup (code/docs scroll fast; 2 is safe)",
    )
    drop_duplicate_ssim: float = Field(
        default=0.992,
        description="Drop a frame if SSIM vs the last kept frame exceeds this (near-exact static dupe)",
    )
    max_dimension: int = Field(default=1400, description="Max width/height of saved frames (keep text crisp)")
    output_format: str = Field(default="png", description="png keeps text sharp; jpg is smaller")
    quality: int = Field(default=92, description="JPEG quality if output_format=jpg")


# ── Reconstruction / LLM cleanup (NEW) ───────────────────────────────────────
#
# The reconstruction/cleanup LLM is a TEXT model and may be the model that was
# wrongly used for vision before (MiniMax-M2 is a fine *text* reasoner). Its job
# is strictly limited: fix stitch seams and re-indentation — never re-invent text.

class ReconstructionConfig(BaseModel):
    """Settings for the optional LLM cleanup/reconstruction pass."""
    backend: InferenceBackend = Field(
        default_factory=default_inference_backend,
        description="OpenAI-compatible text server: vllm or omlx",
    )
    enabled: bool = Field(
        default=False,
        description=(
            "Run an LLM cleanup pass over the stitched transcript. Off by default: "
            "the raw stitched OCR is already verbatim, and an LLM tends to drop "
            "content while 'repairing' (the per-chunk coverage guard then discards "
            "most of its output anyway). Enable for prose where seam/indent tidying "
            "is worth the cost."
        ),
    )
    base_url: str = Field(default="http://127.0.0.1:8000/v1")
    model: Optional[str] = Field(
        default=None,
        description=(
            "Text model id. Uses VLLM_LLM_MODEL/VLLM_MODEL on vLLM, or "
            "LLM_MODEL/MLX_MODEL on oMLX."
        ),
    )
    api_key: Optional[str] = Field(default=None)
    timeout_seconds: float = Field(
        default=1800.0,
        description=(
            "HTTP timeout for long-running text reconstruction requests. "
            "Kept separate from the shorter per-frame caption timeout."
        ),
    )
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=8192)
    model_context: int = Field(default=32768, description="Assumed context window for chunk planning")
    disable_thinking: bool = Field(
        default=True,
        description=(
            "Disable model 'thinking'/reasoning for the seam/indent cleanup pass. "
            "The repair is near-verbatim; a reasoning text model (e.g. MiniMax-M2) "
            "otherwise wastes the token budget reasoning before emitting the "
            "repaired transcript."
        ),
    )


# ── Embedding ───────────────────────────────────────────────────────────────

class EmbeddingConfig(BaseModel):
    """Settings for CLIP embedding generation."""
    model_name: str = Field(default="ViT-B-32", description="OpenCLIP model architecture")
    pretrained: str = Field(default="laion2b_s34b_b79k", description="Pretrained weights")
    batch_size: int = Field(default=64, description="Batch size for embedding generation")
    device: str = Field(
        default_factory=default_embedding_device,
        description="Device: cuda (DGX Spark), mps (Apple Silicon), or cpu",
    )


# ── Vector DB ───────────────────────────────────────────────────────────────

class VectorDBConfig(BaseModel):
    """Settings for ChromaDB vector storage."""
    collection_name: str = Field(default="screenlens_frames", description="ChromaDB collection name")
    persist_directory: str = Field(default="./data/chromadb", description="ChromaDB storage path")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")


# ── Search & Summarization ──────────────────────────────────────────────────

class SearchConfig(BaseModel):
    """Settings for search and summarization."""
    top_k: int = Field(default=10, description="Number of results to return")
    summarization_model: str = Field(
        default="llama3.2",
        description="Fallback model used only when the Ollama backend is selected",
    )
    base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Fallback API URL used only when the Ollama backend is selected",
    )


# ── Top-Level Config ────────────────────────────────────────────────────────

class ScreenLensConfig(BaseModel):
    """Top-level configuration for the entire ScreenLens pipeline."""
    frame_extraction: FrameExtractionConfig = Field(default_factory=FrameExtractionConfig)
    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)
    # NEW verbatim-transcription path:
    frame_selection: FrameSelectionConfig = Field(default_factory=FrameSelectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")

    def ensure_dirs(self):
        """Create all necessary data directories."""
        (self.data_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "captions").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chromadb").mkdir(parents=True, exist_ok=True)
