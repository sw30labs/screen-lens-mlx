"""Unit-level coverage for the DGX Spark integration surface."""

from pathlib import Path
import os
import subprocess
import sys

from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_cli_exposes_vllm_and_provider_neutral_aliases():
    from src.cli import app
    from typer.main import get_command

    result = CliRunner().invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "vllm" in result.output
    assert "--inference-url" in result.output
    ingest = get_command(app).commands["ingest"]
    model_option = next(param for param in ingest.params if param.name == "omlx_model")
    assert set(model_option.opts) == {
        "--inference-model",
        "--vllm-model",
        "--omlx-model",
    }


def test_direct_only_commands_never_default_to_ollama():
    from src.cli import DEFAULT_INFERENCE_BACKEND, app
    from typer.main import get_command

    commands = get_command(app).commands
    assert DEFAULT_INFERENCE_BACKEND in {"vllm", "omlx"}
    for name in ("summarize", "reconstruct", "assemble", "transcribe", "models"):
        backend = next(param for param in commands[name].params if param.name == "backend")
        assert backend.default == DEFAULT_INFERENCE_BACKEND

    env = os.environ.copy()
    env["SCREENLENS_BACKEND"] = "ollama"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.cli import DEFAULT_BACKEND, DEFAULT_INFERENCE_BACKEND; "
                "print(DEFAULT_BACKEND, DEFAULT_INFERENCE_BACKEND)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    caption_default, direct_default = probe.stdout.strip().split()
    assert caption_default == "ollama"
    assert direct_default in {"vllm", "omlx"}


def test_caption_options_route_to_provider_specific_config():
    from src.cli import _apply_captioning_options
    from src.config import CaptionBackend, ScreenLensConfig

    vllm = ScreenLensConfig()
    _apply_captioning_options(
        vllm,
        backend="vllm",
        omlx_url="http://spark.local:9000/v1",
        omlx_model="org/spark-vlm",
        omlx_api_key="spark-key",
    )
    assert vllm.captioning.backend == CaptionBackend.vllm
    assert vllm.captioning.vllm_base_url == "http://spark.local:9000/v1"
    assert vllm.captioning.vllm_model == "org/spark-vlm"
    assert vllm.captioning.vllm_api_key == "spark-key"

    omlx = ScreenLensConfig()
    _apply_captioning_options(
        omlx,
        backend="omlx",
        omlx_url="http://mac.local:8000/v1",
        omlx_model="mlx-community/vision-model",
        omlx_api_key="mlx-key",
    )
    assert omlx.captioning.backend == CaptionBackend.omlx
    assert omlx.captioning.omlx_model == "mlx-community/vision-model"
    assert omlx.captioning.omlx_api_key == "mlx-key"


def test_platform_launchers_have_valid_shell_syntax():
    for launcher in ("setup_and_run_dgx.sh", "setup_and_run_macos.sh"):
        path = ROOT / launcher
        assert path.exists()
        assert path.stat().st_mode & 0o111
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_compose_recipe_is_loopback_only_and_bounded():
    compose = (ROOT / "compose.dgx-spark.yaml").read_text(encoding="utf-8")

    assert '"127.0.0.1:8000:8000"' in compose
    assert "platform: linux/arm64" in compose
    assert "vllm/vllm-openai@sha256:" in compose
    assert "nvidia/Qwen3.6-27B-NVFP4" in compose
    assert '"${VLLM_QUANTIZATION:-modelopt}"' in compose
    assert '"${VLLM_KV_CACHE_DTYPE:-fp8}"' in compose
    assert '"${VLLM_MAX_MODEL_LEN:-262144}"' in compose
    assert '"method":"mtp"' in compose
    assert "--moe-backend" not in compose
    assert "--max-num-seqs" in compose
    assert '      - "2"' in compose


def test_dgx_helper_accepts_a_model_specific_quantization_override():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert "[VLLM_QUANTIZATION]=1" in helper
    assert "[VLLM_KV_CACHE_DTYPE]=1" in helper
    assert 'VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt}"' in helper
    assert 'VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"' in helper
    assert "export VLLM_MODEL VLLM_MODEL_REVISION VLLM_QUANTIZATION" in helper


def test_dgx_smoke_uses_a_real_screen_image():
    helper = (ROOT / "setup_and_run_dgx.sh").read_text(encoding="utf-8")

    assert "assets/ingest-demo.png" in helper
    assert "InferenceClient.from_endpoint" in helper
    assert "test.mov" in helper
