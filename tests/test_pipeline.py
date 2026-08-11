"""
Integration tests for the ScreenLens pipeline.

Run with: pytest tests/test_pipeline.py -v
"""
import json
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest
import yaml

# Load test cases
TEST_CASES_PATH = Path(__file__).parent / "test_cases.yaml"


def load_test_cases():
    """Load test case definitions from YAML."""
    if TEST_CASES_PATH.exists():
        with open(TEST_CASES_PATH) as f:
            return yaml.safe_load(f)
    return {"test_cases": []}


class TestConfig:
    """Test the configuration system."""

    def test_dgx_spark_defaults(self, monkeypatch):
        import src.config as config_module
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", True)
        monkeypatch.setattr(config_module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(config_module.platform, "machine", lambda: "aarch64")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.vllm
        assert config.captioning.vllm_base_url == "http://127.0.0.1:8000/v1"
        assert config.captioning.disable_thinking is True
        assert config.captioning.max_tokens == 32768
        assert config.captioning.retry_attempts == 1
        assert config.captioning.retry_max_tokens == 2048
        assert config.captioning.batch_size == 2
        assert config.reconstruction.timeout_seconds == 1800
        assert config.ocr.backend == InferenceBackend.vllm
        assert config.frame_extraction.fps == 1.0
        assert config.embedding.device == "cuda"
        assert config.vector_db.collection_name == "screenlens_frames"

    def test_apple_silicon_defaults(self, monkeypatch):
        import src.config as config_module
        from src.config import CaptionBackend, ScreenLensConfig

        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", True)
        monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(config_module.platform, "machine", lambda: "arm64")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.omlx
        assert config.captioning.max_tokens == 32768
        assert config.captioning.retry_attempts == 1
        assert config.captioning.retry_max_tokens == 2048
        assert config.captioning.batch_size == 4
        assert config.embedding.device == "mps"

    def test_platform_defaults_accept_environment_overrides(self, monkeypatch):
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        monkeypatch.setenv("SCREENLENS_BACKEND", "ollama")
        monkeypatch.setenv("SCREENLENS_DEVICE", "cpu")
        monkeypatch.setenv("SCREENLENS_BATCH_SIZE", "7")

        config = ScreenLensConfig()
        assert config.captioning.backend == CaptionBackend.ollama
        assert config.ocr.backend in (InferenceBackend.vllm, InferenceBackend.omlx)
        assert config.captioning.batch_size == 7
        assert config.embedding.device == "cpu"

    def test_dotenv_applies_platform_default_overrides(self, monkeypatch, tmp_path):
        import src.config as config_module
        from src.config import CaptionBackend, ScreenLensConfig

        (tmp_path / ".env").write_text(
            "SCREENLENS_BACKEND=ollama\n"
            "SCREENLENS_DEVICE=cpu\n"
            "SCREENLENS_BATCH_SIZE=3\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SCREENLENS_BACKEND", raising=False)
        monkeypatch.delenv("SCREENLENS_DEVICE", raising=False)
        monkeypatch.delenv("SCREENLENS_BATCH_SIZE", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", False)

        config = ScreenLensConfig()

        assert config.captioning.backend == CaptionBackend.ollama
        assert config.captioning.batch_size == 3
        assert config.embedding.device == "cpu"

    def test_config_override(self):
        from src.config import ScreenLensConfig
        config = ScreenLensConfig()
        config.frame_extraction.fps = 0.5
        assert config.frame_extraction.fps == 0.5

    def test_ensure_dirs(self):
        from src.config import ScreenLensConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScreenLensConfig(data_dir=Path(tmpdir) / "test_data")
            config.ensure_dirs()
            assert (config.data_dir / "frames").exists()
            assert (config.data_dir / "captions").exists()
            assert (config.data_dir / "embeddings").exists()


class TestFrameExtractor:
    """Test frame extraction (requires ffmpeg)."""

    def test_format_timestamp(self):
        from src.frame_extractor import _format_timestamp
        assert _format_timestamp(0) == "00:00:00.000"
        assert _format_timestamp(65.5) == "00:01:05.500"
        assert _format_timestamp(3661.123) == "01:01:01.123"

    def test_missing_optional_ffprobe_uses_quiet_opencv_fallback(
        self, monkeypatch, caplog,
    ):
        import logging
        import src.frame_extractor as frame_extractor

        monkeypatch.setattr(frame_extractor.shutil, "which", lambda command: None)
        monkeypatch.setattr(
            frame_extractor.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("ffprobe should not be executed"),
        )

        with caplog.at_level(logging.INFO, logger="screenlens.frame_extractor"):
            assert frame_extractor.get_video_metadata("video.mov") == {}

        assert "reading video metadata with OpenCV" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]

    def test_resize_frame(self):
        from PIL import Image
        from src.frame_extractor import _resize_frame

        img = Image.new("RGB", (1920, 1080))
        resized = _resize_frame(img, 1280)
        assert max(resized.size) <= 1280

        small = Image.new("RGB", (640, 480))
        same = _resize_frame(small, 1280)
        assert same.size == (640, 480)


class TestOMLXClient:
    """Test the oMLX OpenAI-compatible adapter without network access."""

    def test_normalizes_dashboard_url(self):
        from src.omlx_client import normalize_omlx_base_url

        assert (
            normalize_omlx_base_url("http://127.0.0.1:8000/admin/dashboard?tab=status")
            == "http://127.0.0.1:8000/v1"
        )
        assert normalize_omlx_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
        assert normalize_omlx_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"

    def test_dotenv_loads_omlx_values_without_overriding_shell(self, monkeypatch, tmp_path):
        import src.config as config_module
        from src.config import CaptioningConfig
        import src.omlx_client as omlx_client

        (tmp_path / ".env").write_text(
            "\n".join([
                "MLX_API_KEY=your-omlx-api-key-here",
                "OMLX_API_KEY=dotenv-key",
                "MLX_MODEL=dotenv-model",
            ]),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MLX_API_KEY", raising=False)
        monkeypatch.delenv("OMLX_API_KEY", raising=False)
        monkeypatch.delenv("MLX_MODEL", raising=False)
        monkeypatch.setattr(config_module, "_DOTENV_LOADED", False)

        assert omlx_client.resolve_omlx_api_key(CaptioningConfig()) == "dotenv-key"
        assert omlx_client.resolve_omlx_model(CaptioningConfig()) == "dotenv-model"

    def test_rejects_known_text_only_models_for_image_chat(self):
        from src.config import CaptionBackend, CaptioningConfig
        from src.omlx_client import OMLXClient

        client = OMLXClient(CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_model="deepseek-ai-DeepSeek-V4-Flash-8bit",
        ))

        with pytest.raises(ValueError, match="text-only model"):
            client.chat("system", "describe", images=["missing.jpg"])

    def test_vllm_defaults_and_legacy_env_isolation(self, monkeypatch):
        from src.config import CaptionBackend, CaptioningConfig, OCRConfig, ReconstructionConfig
        from src.omlx_client import (
            DEFAULT_VLLM_MODEL,
            resolve_inference_api_key,
            resolve_inference_base_url,
            resolve_inference_context,
            resolve_inference_model,
            resolve_llm_model,
            resolve_ocr_model,
            resolve_role_api_key,
            resolve_role_context,
        )

        monkeypatch.setenv("MLX_MODEL", "legacy-mlx-model")
        monkeypatch.setenv("OCR_MODEL", "legacy-ocr-model")
        monkeypatch.setenv("LLM_MODEL", "legacy-text-model")
        monkeypatch.setenv("VLLM_BASE_URL", "http://spark.local:9000/v1/")
        monkeypatch.setenv("VLLM_API_KEY", "spark-secret")
        monkeypatch.delenv("VLLM_MODEL", raising=False)

        captioning = CaptioningConfig(backend=CaptionBackend.vllm)
        assert resolve_inference_base_url(captioning) == "http://spark.local:9000/v1"
        assert resolve_inference_api_key(captioning) == "spark-secret"
        assert resolve_inference_model(captioning) == DEFAULT_VLLM_MODEL
        assert resolve_ocr_model(OCRConfig(backend="vllm")) == DEFAULT_VLLM_MODEL
        assert resolve_llm_model(ReconstructionConfig(backend="vllm")) == DEFAULT_VLLM_MODEL

        monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "16384")
        assert resolve_inference_context(captioning) == 16384
        assert resolve_role_context(ReconstructionConfig(backend="vllm")) == 16384
        assert resolve_inference_context(
            CaptioningConfig(backend=CaptionBackend.vllm, vllm_model_context=24576)
        ) == 24576

        monkeypatch.setenv("VLLM_OCR_API_KEY", "spark-ocr-secret")
        monkeypatch.setenv("OCR_API_KEY", "legacy-ocr-secret")
        assert resolve_role_api_key(
            OCRConfig(backend="vllm"), "VLLM_OCR_API_KEY", "OCR_API_KEY"
        ) == "spark-ocr-secret"
        assert resolve_role_api_key(
            OCRConfig(backend="omlx"), "VLLM_OCR_API_KEY", "OCR_API_KEY"
        ) == "legacy-ocr-secret"

    def test_nvidia_qwen_spark_model_is_known_multimodal(self):
        from src.omlx_client import DEFAULT_VLLM_MODEL, is_known_vision_model

        assert is_known_vision_model(DEFAULT_VLLM_MODEL)

    def test_loopback_requests_bypass_proxy_environment(self, monkeypatch):
        from urllib import request
        import src.omlx_client as inference_client

        captured = {}
        sentinel = object()

        class FakeOpener:
            def open(self, req, timeout):
                captured["url"] = req.full_url
                captured["timeout"] = timeout
                return sentinel

        def fake_build_opener(*handlers):
            captured["handlers"] = handlers
            return FakeOpener()

        monkeypatch.setattr(inference_client.request, "build_opener", fake_build_opener)
        monkeypatch.setattr(
            inference_client.request,
            "urlopen",
            lambda *args, **kwargs: pytest.fail("loopback request inherited proxy handling"),
        )

        result = inference_client._urlopen(
            request.Request("http://127.0.0.1:8000/v1/models"),
            timeout=3,
        )

        assert result is sentinel
        assert captured["timeout"] == 3
        assert captured["handlers"][0].proxies == {}

    def test_chat_posts_openai_vision_payload(self, monkeypatch, tmp_path):
        from PIL import Image
        from src.config import CaptioningConfig
        from src.omlx_client import OMLXClient
        import src.omlx_client as omlx_client

        img_path = tmp_path / "frame.jpg"
        Image.new("RGB", (4, 4), color="red").save(img_path)

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "<think>hidden</think>visible caption"}}]
                }).encode("utf-8")

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(omlx_client, "_urlopen", fake_urlopen)

        from src.config import CaptionBackend
        cfg = CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_base_url="http://127.0.0.1:8000/admin/dashboard",
            omlx_model="vision-model",
            omlx_api_key="local-key",
            omlx_timeout_seconds=12,
        )
        result = OMLXClient(cfg).chat("system", "describe", images=[str(img_path)])

        assert result == "visible caption"
        assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer local-key"
        assert captured["timeout"] == 12
        assert captured["payload"]["model"] == "vision-model"
        user_content = captured["payload"]["messages"][1]["content"]
        assert user_content[0] == {"type": "text", "text": "describe"}
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.parametrize(
        ("backend", "context_size", "default_max_tokens", "expected_max_tokens"),
        [
            ("vllm", 32768, 32768, None),
            ("vllm", 65536, 32768, 32768),
            ("vllm", 32768, 4096, 4096),
            ("omlx", 32768, 32768, 32768),
        ],
    )
    def test_chat_uses_remaining_vllm_context_at_full_ceiling(
        self,
        backend,
        context_size,
        default_max_tokens,
        expected_max_tokens,
        monkeypatch,
    ):
        from src.omlx_client import InferenceClient
        import src.omlx_client as inference_client

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": "complete"},
                        "finish_reason": "stop",
                    }],
                }).encode("utf-8")

        def fake_urlopen(req, timeout):
            captured.update(json.loads(req.data.decode("utf-8")))
            return FakeResponse()

        monkeypatch.setattr(inference_client, "_urlopen", fake_urlopen)
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend=backend,
            context_size=context_size,
            default_max_tokens=default_max_tokens,
        )

        assert client.chat("system", "user") == "complete"
        if expected_max_tokens is None:
            assert "max_tokens" not in captured
        else:
            assert captured["max_tokens"] == expected_max_tokens

    def test_vllm_context_overflow_retries_with_exact_remaining_tokens(
        self,
        monkeypatch,
    ):
        from src.omlx_client import InferenceClient
        import src.omlx_client as inference_client

        chat_payloads = []
        tokenize_payloads = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(req, timeout):
            payload = json.loads(req.data.decode("utf-8"))
            if req.full_url.endswith("/tokenize"):
                tokenize_payloads.append(payload)
                return FakeResponse({"count": 30721, "max_model_len": 32768, "tokens": []})

            chat_payloads.append(payload)
            if len(chat_payloads) == 1:
                detail = json.dumps({
                    "error": {
                        "message": (
                            "This model's maximum context length is 32768 tokens. "
                            "However, you requested 2048 output tokens and your prompt "
                            "contains at least 30721 input tokens."
                        ),
                        "type": "BadRequestError",
                        "param": "input_tokens",
                        "code": 400,
                    },
                }).encode("utf-8")
                raise HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(detail))
            return FakeResponse({
                "choices": [{
                    "message": {"content": "recovered"},
                    "finish_reason": "stop",
                }],
            })

        monkeypatch.setattr(inference_client, "_urlopen", fake_urlopen)
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            context_size=32768,
            default_max_tokens=4096,
        )

        result = client.chat(
            "system",
            "large prompt",
            max_tokens=2048,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )

        assert result == "recovered"
        assert [payload["max_tokens"] for payload in chat_payloads] == [2048, 2047]
        assert tokenize_payloads == [{
            "model": "vision-model",
            "messages": chat_payloads[0]["messages"],
            "chat_template_kwargs": {"enable_thinking": False},
        }]

    def test_vllm_context_retry_rejects_prompt_larger_than_context(self, monkeypatch):
        from src.omlx_client import InferenceClient

        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            context_size=32768,
        )
        monkeypatch.setattr(client, "_tokenize_chat", lambda payload: (40012, 32768))
        detail = json.dumps({
            "error": {
                "message": "This model's maximum context length is 32768 tokens.",
                "param": "input_tokens",
            },
        })

        with pytest.raises(RuntimeError, match="prompt uses 40,012 tokens"):
            client._context_retry_payload({"max_tokens": 2048}, 400, detail)

    def test_required_complete_generation_rejects_length_finish(self, monkeypatch):
        from src.omlx_client import InferenceClient, InferenceTruncatedError
        import src.omlx_client as inference_client

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": "incomplete prefix"},
                        "finish_reason": "length",
                    }],
                }).encode("utf-8")

        monkeypatch.setattr(inference_client, "_urlopen", lambda req, timeout: FakeResponse())
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            default_max_tokens=4096,
        )

        with pytest.raises(InferenceTruncatedError, match="incomplete output was discarded"):
            client.chat(
                "system",
                "user",
                max_tokens=2048,
                require_complete=True,
            )

    def test_chat_timeout_reports_effective_request_budget(self, monkeypatch):
        from src.omlx_client import InferenceClient
        import src.omlx_client as inference_client

        def raise_timeout(req, timeout):
            assert timeout == 1800
            raise TimeoutError("timed out")

        monkeypatch.setattr(inference_client, "_urlopen", raise_timeout)
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="vision-model",
            api_key="local",
            backend="vllm",
            timeout=1800,
        )

        with pytest.raises(RuntimeError, match="timed out after 1800 seconds"):
            client.chat("system", "user")


    @pytest.mark.parametrize("text,flagged", [
        ("<\uff5cbegin\u2581of\u2581sentence\uff5c>" * 30, True),
        ("INSERT INTO scenarios VALUES ('E" + "0" * 400, True),
        ("The screen shows a spreadsheet. " * 3 + "0" * 500, True),
        ("Too short to judge.", False),
        ("def f(x):\n    return x + 1\n" * 8, False),
        ("INSERT INTO t VALUES (1, 'abc', 'def');\n" * 20, False),
        ("\n".join(f"- item {i} with a distinct description" for i in range(40)), False),
    ])
    def test_degenerate_repetition_detects_stuck_decoders_only(self, text, flagged):
        """A stuck decoder repeats one short token; reconstructed code repeats
        whole statements and must not be mistaken for it."""
        from src.omlx_client import degenerate_repetition

        assert bool(degenerate_repetition(text)) is flagged

    def test_degenerate_output_is_rejected_not_saved(self, monkeypatch):
        """A stuck decoder must not have its output stored as a result."""
        from src.omlx_client import InferenceClient, InferenceDegenerateError
        import src.omlx_client as inference_client

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"content": "<BOS>" * 60},
                        "finish_reason": "stop",
                    }],
                }).encode("utf-8")

        monkeypatch.setattr(inference_client, "_urlopen", lambda req, timeout: FakeResponse())
        client = InferenceClient.from_endpoint(
            base_url="http://127.0.0.1:8000/v1",
            model="text-model",
            api_key="local",
            backend="omlx",
            default_max_tokens=4096,
        )

        with pytest.raises(InferenceDegenerateError, match="degenerate repeated output"):
            client.chat("system", "user", require_complete=True)

        # Without require_complete the caller still gets the text, but warned.
        assert client.chat("system", "user").startswith("<BOS>")

    def test_system_turn_is_folded_for_models_that_mishandle_it(self):
        """DeepSeek-V4-Flash under oMLX emits its BOS token to the token limit
        when given a system turn; the same instruction in the user turn works."""
        from src.config import CaptionBackend, CaptioningConfig
        from src.omlx_client import InferenceClient, mishandles_system_role

        assert mishandles_system_role("DeepSeek-V4-Flash-0731-MLX")
        assert not mishandles_system_role("Qwen3.6-27B-bf16")

        captured = {}

        def fake_post(self, payload, **kwargs):
            captured["messages"] = payload["messages"]
            return "ok"

        config = CaptioningConfig(
            backend=CaptionBackend.omlx, omlx_model="DeepSeek-V4-Flash-0731-MLX"
        )
        client = InferenceClient(config)
        client._post_chat = fake_post.__get__(client, type(client))
        client.chat("SYSTEM RULES", "USER QUESTION")

        assert [m["role"] for m in captured["messages"]] == ["user"]
        assert captured["messages"][0]["content"] == "SYSTEM RULES\n\nUSER QUESTION"

    def test_system_turn_is_kept_for_models_that_handle_it(self):
        from src.config import CaptionBackend, CaptioningConfig
        from src.omlx_client import InferenceClient

        captured = {}

        def fake_post(self, payload, **kwargs):
            captured["messages"] = payload["messages"]
            return "ok"

        client = InferenceClient(
            CaptioningConfig(backend=CaptionBackend.omlx, omlx_model="Qwen3.6-27B-bf16")
        )
        client._post_chat = fake_post.__get__(client, type(client))
        client.chat("SYSTEM RULES", "USER QUESTION")

        assert [m["role"] for m in captured["messages"]] == ["system", "user"]

    def test_folded_system_turn_survives_image_content_blocks(self):
        """Captioning sends content blocks, not a bare string."""
        from src.omlx_client import _prepend_instruction

        blocks = [{"type": "text", "text": "describe this"},
                  {"type": "image_url", "image_url": {"url": "data:..."}}]
        out = _prepend_instruction("RULES", blocks)
        assert out[0]["text"] == "RULES\n\ndescribe this"
        assert out[1]["type"] == "image_url"


class TestCaptioner:
    """Test caption generation controls without contacting oMLX."""

    @pytest.mark.parametrize("disable_thinking", [True, False])
    def test_omlx_captioner_controls_model_thinking(
        self,
        disable_thinking,
        monkeypatch,
        tmp_path,
    ):
        from PIL import Image
        from src.captioner import OMLXCaptioner
        from src.config import CaptionBackend, CaptioningConfig

        img_path = tmp_path / "frame.jpg"
        Image.new("RGB", (4, 4), color="blue").save(img_path)

        config = CaptioningConfig(
            backend=CaptionBackend.omlx,
            omlx_model="vision-model",
            disable_thinking=disable_thinking,
        )
        captioner = OMLXCaptioner(config)
        captured = {}

        def fake_post(payload):
            captured.update(payload)
            return "visible caption"

        monkeypatch.setattr(captioner._client, "_post_chat", fake_post)

        assert captioner.caption(str(img_path)) == "visible caption"
        assert captured["repetition_penalty"] == 1.05
        assert captured["no_repeat_ngram_size"] == 12
        if disable_thinking:
            assert captured["chat_template_kwargs"] == {"enable_thinking": False}
        else:
            assert "chat_template_kwargs" not in captured


class TestEmbedder:
    """Test CLIP embedding generation."""

    def test_public_hub_filter_only_suppresses_auth_advisory(self):
        import logging
        from src.embedder import _PublicHubAuthWarningFilter

        auth_record = logging.LogRecord(
            "huggingface_hub.utils._http",
            logging.WARNING,
            __file__,
            1,
            "Warning: You are sending unauthenticated requests to the HF Hub.",
            (),
            None,
        )
        failure_record = logging.LogRecord(
            "huggingface_hub.utils._http",
            logging.WARNING,
            __file__,
            1,
            "Rate limited while downloading model weights.",
            (),
            None,
        )

        warning_filter = _PublicHubAuthWarningFilter()
        assert warning_filter.filter(auth_record) is False
        assert warning_filter.filter(failure_record) is True

    @pytest.fixture
    def embedder(self, monkeypatch):
        """Use a deterministic local OpenCLIP stand-in; live CUDA is helper-smoked."""
        import sys
        from types import SimpleNamespace
        import numpy as np
        import torch

        class FakeModel:
            visual = SimpleNamespace(output_dim=512)

            def eval(self):
                return self

            def encode_image(self, images):
                rgb = images.mean(dim=(-2, -1))
                repeats = (512 + rgb.shape[1] - 1) // rgb.shape[1]
                return rgb.repeat(1, repeats)[:, :512]

            def encode_text(self, tokens):
                rows = torch.arange(1, tokens.shape[0] + 1, dtype=torch.float32)
                return rows[:, None].repeat(1, 512)

        def preprocess(image):
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                FakeModel(), None, preprocess
            ),
            get_tokenizer=lambda model_name: (
                lambda queries: torch.ones((len(queries), 4), dtype=torch.long)
            ),
        )
        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

        from src.config import EmbeddingConfig
        from src.embedder import CLIPEmbedder
        config = EmbeddingConfig(device="cpu")
        return CLIPEmbedder(config)

    def test_embed_text(self, embedder):
        """Test text embedding generation."""
        embs = embedder.embed_text(["a cat sitting on a mat"])
        assert embs.shape[0] == 1
        assert embs.shape[1] == 512  # ViT-B-32

    def test_embed_images(self, embedder, tmp_path):
        """Test image embedding generation."""
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        img_path = str(tmp_path / "test.jpg")
        img.save(img_path)

        embs = embedder.embed_images([img_path])
        assert embs.shape == (1, 512)

    def test_embedding_similarity(self, embedder, tmp_path):
        """Test that similar content produces similar embeddings."""
        import numpy as np
        from PIL import Image

        # Create two similar images (red) and one different (blue)
        red1 = Image.new("RGB", (224, 224), color="red")
        red2 = Image.new("RGB", (224, 224), color=(255, 10, 10))
        blue = Image.new("RGB", (224, 224), color="blue")

        for name, img in [("red1.jpg", red1), ("red2.jpg", red2), ("blue.jpg", blue)]:
            img.save(str(tmp_path / name))

        embs = embedder.embed_images([
            str(tmp_path / "red1.jpg"),
            str(tmp_path / "red2.jpg"),
            str(tmp_path / "blue.jpg"),
        ])

        # Red images should be more similar to each other than to blue
        sim_red = np.dot(embs[0], embs[1])
        sim_diff = np.dot(embs[0], embs[2])
        assert sim_red > sim_diff, "Similar images should have higher similarity"


class TestVectorStore:
    """Test ChromaDB vector store operations."""

    @pytest.fixture
    def store(self, tmp_path):
        from src.config import VectorDBConfig
        from src.vector_store import ScreenLensVectorStore
        config = VectorDBConfig(
            persist_directory=str(tmp_path / "chromadb"),
            collection_name="test_collection",
        )
        return ScreenLensVectorStore(config)

    def test_add_and_count(self, store):
        import numpy as np
        frames = [
            {"frame_id": 0, "timestamp": 0.0, "timestamp_str": "00:00:00.000",
             "path": "/tmp/f0.jpg", "caption": "A red screen"},
            {"frame_id": 1, "timestamp": 1.0, "timestamp_str": "00:00:01.000",
             "path": "/tmp/f1.jpg", "caption": "A blue menu bar"},
        ]
        embeddings = np.random.randn(2, 512).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        store.add_frames(frames, embeddings)
        assert store.count() == 2

    def test_search_by_embedding(self, store):
        import numpy as np
        frames = [
            {"frame_id": 0, "timestamp": 0.0, "timestamp_str": "00:00:00.000",
             "path": "/tmp/f0.jpg", "caption": "Red screen"},
        ]
        emb = np.random.randn(1, 512).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        store.add_frames(frames, emb)

        results = store.search_by_embedding(emb[0], top_k=1)
        assert len(results) == 1
        assert results[0]["caption"] == "Red screen"

    def test_reset(self, store):
        import numpy as np
        frames = [{"frame_id": 0, "timestamp": 0.0, "path": "/tmp/f.jpg", "caption": "test"}]
        emb = np.random.randn(1, 512).astype(np.float32)
        store.add_frames(frames, emb)
        assert store.count() == 1
        store.reset()
        assert store.count() == 0


class TestPipeline:
    """Test LangGraph pipeline construction."""

    def test_ingest_graph_builds(self):
        from src.pipeline import build_ingest_graph
        graph = build_ingest_graph()
        assert graph is not None

    def test_search_graph_builds(self):
        from src.pipeline import build_search_graph
        graph = build_search_graph()
        assert graph is not None

    def test_full_graph_builds(self):
        from src.pipeline import build_full_graph
        graph = build_full_graph()
        assert graph is not None

    def test_search_summary_uses_selected_vllm_client(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        captured = {}

        class FakeClient:
            def __init__(self, config):
                captured["backend"] = config.backend

            def chat(self, system, user, **kwargs):
                captured["system"] = system
                captured["user"] = user
                captured["kwargs"] = kwargs
                return "DGX summary"

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = ScreenLensConfig()
        # Isolate the summary cache from any real ./data run folder.
        config.vector_db.persist_directory = str(tmp_path / "chromadb")
        config.captioning.backend = CaptionBackend.vllm
        # Summaries run on the TEXT role, so it is the reconstruction backend
        # that decides the client (both are vLLM on Spark).
        config.reconstruction.backend = InferenceBackend.vllm

        result = pipeline.summarize_node({
            "query": "What application is shown?",
            "search_results": [{
                "timestamp_str": "00:00:01.000",
                "caption": "A terminal shows ScreenLens.",
                "score": 0.9,
            }],
            "config": config.model_dump(),
        })

        assert result["summary"] == "DGX summary"
        assert captured["backend"] == CaptionBackend.vllm
        assert captured["kwargs"]["extra"] == {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_summary_refuses_to_present_degenerate_output_as_an_answer(self, tmp_path, monkeypatch):
        """A summary is shown to the user and written to disk; a model stuck in
        a repetition loop must not have its output pass for one."""
        import src.pipeline as pipeline
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig
        from src.omlx_client import InferenceDegenerateError

        class FakeClient:
            backend = InferenceBackend.omlx
            model = "text-model"

            def __init__(self, config):
                pass

            def chat(self, system, user, **kwargs):
                return "<BOS>" * 60

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = ScreenLensConfig()
        # Isolate the summary cache from any real ./data run folder.
        config.vector_db.persist_directory = str(tmp_path / "chromadb")
        config.captioning.backend = CaptionBackend.omlx
        config.reconstruction.backend = InferenceBackend.omlx

        with pytest.raises(InferenceDegenerateError, match="degenerate repeated output"):
            pipeline.summarize_node({
                "query": "what is shown?",
                "search_results": [{
                    "timestamp_str": "00:00:01.000",
                    "caption": "A terminal.",
                    "score": 0.9,
                }],
                "config": config.model_dump(),
            })

    def test_caption_chunks_budget_each_skewed_caption_in_order(self):
        from src.pipeline import (
            _chunk_captions_by_budget,
            _compute_chunk_strategy,
            _estimated_caption_tokens,
        )

        captions = [
            {
                "frame_id": i,
                "timestamp_str": f"00:00:{i:02d}.000",
                "caption": "x" * 3000,
            }
            for i in range(54)
        ]
        captions[20]["caption"] = "runaway `...`, " * 5000  # ~75K chars

        strategy = _compute_chunk_strategy(captions, 32768)
        chunks = _chunk_captions_by_budget(
            captions,
            strategy["safe_context_tokens"],
        )

        assert strategy["strategy"] == "hierarchical"
        assert len(chunks) > 2
        flattened = [item for chunk in chunks for item in chunk]
        frame_ids = [item["frame_id"] for item in flattened]
        assert frame_ids == sorted(frame_ids)
        rebuilt = {i: "" for i in range(54)}
        for item in flattened:
            rebuilt[item["frame_id"]] += item["caption"]
        assert [rebuilt[i] for i in range(54)] == [item["caption"] for item in captions]
        assert all(
            sum(_estimated_caption_tokens(item) for item in chunk)
            <= strategy["safe_context_tokens"]
            for chunk in chunks
        )

    def test_caption_chunks_split_one_caption_larger_than_budget(self):
        from src.pipeline import _chunk_captions_by_budget, _estimated_caption_tokens

        original = "0123456789" * 1500
        chunks = _chunk_captions_by_budget(
            [{"frame_id": 7, "timestamp_str": "00:00:07.000", "caption": original}],
            1000,
        )
        pieces = [item for chunk in chunks for item in chunk]

        assert len(pieces) > 1
        assert "".join(item["caption"] for item in pieces) == original
        assert all(_estimated_caption_tokens(item) <= 1000 for item in pieces)

    def test_reconstruction_single_pass_extraction_uses_full_context_headroom(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct

        calls = []

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append({"user": user, "max_tokens": max_tokens})
            return "extracted detail"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        result = reconstruct._extract_segment_notes(
            [{
                "frame_id": 0,
                "timestamp_str": "00:00:00.000",
                "caption": "A terminal displays app.py.",
            }],
            LegacyClient(),
            model_context=32768,
        )

        assert result == ["[Full recording]\nextracted detail"]
        assert [call["max_tokens"] for call in calls] == [32768]
        assert "Keep the response at or below 1,400 tokens" in calls[0]["user"]

    def test_reconstruction_multi_chunk_extraction_uses_full_context_headroom(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct

        calls = []

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append({"user": user, "max_tokens": max_tokens})
            return f"segment-{len(calls)}"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)
        captions = [
            {
                "frame_id": i,
                "timestamp_str": f"00:00:{i:02d}.000",
                "caption": f"Frame {i} shows source code.",
            }
            for i in range(reconstruct.MAX_CAPTIONS_PER_CHUNK + 1)
        ]

        result = reconstruct._extract_segment_notes(
            captions,
            LegacyClient(),
            model_context=32768,
        )

        assert len(result) == 2
        assert len(calls) == 2
        assert all(call["max_tokens"] == 32768 for call in calls)

    def test_reconstruction_long_form_ceiling_tracks_larger_server_context(self):
        import src.reconstruct as reconstruct

        class ClientWithSmallerCaptionDefault:
            _default_max_tokens = 32768

        assert reconstruct._long_form_output_ceiling(
            ClientWithSmallerCaptionDefault(),
            262144,
        ) == 262144

    def test_reconstruction_extraction_splits_truncated_caption_group(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct
        from src.omlx_client import InferenceTruncatedError

        calls = []

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append(user)
            if len(calls) == 1:
                raise InferenceTruncatedError("vllm", max_tokens)
            return f"bounded-{len(calls)}"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)
        captions = [
            {
                "frame_id": i,
                "timestamp_str": f"00:00:0{i}.000",
                "caption": f"Frame {i} source code.",
            }
            for i in range(4)
        ]

        result = reconstruct._extract_segment_notes(
            captions,
            object(),
            model_context=32768,
        )

        assert len(calls) == 3
        assert len(result) == 2
        assert result[0].startswith("[Segment 1a:")
        assert result[1].startswith("[Segment 1b:")
        assert len(calls[1]) < len(calls[0])
        assert len(calls[2]) < len(calls[0])

    def test_reconstruction_synthesis_uses_full_ceiling_after_planning_headroom(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct

        captured = {}

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            captured["user"] = user
            captured["max_tokens"] = max_tokens
            return "artifact"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        result = reconstruct._hierarchical_synthesize(
            ["a short extraction note"],
            "Rebuild the artifact.",
            "Return only the artifact.",
            LegacyClient(),
            model_context=32768,
        )

        assert result == "artifact"
        assert reconstruct._estimated_text_tokens(captured["user"]) < 32768 - 8192
        assert captured["max_tokens"] == 32768

    def test_reconstruction_synthesis_splits_one_oversized_note(self, monkeypatch):
        import src.reconstruct as reconstruct

        calls = []
        chunk_budgets = []

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append({
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
            })
            return f"condensed-{len(calls)}"

        real_chunk_texts = reconstruct._chunk_texts_by_budget

        def capture_chunk_budget(items, token_budget):
            chunk_budgets.append(token_budget)
            return real_chunk_texts(items, token_budget)

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)
        monkeypatch.setattr(
            reconstruct,
            "_chunk_texts_by_budget",
            capture_chunk_budget,
        )

        result = reconstruct._hierarchical_synthesize(
            ["`...`, " * 18000],
            "Rebuild the artifact.",
            "Return only the artifact.",
            LegacyClient(),
            model_context=32768,
        )

        assert result.startswith("condensed-")
        assert len(calls) >= 4
        assert all(len(call["user"]) < 40000 for call in calls)
        assert 2048 < chunk_budgets[0] < 32768 - 2548

        intermediate_calls = [
            call for call in calls
            if call["system"] == reconstruct.EXTRACT_SEGMENT_SYSTEM
        ]
        assert all("TASK FOCUS:\nRebuild the artifact." in call["user"]
                   for call in intermediate_calls)
        assert all("discard material solely about other files/artifacts" in call["user"]
                   for call in intermediate_calls)
        assert all(call["max_tokens"] == 32768 for call in intermediate_calls)
        assert calls[-1]["max_tokens"] == 32768

    def test_reconstruction_synthesis_retries_truncated_group_with_less_input(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct
        from src.omlx_client import InferenceTruncatedError

        calls = []
        truncated_once = False

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            nonlocal truncated_once
            calls.append({
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
            })
            if system == reconstruct.EXTRACT_SEGMENT_SYSTEM and not truncated_once:
                truncated_once = True
                raise InferenceTruncatedError("vllm", max_tokens)
            return "focused bounded notes"

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        result = reconstruct._hierarchical_synthesize(
            ["agent.py exact source detail " * 2400],
            "Reconstruct agent.py only.",
            "Return only agent.py.",
            LegacyClient(),
            model_context=32768,
        )

        intermediate_calls = [
            call for call in calls
            if call["system"] == reconstruct.EXTRACT_SEGMENT_SYSTEM
        ]
        assert result == "focused bounded notes"
        assert len(intermediate_calls) >= 3
        assert len(intermediate_calls[1]["user"]) < len(intermediate_calls[0]["user"])
        assert "TASK FOCUS:\nReconstruct agent.py only." in intermediate_calls[1]["user"]

    def test_reconstruction_synthesis_stops_when_condensation_makes_no_progress(
        self, monkeypatch,
    ):
        import src.reconstruct as reconstruct

        calls = []

        class LegacyClient:
            _default_max_tokens = 32768

        def fake_generate(client, system, user, *, max_tokens, temperature):
            calls.append({"user": user, "max_tokens": max_tokens})
            # Deliberately violate the requested compression contract. The
            # recursion guard must reject this instead of calling itself until
            # Python raises RecursionError.
            return "not-condensed " * 4000

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate)

        with pytest.raises(RuntimeError, match="(?i)(progress|condens)"):
            reconstruct._hierarchical_synthesize(
                ["first detail " * 2500, "second detail " * 2500],
                "Rebuild the artifact.",
                "Return only the artifact.",
                LegacyClient(),
                model_context=32768,
            )

        assert 1 <= len(calls) <= 4

    @pytest.mark.parametrize("backend", ["vllm", "omlx"])
    def test_direct_caption_config_uses_reconstruction_timeout(self, backend):
        import src.reconstruct as reconstruct
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend(backend)
        # Reconstruction runs on the text role, so its own backend selects the
        # client; both roles point at one endpoint on a real deployment.
        config.reconstruction.backend = InferenceBackend(backend)
        config.captioning.vllm_timeout_seconds = 120
        config.captioning.omlx_timeout_seconds = 120
        config.reconstruction.timeout_seconds = 2400

        direct = reconstruct._reconstruction_captioning_config(config)

        assert direct.backend == CaptionBackend(backend)
        assert direct.vllm_timeout_seconds == (2400 if backend == "vllm" else 120)
        assert direct.omlx_timeout_seconds == (2400 if backend == "omlx" else 120)

    def test_ollama_caption_config_uses_direct_reconstruction_backend(self):
        import src.reconstruct as reconstruct
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend.ollama
        config.reconstruction.backend = InferenceBackend.vllm
        config.reconstruction.model = "org/reconstruction-model"
        config.reconstruction.api_key = "direct-key"

        direct = reconstruct._reconstruction_captioning_config(config)

        assert direct.backend == CaptionBackend.vllm
        assert direct.vllm_model == "org/reconstruction-model"
        assert direct.vllm_api_key == "direct-key"
        assert direct.vllm_timeout_seconds == config.reconstruction.timeout_seconds
        assert direct.max_tokens == config.reconstruction.max_tokens


class TestRunReuse:
    """Run-folder reuse helpers shared by the CLI and the web command deck."""

    @staticmethod
    def _write_frames_meta(run, video, config, frames):
        import json
        (run / "frames").mkdir(parents=True, exist_ok=True)
        meta = {
            "video": str(video.resolve()),
            "video_size": video.stat().st_size,
            "extraction": config.frame_extraction.model_dump(mode="json"),
            "frames": frames,
        }
        (run / "frames" / "frames_meta.json").write_text(json.dumps(meta))

    def test_find_reusable_run_picks_newest_with_marker(self, tmp_path):
        from src.config import ScreenLensConfig
        from src.session import find_reusable_run

        config = ScreenLensConfig()
        config.data_dir = tmp_path
        for name in ("demo_20260101_000000", "demo_20260102_000000",
                     "demo_20260103_000000", "other_20260104_000000"):
            (tmp_path / name).mkdir()
        (tmp_path / "demo_20260101_000000" / "ocr").mkdir()
        (tmp_path / "demo_20260102_000000" / "ocr").mkdir()

        found = find_reusable_run(config, tmp_path / "demo.mov", "ocr")
        assert found == tmp_path / "demo_20260102_000000"
        assert find_reusable_run(config, tmp_path / "missing.mov", "ocr") is None

    def test_reuse_video_run_keeps_stem_collection(self, tmp_path):
        from src.config import ScreenLensConfig
        from src.session import reuse_video_run

        config = ScreenLensConfig()
        config.data_dir = tmp_path
        run = tmp_path / "demo_20260102_000000"
        run.mkdir()

        slug = reuse_video_run(config, tmp_path / "demo.mov", run)
        assert slug == run.name
        assert config.data_dir == run
        assert config.vector_db.persist_directory == str(run / "chromadb")
        assert config.vector_db.collection_name == "screenlens_demo"

    def test_extraction_meta_matches_guards_video_and_config(self, tmp_path):
        from src.config import ExtractionStrategy, ScreenLensConfig
        from src.session import extraction_meta_matches

        video = tmp_path / "demo.mov"
        video.write_bytes(b"x" * 123)
        config = ScreenLensConfig()
        config.data_dir = tmp_path
        run = tmp_path / "demo_20260102_000000"
        self._write_frames_meta(run, video, config, [])

        assert extraction_meta_matches(run, video, config)

        config.frame_extraction.strategy = ExtractionStrategy.fixed_fps
        assert not extraction_meta_matches(run, video, config)
        config.frame_extraction.strategy = ExtractionStrategy.keyframe

        video.write_bytes(b"y" * 124)
        assert not extraction_meta_matches(run, video, config)
        assert not extraction_meta_matches(tmp_path / "no_such_run", video, config)

    def test_load_cached_frames_requires_frame_files(self, tmp_path):
        from src.config import ScreenLensConfig
        from src.session import load_cached_frames

        video = tmp_path / "demo.mov"
        video.write_bytes(b"vid")
        config = ScreenLensConfig()
        config.data_dir = tmp_path
        run = tmp_path / "demo_20260102_000000"
        frame_path = run / "frames" / "frame_000000.jpg"
        frames = [{"frame_id": 0, "timestamp": 0.0, "path": str(frame_path)}]
        self._write_frames_meta(run, video, config, frames)

        assert load_cached_frames(run, video, config) is None  # image missing
        frame_path.write_bytes(b"jpg")
        assert load_cached_frames(run, video, config) == frames

    def test_transcribe_run_matches(self, tmp_path):
        import json
        from src.session import transcribe_run_matches

        video = tmp_path / "demo.mov"
        video.write_bytes(b"vid")
        run = tmp_path / "demo_20260102_000000"
        run.mkdir()
        assert transcribe_run_matches(run, video)  # no meta yet → reusable

        out = run / "output"
        out.mkdir()
        (out / "transcribe_meta.json").write_text(json.dumps(
            {"video": str(video.resolve()), "video_size": video.stat().st_size}))
        assert transcribe_run_matches(run, video)

        video.write_bytes(b"longer video bytes")
        assert not transcribe_run_matches(run, video)

        (out / "transcribe_meta.json").write_text(json.dumps(
            {"video": str((tmp_path / "other.mov").resolve())}))
        assert not transcribe_run_matches(run, video)

    def test_ingest_node_reuses_cached_extraction(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline
        from src.config import ScreenLensConfig

        video = tmp_path / "demo.mov"
        video.write_bytes(b"vid")
        config = ScreenLensConfig()
        config.data_dir = tmp_path / "demo_20260102_000000"
        frame_path = config.data_dir / "frames" / "frame_000000.jpg"
        frames = [{"frame_id": 0, "timestamp": 0.0, "timestamp_str": "00:00:00.000",
                   "path": str(frame_path), "width": 4, "height": 4}]
        self._write_frames_meta(config.data_dir, video, config, frames)
        frame_path.write_bytes(b"jpg")

        monkeypatch.setattr(pipeline, "get_video_metadata", lambda *a: {"mock": True})

        def _no_extract(*a, **k):
            raise AssertionError("extract_frames must not run on a meta cache hit")

        monkeypatch.setattr(pipeline, "extract_frames", _no_extract)
        result = pipeline.ingest_node(
            {"video_path": str(video), "config": config.model_dump()})
        assert result["num_frames"] == 1
        assert result["frames_meta"] == frames
        assert result["stage"] == "ingested"

    def test_embed_node_skips_populated_store(self, tmp_path, monkeypatch):
        import numpy as np
        import src.pipeline as pipeline
        from src.config import ScreenLensConfig
        from src.vector_store import ScreenLensVectorStore

        config = ScreenLensConfig()
        config.data_dir = tmp_path
        config.vector_db.persist_directory = str(tmp_path / "chromadb")
        config.vector_db.collection_name = "test_embed_reuse"

        frames = [{"frame_id": i, "timestamp": float(i), "timestamp_str": f"00:00:0{i}.000",
                   "path": f"/tmp/f{i}.jpg", "width": 1, "height": 1, "caption": f"c{i}"}
                  for i in range(2)]
        store = ScreenLensVectorStore(config.vector_db)
        store.add_frames(frames, np.random.randn(2, 512).astype(np.float32))

        class _BombEmbedder:
            def __init__(self, *a, **k):
                raise AssertionError("CLIP must not load when the store is complete")

        monkeypatch.setattr(pipeline, "CLIPEmbedder", _BombEmbedder)
        result = pipeline.embed_node(
            {"captioned_frames": frames, "config": config.model_dump()})
        assert result["stage"] == "embedded"
        assert result["embeddings_shape"] == [2]

    def test_embed_node_rebuilds_partial_store(self, tmp_path, monkeypatch):
        import numpy as np
        import src.pipeline as pipeline
        from src.config import ScreenLensConfig
        from src.vector_store import ScreenLensVectorStore

        config = ScreenLensConfig()
        config.data_dir = tmp_path
        config.vector_db.persist_directory = str(tmp_path / "chromadb")
        config.vector_db.collection_name = "test_embed_partial"

        frames = [{"frame_id": i, "timestamp": float(i), "timestamp_str": f"00:00:0{i}.000",
                   "path": f"/tmp/f{i}.jpg", "width": 1, "height": 1, "caption": f"c{i}"}
                  for i in range(2)]
        store = ScreenLensVectorStore(config.vector_db)
        store.add_frames(frames[:1], np.random.randn(1, 512).astype(np.float32))

        class _FakeEmbedder:
            def __init__(self, cfg):
                pass

            def embed_images(self, paths):
                return np.random.randn(len(paths), 512).astype(np.float32)

        monkeypatch.setattr(pipeline, "CLIPEmbedder", _FakeEmbedder)
        result = pipeline.embed_node(
            {"captioned_frames": frames, "config": config.model_dump()})
        assert result["embeddings_shape"] == [2, 512]
        assert ScreenLensVectorStore(config.vector_db).count() == 2


# ── Round-2 efficiency: shared embedder + summary cache ─────────────────────

class TestSharedEmbedder:
    def test_shared_embedder_reuses_one_instance_per_model_and_device(self):
        from src.config import EmbeddingConfig
        from src.embedder import _SHARED, get_shared_embedder

        _SHARED.clear()
        try:
            a = get_shared_embedder(EmbeddingConfig(device="cpu"))
            b = get_shared_embedder(EmbeddingConfig(device="cpu"))
            other = get_shared_embedder(EmbeddingConfig(device="mps"))
            assert a is b
            assert a is not other
        finally:
            _SHARED.clear()

    def test_search_node_uses_the_shared_embedder(self, monkeypatch):
        import numpy as np
        import src.pipeline as pipeline
        from src.config import ScreenLensConfig

        calls = []

        class _FakeEmbedder:
            def embed_text(self, queries):
                return np.ones((len(queries), 4), dtype=np.float32)

        def fake_shared(config):
            calls.append(config)
            return _FakeEmbedder()

        class _FakeStore:
            def __init__(self, vector_config):
                pass

            def search_by_embedding(self, emb, top_k=10):
                return [{
                    "timestamp_str": "00:00:01.000",
                    "caption": "A terminal.",
                    "score": 0.9,
                }]

        monkeypatch.setattr(pipeline, "get_shared_embedder", fake_shared)
        monkeypatch.setattr(pipeline, "ScreenLensVectorStore", _FakeStore)

        config = ScreenLensConfig()
        result = pipeline.search_node(
            {"query": "terminal", "config": config.model_dump()})
        assert len(calls) == 1
        assert result["search_results"][0]["score"] == 0.9


class TestSummaryCache:
    def _config(self, tmp_path):
        from src.config import CaptionBackend, InferenceBackend, ScreenLensConfig

        config = ScreenLensConfig()
        config.captioning.backend = CaptionBackend.omlx
        config.reconstruction.backend = InferenceBackend.omlx
        config.vector_db.persist_directory = str(tmp_path / "chromadb")
        return config

    def _state(self, config, query="what is shown?"):
        return {
            "query": query,
            "search_results": [{
                "timestamp_str": "00:00:01.000",
                "caption": "A terminal.",
                "score": 0.9,
            }],
            "config": config.model_dump(),
        }

    def test_summarize_node_reuses_cached_answer(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline
        from src.config import InferenceBackend

        calls = {"n": 0}

        class FakeClient:
            backend = InferenceBackend.omlx
            model = "text-model"

            def __init__(self, config):
                pass

            def chat(self, system, user, **kwargs):
                calls["n"] += 1
                return "cached answer"

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = self._config(tmp_path)

        first = pipeline.summarize_node(self._state(config))
        second = pipeline.summarize_node(self._state(config))

        assert first["summary"] == second["summary"] == "cached answer"
        assert calls["n"] == 1
        assert (tmp_path / "summary_cache.json").exists()

    def test_summarize_node_cache_misses_on_a_new_query(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline
        from src.config import InferenceBackend

        calls = {"n": 0}

        class FakeClient:
            backend = InferenceBackend.omlx
            model = "text-model"

            def __init__(self, config):
                pass

            def chat(self, system, user, **kwargs):
                calls["n"] += 1
                return f"answer {calls['n']}"

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = self._config(tmp_path)

        pipeline.summarize_node(self._state(config, query="first question"))
        pipeline.summarize_node(self._state(config, query="second question"))

        assert calls["n"] == 2

    def test_summarize_all_node_caches_full_video_summary(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline
        from src.config import InferenceBackend

        calls = {"n": 0}

        class FakeClient:
            backend = InferenceBackend.omlx
            model = "text-model"
            base_url = "http://127.0.0.1:8102/v1"

            def __init__(self, config):
                pass

            def chat(self, system, user, **kwargs):
                calls["n"] += 1
                return "whole-video summary"

        monkeypatch.setattr(pipeline, "InferenceClient", FakeClient)
        config = self._config(tmp_path)
        config.data_dir = tmp_path
        state = {
            "captioned_frames": [
                {"frame_id": 0, "timestamp_str": "00:00:00.000",
                 "caption": "Open the app."},
                {"frame_id": 1, "timestamp_str": "00:00:02.000",
                 "caption": "Click save."},
            ],
            "config": config.model_dump(),
        }

        first = pipeline.summarize_all_node(state)
        second = pipeline.summarize_all_node(state)

        assert first["summary"] == second["summary"] == "whole-video summary"
        assert calls["n"] == 1
        assert (tmp_path / "summary_cache.json").exists()
