"""
Tests for reconstruction efficiency: plan caching across QA retries and
Pass-1 segment-notes persistence.

Run with: pytest tests/test_reconstruct.py -v
"""
import hashlib
import json

import pytest

import src.reconstruct as reconstruct
from src.config import ScreenLensConfig


class _DummyClient:
    """Minimal stand-in for InferenceClient — only what the nodes touch."""

    context_size = 32768

    def chat(self, *args, **kwargs):
        raise AssertionError("unexpected direct inference call")


@pytest.fixture()
def patched_client(monkeypatch):
    """Route get_inference_client to a dummy so no real client is built."""
    client = _DummyClient()
    monkeypatch.setattr(reconstruct, "get_inference_client", lambda config: client)
    return client


def _python_state(**overrides):
    """State sufficient for plan_node's python_code branch."""
    state = {
        "content_type": "python_code",
        "captions": [{"timestamp_str": "00:00", "caption": "editing a.py"}],
        "config": ScreenLensConfig().model_dump(),
        "qa_feedback": "missing error handling in a.py",
        "qa_iteration": 1,
    }
    state.update(overrides)
    return state


def _sequential_state(tmp_path, **overrides):
    """State sufficient for reconstruct_sequential."""
    state = {
        "folder_path": str(tmp_path),
        "folder_name": tmp_path.name,
        "captions": [{"timestamp_str": "00:00", "caption": "c"}],
        "captions_sha1": "abc",
        "config": ScreenLensConfig().model_dump(),
        "reconstruction_tasks": [
            {
                "filename": "a.py",
                "description": "x",
                "prompt": "Reconstruct it.",
                "output_type": "python",
            }
        ],
        "system_prompt": "sys",
        "qa_iteration": 0,
    }
    state.update(overrides)
    return state


class TestPlanNodeCaching:
    """CHANGE 1: python_code plan is cached across QA retries."""

    def test_cached_plan_skips_generation(self, monkeypatch, patched_client, capsys):
        calls = []

        def forbidden_generate(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("generate_text must not run with a cached plan")

        monkeypatch.setattr(reconstruct, "generate_text", forbidden_generate)

        cached = {
            "files": [{"filename": "a.py", "description": "x"}],
            "parallel_safe": False,
        }
        result = reconstruct.plan_node(_python_state(reconstruction_plan=cached))

        assert calls == []
        # Tasks are still rebuilt and embed the current QA feedback.
        tasks = result["reconstruction_tasks"]
        assert [t["filename"] for t in tasks] == ["a.py"]
        assert "missing error handling in a.py" in tasks[0]["prompt"]
        assert result["parallel_safe"] is False
        assert result["reconstruction_plan"] == cached

        out = capsys.readouterr().out
        assert "cached plan" in out
        assert "- a.py: x" in out

    def test_uncached_plan_generates_once_and_returns_plan(
        self, monkeypatch, patched_client
    ):
        calls = []
        payload = (
            '{"files": [{"filename": "a.py", "description": "x"}],'
            ' "parallel_safe": false}'
        )

        def fake_generate_text(client, system, user, **kwargs):
            calls.append((system, user))
            return payload

        monkeypatch.setattr(reconstruct, "generate_text", fake_generate_text)

        result = reconstruct.plan_node(_python_state(qa_iteration=0, qa_feedback=""))

        assert len(calls) == 1
        assert result["reconstruction_plan"] == {
            "files": [{"filename": "a.py", "description": "x"}],
            "parallel_safe": False,
        }
        assert [t["filename"] for t in result["reconstruction_tasks"]] == ["a.py"]


class TestSegmentNotesPersistence:
    """CHANGE 2: Pass-1 notes persist/load helpers."""

    def test_captions_sha1(self, tmp_path):
        captions_file = tmp_path / "all_captions.json"
        captions_file.write_bytes(b"[]")
        assert reconstruct._captions_sha1(captions_file) == hashlib.sha1(b"[]").hexdigest()

    def test_persist_load_round_trip(self, tmp_path):
        notes = ["note one", "note two"]
        reconstruct._persist_segment_notes(tmp_path, "abc123", notes)
        assert reconstruct._load_persisted_segment_notes(tmp_path, "abc123") == notes

    def test_load_with_wrong_sha1_returns_empty(self, tmp_path):
        reconstruct._persist_segment_notes(tmp_path, "abc123", ["n1"])
        assert reconstruct._load_persisted_segment_notes(tmp_path, "different") == []

    def test_load_missing_or_corrupt_returns_empty(self, tmp_path):
        assert reconstruct._load_persisted_segment_notes(tmp_path, "abc123") == []

        path = reconstruct._segment_notes_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json {", encoding="utf-8")
        assert reconstruct._load_persisted_segment_notes(tmp_path, "abc123") == []


class TestReconstructSequentialNotes:
    """CHANGE 2: in-state reuse and fresh-extraction persistence."""

    def test_reuses_in_state_notes_without_extraction(
        self, monkeypatch, patched_client, tmp_path
    ):
        def forbidden_extract(*args, **kwargs):
            raise AssertionError("_extract_segment_notes must not run")

        synth_calls = []

        def fake_synthesize(notes, prefix, system, client, model_context, **kwargs):
            synth_calls.append((notes, prefix, system, model_context))
            return "content"

        monkeypatch.setattr(reconstruct, "_extract_segment_notes", forbidden_extract)
        monkeypatch.setattr(reconstruct, "_hierarchical_synthesize", fake_synthesize)

        state = _sequential_state(tmp_path, segment_notes=["cached note"])
        result = reconstruct.reconstruct_sequential(state)

        assert result["segment_notes"] == ["cached note"]
        assert result["artifacts"][0]["content"] == "content"
        assert synth_calls == [(["cached note"], "Reconstruct it.", "sys", 32768)]
        # Nothing was computed, so nothing is persisted.
        assert not reconstruct._segment_notes_path(tmp_path).exists()

    def test_persists_fresh_notes(self, monkeypatch, patched_client, tmp_path):
        monkeypatch.setattr(
            reconstruct, "_extract_segment_notes", lambda *a, **k: ["n1"]
        )
        monkeypatch.setattr(
            reconstruct, "_hierarchical_synthesize", lambda *a, **k: "content"
        )

        result = reconstruct.reconstruct_sequential(_sequential_state(tmp_path))

        assert result["segment_notes"] == ["n1"]
        persisted = json.loads(
            reconstruct._segment_notes_path(tmp_path).read_text(encoding="utf-8")
        )
        assert persisted == {"captions_sha1": "abc", "notes": ["n1"]}

    def test_skips_persist_without_sha1(self, monkeypatch, patched_client, tmp_path):
        monkeypatch.setattr(
            reconstruct, "_extract_segment_notes", lambda *a, **k: ["n1"]
        )
        monkeypatch.setattr(
            reconstruct, "_hierarchical_synthesize", lambda *a, **k: "content"
        )

        state = _sequential_state(tmp_path, captions_sha1="")
        reconstruct.reconstruct_sequential(state)

        assert not reconstruct._segment_notes_path(tmp_path).exists()


class TestReconstructFolderSeeding:
    """CHANGE 2: reconstruct_folder seeds initial_state from persisted notes."""

    def _write_captions(self, tmp_path):
        captions = [{"timestamp_str": "00:00", "caption": "c"}]
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()
        captions_file = captions_dir / "all_captions.json"
        captions_file.write_text(json.dumps(captions), encoding="utf-8")
        return captions, captions_file

    def _capture_graph(self, monkeypatch, captured):
        class _DummyPipeline:
            def invoke(self, initial_state):
                captured.update(initial_state)
                return {"stage": "done"}

        monkeypatch.setattr(
            reconstruct, "build_reconstruct_graph", lambda: _DummyPipeline()
        )

    def test_seeds_persisted_notes(self, monkeypatch, tmp_path):
        captions, captions_file = self._write_captions(tmp_path)
        sha1 = reconstruct._captions_sha1(captions_file)
        reconstruct._persist_segment_notes(tmp_path, sha1, ["persisted note"])

        captured = {}
        self._capture_graph(monkeypatch, captured)

        result = reconstruct.reconstruct_folder(str(tmp_path), ScreenLensConfig())

        assert result == {"stage": "done"}
        assert captured["captions"] == captions
        assert captured["captions_sha1"] == sha1
        assert captured["segment_notes"] == ["persisted note"]

    def test_no_persisted_notes_leaves_state_without_them(
        self, monkeypatch, tmp_path
    ):
        self._write_captions(tmp_path)

        captured = {}
        self._capture_graph(monkeypatch, captured)

        reconstruct.reconstruct_folder(str(tmp_path), ScreenLensConfig())

        assert "segment_notes" not in captured
        assert captured["captions_sha1"]
