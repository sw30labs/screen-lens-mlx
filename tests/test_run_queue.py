"""Queue, skip-success, and content-naming helpers."""
from datetime import datetime
from pathlib import Path

import pytest

from src.config import ScreenLensConfig
from src.session import (
    apply_content_name,
    apply_video_slug,
    chroma_collection_name,
    find_reusable_run,
    find_successful_run,
    infer_content_slug,
    ingest_succeeded,
    list_videos_chronological,
    recording_stamp,
    resolve_video_queue,
    reuse_video_run,
    run_belongs_to_video,
    safe_video_stem,
    slugify_title,
    video_recorded_at,
    write_run_identity,
)


def _touch_video(path: Path, size: int = 64) -> Path:
    path.write_bytes(b"x" * size)
    return path


def _successful_ingest(run: Path, video: Path, config: ScreenLensConfig) -> None:
    (run / "captions").mkdir(parents=True)
    (run / "captions" / "all_captions.json").write_text(
        '[{"caption": "# MRM Guidelines\\n\\nA policy walkthrough."}]',
        encoding="utf-8",
    )
    (run / "chromadb").mkdir(parents=True)
    (run / "chromadb" / "chroma.sqlite3").write_bytes(b"sqlite")
    write_run_identity(run, {
        "video": str(video.resolve()),
        "video_size": video.stat().st_size,
        "source_name": video.name,
        "collection": config.vector_db.collection_name,
        "status": "success",
    })


class TestChronologicalQueue:
    def test_screen_recording_clock_beats_lexicographic_name(self, tmp_path):
        later = _touch_video(tmp_path / "Screen Recording 2026-08-17 at 10.00.00 AM.mov")
        earlier = _touch_video(tmp_path / "Screen Recording 2026-08-17 at 9.00.00 AM.mov")
        _touch_video(tmp_path / "notes.txt")
        ordered = list_videos_chronological(tmp_path)
        assert [p.name for p in ordered] == [earlier.name, later.name]
        assert video_recorded_at(earlier) == datetime(2026, 8, 17, 9, 0, 0)
        assert video_recorded_at(later) == datetime(2026, 8, 17, 10, 0, 0)

    def test_narrow_nbsp_before_am_parses(self, tmp_path):
        # macOS often inserts U+202F before AM/PM
        path = _touch_video(tmp_path / "Screen Recording 2026-08-17 at 9.29.20\u202fAM.mov")
        assert video_recorded_at(path) == datetime(2026, 8, 17, 9, 29, 20)
        assert recording_stamp(path) == "20260817_092920"

    def test_compact_stamp_filename_keeps_clock(self, tmp_path):
        path = _touch_video(tmp_path / "20260817_092920.mov")
        assert video_recorded_at(path) == datetime(2026, 8, 17, 9, 29, 20)
        assert recording_stamp(path) == "20260817_092920"

    def test_empty_source_means_default_folder(self, tmp_path):
        inbox = tmp_path / "input"
        inbox.mkdir()
        first = _touch_video(inbox / "Screen Recording 2026-08-17 at 9.00.00 AM.mov")
        assert resolve_video_queue(None, default_folder=inbox) == [first.resolve()]
        assert resolve_video_queue("", default_folder=inbox) == [first.resolve()]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="video not found"):
            resolve_video_queue(tmp_path / "nope.mov")


class TestSkipSuccessful:
    def test_skip_survives_content_rename(self, tmp_path):
        video = _touch_video(tmp_path / "Screen Recording 2026-08-17 at 9.29.20 AM.mov")
        data = tmp_path / "data"
        named = data / "mrm_guidelines_20260817_092920"
        named.mkdir(parents=True)
        config = ScreenLensConfig()
        _successful_ingest(named, video, config)

        assert ingest_succeeded(named)
        assert run_belongs_to_video(named, video)
        found = find_successful_run(data, video, "ingest")
        assert found == named

    def test_partial_run_is_not_success(self, tmp_path):
        video = _touch_video(tmp_path / "clip.mov")
        data = tmp_path / "data"
        partial = data / "clip_20260817_090000"
        (partial / "frames").mkdir(parents=True)
        write_run_identity(partial, {
            "video": str(video.resolve()),
            "video_size": video.stat().st_size,
            "source_name": video.name,
            "status": "running",
        })
        assert find_successful_run(data, video, "ingest") is None

    def test_error_status_is_not_success_even_with_empty_chroma(self, tmp_path):
        video = _touch_video(tmp_path / "clip.mov")
        data = tmp_path / "data"
        failed = data / "clip_20260817_090000"
        (failed / "captions").mkdir(parents=True)
        (failed / "captions" / "all_captions.json").write_text("[]", encoding="utf-8")
        (failed / "chromadb").mkdir(parents=True)
        (failed / "chromadb" / "chroma.sqlite3").write_bytes(b"sqlite")
        write_run_identity(failed, {
            "video": str(video.resolve()),
            "video_size": video.stat().st_size,
            "source_name": video.name,
            "collection": "screenlens_clip",
            "status": "error",
            "error": "Validation error: name: ...",
        })
        assert ingest_succeeded(failed) is False
        assert find_successful_run(data, video, "ingest") is None

    def test_reusable_run_finds_renamed_folder(self, tmp_path):
        video = _touch_video(tmp_path / "clip.mov")
        data = tmp_path / "data"
        named = data / "policy_doc_20260817_090000"
        (named / "frames").mkdir(parents=True)
        (named / "frames" / "frames_meta.json").write_text(
            '{"video": "%s", "video_size": %s}' % (video.resolve(), video.stat().st_size),
            encoding="utf-8",
        )
        config = ScreenLensConfig()
        config.data_dir = data
        assert find_reusable_run(config, video, "frames/frames_meta.json") == named


class TestSafeStem:
    def test_macos_screen_recording_becomes_clock_stamp(self, tmp_path):
        video = _touch_video(
            tmp_path / "Screen Recording 2026-08-17 at 9.29.20\u202fAM.mov"
        )
        assert safe_video_stem(video) == "20260817_092920"
        name = chroma_collection_name(safe_video_stem(video))
        assert name == "screenlens_20260817_092920"

        config = ScreenLensConfig()
        config.data_dir = tmp_path / "data"
        slug = apply_video_slug(config, video)
        assert slug.startswith("20260817_092920_")
        assert config.vector_db.collection_name == "screenlens_20260817_092920"

    def test_reuse_replaces_illegal_stored_collection(self, tmp_path):
        video = _touch_video(
            tmp_path / "Screen Recording 2026-08-17 at 9.29.20\u202fAM.mov"
        )
        run = tmp_path / "data" / "old_run"
        run.mkdir(parents=True)
        write_run_identity(run, {
            "video": str(video.resolve()),
            "collection": "screenlens_Screen_Recording_2026-08-17_at_9.29.20\u202fAM",
        })
        config = ScreenLensConfig()
        reuse_video_run(config, video, run)
        assert config.vector_db.collection_name == "screenlens_20260817_092920"


class TestContentName:
    def test_slugify_and_heading_heuristic(self):
        assert slugify_title("MRM Guidelines!") == "mrm_guidelines"
        text = "# Existing Investment Dashboard\n\nA table of holdings."
        assert infer_content_slug(text) == "existing_investment_dashboard"

    def test_generate_callback_wins_then_falls_back(self):
        text = "# Something\nvisible in the frame"
        assert infer_content_slug(text, generate=lambda s, u: "Policy Review Walkthrough") == (
            "policy_review_walkthrough"
        )
        assert infer_content_slug(text, generate=lambda s, u: "screen_recording") == (
            "something"
        )
        assert infer_content_slug(text, generate=lambda s, u: (_ for _ in ()).throw(RuntimeError("down"))) == (
            "something"
        )

    def test_rename_uses_recording_clock(self, tmp_path):
        video = _touch_video(tmp_path / "Screen Recording 2026-08-17 at 9.29.20 AM.mov")
        config = ScreenLensConfig()
        config.data_dir = tmp_path / "data"
        apply_video_slug(config, video)
        original = Path(config.data_dir)
        assert original.is_dir()
        (original / "captions").mkdir()
        (original / "captions" / "all_captions.json").write_text(
            '[{"caption": "# MRM Guidelines"}]', encoding="utf-8"
        )
        slug = apply_content_name(config, video, "mrm_guidelines")
        assert slug == "mrm_guidelines_20260817_092920"
        assert not original.exists()
        assert (tmp_path / "data" / slug).is_dir()
        assert run_belongs_to_video(tmp_path / "data" / slug, video)
