"""
Corpus Assembly Pipeline — LangGraph.

After ``reconstruct`` produces per-folder artifacts under ``data/*/output/``,
this pipeline detects whether the corpus represents a coherent coding project,
infers the original source-tree path of every artifact via batched LLM
sub-agents, validates the assembled tree (import resolution, orphans,
structure), and writes a single unified project tree to
``OUTPUT/<timestamp>/``.

Architecture (mirrors reconstruct.py):
  1. Discover         — walk data/*/output/, load meta + content snippets (no LLM)
  2. Gate             — is this a coding project?               (1 LLM call)
  3. Classify Corpus  — what are the project root directories?  (1 LLM call)
  4. Plan Paths       — partition into batches for sub-agents   (no LLM)
  5. Infer Workers    — sequential Send fan-out, batched inference (N LLM calls)
  6. Cluster          — group by inferred root, detect collisions (no LLM)
  7. QA Reflect       — mechanical (AST imports, orphans) + LLM judgment
  8. Materialize      — write OUTPUT/<timestamp>/ + MANIFEST + REPORT (no LLM)

The gate-fail and qa-fail paths exit gracefully without writing OUTPUT.
The pipeline reuses ``reconstruct.get_inference_client``, ``reconstruct.generate_text``,
and ``reconstruct.parse_json_response``, so a single Python process running
``reconstruct --assemble`` loads the 122B client only once.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.config import ScreenLensConfig
from src.workflows.reconstruct import get_inference_client, generate_text, parse_json_response

logger = logging.getLogger("screenlens.assemble")


# ── Constants ────────────────────────────────────────────────────────────────

MAX_QA_ITERATIONS = 3
INFERENCE_BATCH_SIZE = 10
SNIPPET_CHARS = 400


# ── System Prompts ───────────────────────────────────────────────────────────

GATE_SYSTEM = (
    "You are evaluating whether a corpus of reconstructed screen-recording artifacts "
    "represents a coherent coding project that should be assembled into a single "
    "source tree.\n\n"
    "INPUT: a list of folder slugs + their reconstructed content types + brief "
    "descriptions.\n\n"
    "A 'coding project' means: the artifacts come from one or more related codebases — "
    "shared package roots, cross-references, project files (pyproject.toml, "
    "requirements.txt, .env), or naming conventions suggesting a directory tree. "
    "Even a corpus that mixes Python files with READMEs, YAML configs, and shell "
    "scripts qualifies if those files plausibly belong to the same project(s).\n\n"
    "It is NOT a coding project if the artifacts are: a single non-code document, a "
    "GUI walkthrough, unrelated PDFs, or a random grab-bag of files with no shared "
    "structure.\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"is_code_project": true/false, "reasoning": "brief explanation", '
    '"estimated_root_count": N}'
)

CLASSIFY_CORPUS_SYSTEM = (
    "You are detecting project root directories in a corpus of reconstructed "
    "artifacts. The downstream path-inference step will use your roots as the "
    "valid choices for where each file belongs.\n\n"
    "INPUT: folder slugs + brief descriptions.\n\n"
    "A 'root' is the top directory of a self-contained project (e.g. "
    "'my-package', 'my-package-dashboard' if separate). It is COMMON for a corpus "
    "to contain multiple roots — a backend package, a separate dashboard, a tests "
    "directory, etc. Use the empty string '' to denote 'project root' (files at the "
    "top level, like .env, pyproject.toml).\n\n"
    "Heuristics:\n"
    "- Slugs prefixed with the same package-like name often share a root\n"
    '- "standalone_*" slugs typically belong inside a "*-api/src/standalone_*/" tree\n'
    '- "src_*", "tests_*", "seed_*", "scripts_*" are intra-project paths, NOT roots\n'
    "- Different naming styles (e.g. snake_case vs kebab-case top dirs) often signal "
    "separate projects\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"roots": ["root1", "root2", ...], "confidence": 0.0-1.0, '
    '"reasoning": "brief explanation"}'
)


QA_ASSEMBLY_SYSTEM = (
    "You are reviewing an assembled project tree for structural coherence.\n\n"
    "INPUT: the proposed file tree (as a flat list of paths) plus mechanical "
    "findings — unresolved local imports, orphan files, and root-distribution "
    "anomalies.\n\n"
    "Decide whether the assembly is acceptable or whether path inference should "
    "be retried with feedback.\n\n"
    "PASS criteria:\n"
    "- Local imports either resolve OR clearly reference an absent third-party "
    "name (not the project's own modules)\n"
    "- Orphans are limited to plausible entry points (main.py, __main__.py, "
    "app.py, cli.py, conftest.py, test_*.py, run_*.py)\n"
    "- The directory hierarchy is internally consistent — files that share a "
    "package prefix live under the same root\n\n"
    "Severity levels:\n"
    '- "ok": the assembly is fine, materialize as-is\n'
    '- "minor": small issues, materialize but flag in the report\n'
    '- "critical": structural drift (wrong root, missing parent dir, broken '
    "package layout) — RETRY path inference with feedback\n\n"
    "When severity is critical, populate specific_remappings with concrete fixes "
    "the next inference pass should apply. Each entry: {folder, from, to}.\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    "{\n"
    '  "passed": true/false,\n'
    '  "severity": "ok|minor|critical",\n'
    '  "feedback": "specific issues to address if retrying",\n'
    '  "specific_remappings": [{"folder": "...", "from": "...", "to": "..."}]\n'
    "}"
)


INFER_PATHS_SYSTEM = (
    "You are inferring the original source-tree paths for reconstructed "
    "video-recording artifacts in a coding project. Each input record gives "
    "you:\n"
    "  - folder: the recording slug, often encoding the original path\n"
    "  - description: a one-line summary of what the file contains\n"
    "  - snippet: the first ~400 chars of the reconstructed file\n\n"
    "Use ALL THREE signals — slug structure, description, and content snippet — "
    "to infer the most likely original path relative to the project root.\n\n"
    "CONTEXT (hint, not a hard constraint): the corpus may have these top-level "
    "directories: {roots}. You MAY also propose paths under directories not in "
    "this list if the slug + content clearly indicate them — the classifier is "
    "best-effort and may have missed sub-roots.\n\n"
    "Heuristics:\n"
    "1. Underscores between known directory tokens (src, tests, seed, services, "
    "domain, orchestration, scripts, api, docs, database) are usually path "
    "separators. So 'src_services_main.py' → 'src/services/main.py'.\n"
    "2. 'dot.X' → '.X' (hidden file). e.g. 'dot.env' → '.env'.\n"
    '3. "stanadalone_" / "src_servies_" are common typos — treat as standalone_/'
    "src_services_ respectively.\n"
    '4. "standalone_*" files are part of a "standalone_graph_api" package — likely '
    'path "asr-graph-compliance-api/src/standalone_graph_api/<basename>". Verify '
    "with content (imports, docstrings).\n"
    "5. When the slug is generic (document.md, app.py, tree), use the snippet's "
    "imports / headers / first lines to decide where it belongs.\n"
    "6. If the file content has a 'File: ...' header docstring, TRUST it.\n"
    "7. Two artifacts must NOT map to the same destination path. If two slugs "
    "look like they want the same path, distinguish them by content (e.g. one is "
    "'__main__.py' if it imports .main, the other is 'main.py').\n"
    "8. Be decisive. Mark confidence 'low' if you're guessing, but still pick a "
    "path.\n\n"
    "Respond with ONLY a valid JSON ARRAY (no markdown fences). One object per "
    "input record, in the SAME ORDER as the input:\n"
    "[\n"
    '  {"folder": "<from input>", "src_rel": "<from input>", '
    '"dst_rel": "<your inferred path>", "confidence": "high|medium|low", '
    '"reasoning": "<one short sentence>"},\n'
    "  ...\n"
    "]"
)


# ── State ────────────────────────────────────────────────────────────────────

class AssembleState(TypedDict, total=False):
    # Input
    data_dir: str
    output_dir: str
    config: dict
    mapping_override: Optional[str]
    dry_run: bool

    # Discovery
    artifacts: list[dict]   # {folder, src_rel, content_type, description, snippet, size, qa_scores}

    # Gate
    is_code_project: bool
    gate_reasoning: str
    estimated_root_count: int

    # Corpus classification
    project_roots: list[str]
    roots_confidence: float
    roots_reasoning: str

    # Path inference (single sequential node — no reducer needed because there
    # is only one writer).
    inference_batches: list[list[dict]]
    path_mappings: list[dict]

    # Cluster
    clusters: dict
    collisions: list[str]

    # QA
    qa_findings: dict
    qa_passed: bool
    qa_feedback: str
    qa_iteration: int

    # Output
    timestamp: str
    materialized_files: list[str]

    # Bookkeeping
    stage: str
    elapsed_seconds: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_snippet(path: Path, max_chars: int = SNIPPET_CHARS) -> str:
    try:
        return path.read_text(errors="replace")[:max_chars]
    except Exception as e:
        return f"<read error: {e}>"


def _format_artifact_summary(artifacts: list[dict], max_per_line: int = 100) -> str:
    """Compact one-line-per-artifact summary for use in LLM prompts."""
    lines = []
    for a in artifacts:
        desc = (a.get("description") or "").strip().replace("\n", " ")[:max_per_line]
        ct = a.get("content_type", "?")
        lines.append(f"- {a['folder']}  [{ct}]  {desc}")
    return "\n".join(lines)


# ── Pipeline Nodes ───────────────────────────────────────────────────────────

def discover_node(state: AssembleState) -> dict:
    """Walk data/*/output/ and load all reconstructed artifacts + their meta."""
    t0 = time.time()
    data_dir = Path(state["data_dir"])

    print(f"\n{'='*60}")
    print(f"[1/8] DISCOVERING ARTIFACTS")
    print(f"      Scanning {data_dir}/")
    print(f"{'='*60}")

    if not data_dir.is_dir():
        print(f"  ERROR: {data_dir}/ not found")
        return {"artifacts": [], "stage": "error",
                "elapsed_seconds": {"discover": 0.0}}

    artifacts: list[dict] = []
    folders_seen = 0
    folders_with_meta = 0

    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        out = sub / "output"
        if not out.is_dir():
            continue
        folders_seen += 1

        meta_path = out / "reconstruction_meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                folders_with_meta += 1
            except Exception as e:
                logger.warning(f"Failed to parse {meta_path}: {e}")

        # Per-artifact descriptions live in meta["artifacts"], keyed by filename
        meta_artifacts = {a["filename"]: a for a in meta.get("artifacts", [])}

        for f in sorted(out.rglob("*")):
            if not f.is_file() or f.name == "reconstruction_meta.json":
                continue
            rel = str(f.relative_to(out))
            meta_entry = meta_artifacts.get(rel, {})
            artifacts.append({
                "folder": sub.name,
                "src_rel": rel,
                "src_abs": str(f),
                "content_type": meta.get("content_type", "unknown"),
                "classification_confidence": meta.get("classification_confidence", 0.0),
                "description": meta_entry.get("description", ""),
                "snippet": _read_snippet(f),
                "size": f.stat().st_size,
                "qa_scores": meta.get("qa_scores", {}),
            })

    elapsed = time.time() - t0
    print(f"\n  Folders scanned: {folders_seen}")
    print(f"  With meta.json:  {folders_with_meta}")
    print(f"  Total artifacts: {len(artifacts)}")
    print(f"  Discovered in {elapsed:.1f}s")

    return {
        "artifacts": artifacts,
        "stage": "discovered",
        "elapsed_seconds": {"discover": round(elapsed, 2)},
    }


def gate_node(state: AssembleState) -> dict:
    """Decide whether the corpus represents a coherent coding project."""
    t0 = time.time()
    artifacts = state["artifacts"]

    print(f"\n{'='*60}")
    print(f"[2/8] GATE — IS THIS A CODING PROJECT?")
    print(f"      Evaluating {len(artifacts)} artifact(s)")
    print(f"{'='*60}")

    if not artifacts:
        print("  No artifacts found — gate skipped")
        return {
            "is_code_project": False,
            "gate_reasoning": "No artifacts discovered",
            "estimated_root_count": 0,
            "stage": "gated",
            "elapsed_seconds": {"gate": 0.0},
        }

    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)

    # Aggregate content type distribution
    type_counts: dict[str, int] = {}
    for a in artifacts:
        type_counts[a["content_type"]] = type_counts.get(a["content_type"], 0) + 1
    type_summary = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))

    user_prompt = (
        f"Corpus of {len(artifacts)} reconstructed artifacts.\n"
        f"Content type distribution: {type_summary}\n\n"
        f"Artifacts (folder slug + content_type + description):\n\n"
        f"{_format_artifact_summary(artifacts)}\n\n"
        "Is this a coherent coding project that should be assembled into a unified "
        "source tree?"
    )

    response = generate_text(client, GATE_SYSTEM, user_prompt,
                            max_tokens=512, temperature=0.1)
    result = parse_json_response(response)

    is_code = bool(result.get("is_code_project", False))
    reasoning = result.get("reasoning", "No reasoning provided")
    root_hint = int(result.get("estimated_root_count", 1))

    elapsed = time.time() - t0
    print(f"\n  Decision:        {'YES — coding project' if is_code else 'NO'}")
    print(f"  Estimated roots: {root_hint}")
    print(f"  Reasoning:       {reasoning}")
    print(f"  Gated in {elapsed:.1f}s")

    return {
        "is_code_project": is_code,
        "gate_reasoning": reasoning,
        "estimated_root_count": root_hint,
        "stage": "gated",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "gate": round(elapsed, 2)},
    }


def classify_corpus_node(state: AssembleState) -> dict:
    """Detect project root directories from the corpus."""
    t0 = time.time()
    artifacts = state["artifacts"]
    root_hint = state.get("estimated_root_count", 1)

    print(f"\n{'='*60}")
    print(f"[3/8] CLASSIFY CORPUS — DETECT PROJECT ROOTS")
    print(f"      Hint from gate: ~{root_hint} root(s)")
    print(f"{'='*60}")

    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)

    user_prompt = (
        f"Corpus of {len(artifacts)} artifacts. The previous gate step estimated "
        f"~{root_hint} project root(s).\n\n"
        f"Artifacts (folder slug + content_type + description):\n\n"
        f"{_format_artifact_summary(artifacts)}\n\n"
        "Identify the top-level project root directories. Return them as a list — "
        "use '' for 'top of project'."
    )

    response = generate_text(client, CLASSIFY_CORPUS_SYSTEM, user_prompt,
                            max_tokens=512, temperature=0.1)
    result = parse_json_response(response)

    roots = result.get("roots", [])
    if not isinstance(roots, list):
        roots = []
    confidence = float(result.get("confidence", 0.0))
    reasoning = result.get("reasoning", "No reasoning provided")

    elapsed = time.time() - t0
    print(f"\n  Detected roots ({len(roots)}):")
    for r in roots:
        display = "(project top)" if r == "" else r
        print(f"    - {display}")
    print(f"  Confidence:  {confidence:.0%}")
    print(f"  Reasoning:   {reasoning}")
    print(f"  Classified in {elapsed:.1f}s")

    return {
        "project_roots": roots,
        "roots_confidence": confidence,
        "roots_reasoning": reasoning,
        "stage": "classified",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "classify": round(elapsed, 2)},
    }


def plan_paths_node(state: AssembleState) -> dict:
    """Partition the artifact list into batches for inference workers.

    Each batch becomes one ``Send`` payload dispatched to ``infer_paths_worker``.
    On retry (qa_iteration > 0), the qa_feedback is attached to each batch so
    workers can incorporate it.
    """
    t0 = time.time()
    artifacts = state["artifacts"]
    qa_iteration = state.get("qa_iteration", 0)
    qa_feedback = state.get("qa_feedback", "")

    print(f"\n{'='*60}")
    print(f"[4/8] PLAN PATHS — PARTITION INTO INFERENCE BATCHES")
    if qa_iteration > 0:
        print(f"      Retry #{qa_iteration} — incorporating QA feedback")
    print(f"{'='*60}")

    batches = []
    for i in range(0, len(artifacts), INFERENCE_BATCH_SIZE):
        batches.append(artifacts[i:i + INFERENCE_BATCH_SIZE])

    elapsed = time.time() - t0
    print(f"  {len(artifacts)} artifact(s) → {len(batches)} batch(es) of "
          f"≤{INFERENCE_BATCH_SIZE}")
    print(f"  Planned in {elapsed:.1f}s")

    return {
        "inference_batches": batches,
        "stage": "planned",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "plan": round(elapsed, 2)},
    }


def _parse_inference_response(parsed, batch: list[dict]) -> list[dict]:
    """Normalize an LLM inference response into a clean list of mapping dicts."""
    if isinstance(parsed, list):
        raw = parsed
    elif isinstance(parsed, dict):
        raw = parsed.get("mappings") or parsed.get("results") or []
    else:
        raw = []

    cleaned: list[dict] = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            continue
        # Backfill folder/src_rel from input order if the client omits them
        if i < len(batch):
            m.setdefault("folder", batch[i]["folder"])
            m.setdefault("src_rel", batch[i]["src_rel"])
        if "dst_rel" not in m:
            continue
        cleaned.append({
            "folder": m["folder"],
            "src_rel": m["src_rel"],
            "dst_rel": m["dst_rel"],
            "confidence": m.get("confidence", "low"),
            "reasoning": m.get("reasoning", ""),
        })
    return cleaned


def infer_paths_sequential(state: AssembleState) -> dict:
    """Loop over inference batches sequentially, one LLM call per batch.

    The whole loop runs inside this single graph node so the path_mappings
    list is built in-process and returned as a single state update — no
    reducer needed.
    """
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    batches: list[list[dict]] = state.get("inference_batches", [])
    project_roots: list[str] = state.get("project_roots", [])
    qa_feedback: str = state.get("qa_feedback", "")

    client = get_inference_client(config)

    print(f"\n{'='*60}")
    print(f"[5/8] INFER PATHS — SEQUENTIAL ({len(batches)} batch(es))")
    print(f"{'='*60}")

    roots_display = ", ".join(repr(r) for r in project_roots) if project_roots else "(none detected)"
    # Use .replace() rather than .format() — the prompt template contains
    # literal JSON examples with braces that would otherwise be interpreted
    # as format placeholders.
    system_prompt = INFER_PATHS_SYSTEM.replace("{roots}", roots_display)

    all_mappings: list[dict] = []

    for batch_index, batch in enumerate(batches, 1):
        bt0 = time.time()
        print(f"\n  [Batch {batch_index}/{len(batches)}] Inferring paths for "
              f"{len(batch)} artifact(s)...")

        records_text = []
        for a in batch:
            snippet = a.get("snippet", "")[:SNIPPET_CHARS]
            records_text.append(
                f"---\n"
                f"folder: {a['folder']}\n"
                f"src_rel: {a['src_rel']}\n"
                f"description: {(a.get('description') or '').strip()}\n"
                f"snippet:\n{snippet}\n"
            )

        user_prompt = (
            f"Infer the original source-tree paths for the following "
            f"{len(batch)} artifact(s):\n\n"
            + "\n".join(records_text)
        )
        if qa_feedback:
            user_prompt += (
                f"\n\nPREVIOUS QA FEEDBACK (incorporate this in your decisions):\n"
                f"{qa_feedback}\n"
            )

        response = generate_text(
            client, system_prompt, user_prompt,
            max_tokens=2048, temperature=0.1,
        )
        parsed = parse_json_response(response)
        cleaned = _parse_inference_response(parsed, batch)
        all_mappings.extend(cleaned)

        bt = time.time() - bt0
        print(f"  [Batch {batch_index}/{len(batches)}] {len(cleaned)} mapping(s) "
              f"in {bt:.1f}s")
        if len(cleaned) < len(batch):
            missing = len(batch) - len(cleaned)
            print(f"  [Batch {batch_index}/{len(batches)}] WARNING: {missing} "
                  f"artifact(s) returned no usable mapping")

    elapsed = time.time() - t0
    print(f"\n  Total: {len(all_mappings)} mapping(s) across {len(batches)} batch(es) "
          f"in {elapsed:.1f}s")

    return {
        "path_mappings": all_mappings,
        "stage": "inferred",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "infer": round(elapsed, 2)},
    }


def cluster_node(state: AssembleState) -> dict:
    """Group inferred mappings by their top-level directory and detect collisions."""
    t0 = time.time()
    mappings = state.get("path_mappings", [])

    print(f"\n{'='*60}")
    print(f"[6/8] CLUSTER — GROUP BY ROOT, DETECT COLLISIONS")
    print(f"      Reviewing {len(mappings)} mapping(s)")
    print(f"{'='*60}")

    clusters: dict[str, list[dict]] = {}
    for m in mappings:
        dst = m["dst_rel"].lstrip("/")
        # First path segment is the cluster key. Files at root (e.g. ".env",
        # "pyproject.toml") cluster under "" — printed as "(root)".
        head, _, _ = dst.partition("/")
        if "/" not in dst:
            head = ""
        clusters.setdefault(head, []).append(m)

    seen_dst: dict[str, list[str]] = {}
    for m in mappings:
        seen_dst.setdefault(m["dst_rel"], []).append(m["folder"])
    collisions = [path for path, folders in seen_dst.items() if len(folders) > 1]

    elapsed = time.time() - t0
    print(f"\n  Clusters detected: {len(clusters)}")
    for root in sorted(clusters.keys()):
        display = "(root)" if root == "" else root
        print(f"    - {display}: {len(clusters[root])} file(s)")
    if collisions:
        print(f"\n  COLLISIONS ({len(collisions)}):")
        for c in collisions:
            print(f"    - {c}  ← {seen_dst[c]}")
    else:
        print(f"  No collisions ✓")

    print(f"  Clustered in {elapsed:.1f}s")

    return {
        "clusters": clusters,
        "collisions": collisions,
        "stage": "clustered",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "cluster": round(elapsed, 2)},
    }


# ── Stub nodes (filled in step 4+) ───────────────────────────────────────────


ENTRYPOINT_PATTERNS = (
    "main.py", "__main__.py", "app.py", "cli.py", "conftest.py",
    "run.py", "manage.py", "setup.py", "wsgi.py", "asgi.py",
)


def _is_entrypoint(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if name in ENTRYPOINT_PATTERNS or name.startswith("test_") or name.startswith("run_"):
        return True
    return False


def _extract_python_imports(source: str) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Return (absolute_imports, from_imports) where from_imports is [(module, name), ...].
    Falls back to regex if AST parsing fails (partial reconstructions are possible)."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex extraction — better than nothing on broken files
        absolute = re.findall(r"^\s*import\s+([\w.]+)", source, flags=re.MULTILINE)
        from_lines = re.findall(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$", source, flags=re.MULTILINE)
        from_imports = []
        for mod, names in from_lines:
            for n in names.split(","):
                from_imports.append((mod, n.strip().split(" as ")[0]))
        return absolute, from_imports

    absolute_imports: list[str] = []
    from_imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                absolute_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                # relative import like "from . import x" — module unknown without context
                continue
            # Skip relative imports (level > 0) — they're contextual and AST-resolvable
            # only with the package layout. We'll detect them via the prefix instead.
            if node.level > 0:
                continue
            for alias in node.names:
                from_imports.append((node.module, alias.name))
    return absolute_imports, from_imports


# A small stdlib allow-list for the common cases. We don't need to be exhaustive
# — anything that's NOT in the assembled tree and NOT in this list is treated as
# third-party and ignored. Only "looks like project module but doesn't resolve"
# triggers a finding.
_STDLIB_PREFIXES = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "glob", "hashlib", "heapq", "http", "importlib", "inspect",
    "io", "itertools", "json", "logging", "math", "operator", "os", "pathlib",
    "pickle", "platform", "queue", "random", "re", "shlex", "shutil", "signal",
    "socket", "sqlite3", "string", "struct", "subprocess", "sys", "tempfile",
    "textwrap", "threading", "time", "traceback", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "warnings", "weakref", "xml", "yaml", "zipfile",
    "__future__",
}


def _qa_mechanical_check(mappings: list[dict], artifacts_by_key: dict) -> dict:
    """Run AST-based import resolution + orphan detection on the assembled tree.

    Returns a findings dict with keys:
      - assembled_paths: sorted list of all dst_rel paths
      - unresolved_imports: [(file, module)] — local-looking imports that don't resolve
      - orphans: [path] — files no one imports and not entrypoint-shaped
      - root_distribution: {root: count}
    """
    # Build the set of assembled module paths. A file at "src/services/main.py"
    # corresponds to the importable name "src.services.main".
    assembled_paths = sorted(m["dst_rel"] for m in mappings)
    py_files = [m for m in mappings if m["dst_rel"].endswith(".py")]

    importable_names: set[str] = set()
    for m in py_files:
        path = m["dst_rel"]
        # Strip .py and convert / to .
        modname = path[:-3].replace("/", ".")
        importable_names.add(modname)
        # Also register parent packages (e.g. "src", "src.services")
        parts = modname.split(".")
        for i in range(1, len(parts)):
            importable_names.add(".".join(parts[:i]))
        # And bare-package import (drop "__init__")
        if parts[-1] == "__init__":
            importable_names.add(".".join(parts[:-1]))

    # Detect what each file imports and whether it resolves.
    unresolved: list[tuple[str, str]] = []
    imported_by: dict[str, set[str]] = {}  # module → set of files that import it
    for m in py_files:
        key = (m["folder"], m["src_rel"])
        artifact = artifacts_by_key.get(key)
        if not artifact:
            continue
        try:
            source = Path(artifact["src_abs"]).read_text(errors="replace")
        except Exception:
            continue

        absolute, from_imports = _extract_python_imports(source)
        all_modules = list(absolute) + [mod for mod, _ in from_imports]

        for mod in all_modules:
            top = mod.split(".")[0]
            # Skip stdlib
            if top in _STDLIB_PREFIXES:
                continue
            # If it resolves to something in the assembled tree, record it
            if mod in importable_names or any(name == mod or name.startswith(mod + ".") for name in importable_names):
                imported_by.setdefault(mod, set()).add(m["dst_rel"])
                continue
            # Looks like a local module if it shares a prefix with any assembled root
            project_roots = {p.split(".")[0] for p in importable_names}
            if top in project_roots:
                unresolved.append((m["dst_rel"], mod))

    # Detect orphans: files that nothing imports AND aren't entrypoint-shaped
    referenced_paths: set[str] = set()
    for paths in imported_by.values():
        referenced_paths.update(paths)
    orphans: list[str] = []
    for m in py_files:
        path = m["dst_rel"]
        if path in referenced_paths:
            continue
        if _is_entrypoint(path):
            continue
        # __init__.py is structurally important even if not imported by name
        if path.endswith("__init__.py"):
            continue
        orphans.append(path)

    # Root distribution
    root_dist: dict[str, int] = {}
    for path in assembled_paths:
        head, _, _ = path.partition("/")
        if "/" not in path:
            head = "(root)"
        root_dist[head] = root_dist.get(head, 0) + 1

    return {
        "total_files": len(assembled_paths),
        "python_files": len(py_files),
        "unresolved_imports": unresolved,
        "orphans": orphans,
        "root_distribution": root_dist,
    }


def qa_reflect_node(state: AssembleState) -> dict:
    """Validate the assembled tree: mechanical checks + LLM judgment.

    On critical severity, sets qa_passed=False so the graph loops back to
    plan_paths_node with qa_feedback for a retry. On ok or minor, sets
    qa_passed=True so the graph proceeds to materialize.
    """
    t0 = time.time()
    qa_iteration = state.get("qa_iteration", 0)
    mappings = state.get("path_mappings", [])
    artifacts = state.get("artifacts", [])

    print(f"\n{'='*60}")
    print(f"[7/8] QA REFLECT — iteration {qa_iteration + 1}/{MAX_QA_ITERATIONS}")
    print(f"      Validating {len(mappings)} mapping(s)")
    print(f"{'='*60}")

    artifacts_by_key = {(a["folder"], a["src_rel"]): a for a in artifacts}

    # Phase 1: mechanical
    findings = _qa_mechanical_check(mappings, artifacts_by_key)
    print(f"\n  Mechanical findings:")
    print(f"    Files:               {findings['total_files']} ({findings['python_files']} Python)")
    print(f"    Unresolved imports:  {len(findings['unresolved_imports'])}")
    print(f"    Orphans:             {len(findings['orphans'])}")
    print(f"    Root distribution:   {findings['root_distribution']}")
    if findings["unresolved_imports"]:
        print(f"    Sample unresolved:")
        for f, mod in findings["unresolved_imports"][:5]:
            print(f"      {f} → {mod}")
    if findings["orphans"]:
        print(f"    Sample orphans:")
        for o in findings["orphans"][:5]:
            print(f"      {o}")

    # Phase 2: LLM judgment.
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)

    tree_lines = "\n".join(f"  {p}" for p in sorted(m["dst_rel"] for m in mappings))
    unresolved_text = "\n".join(f"  - {f}: imports '{mod}' (not in tree)"
                                  for f, mod in findings["unresolved_imports"][:30]) or "  (none)"
    orphans_text = "\n".join(f"  - {o}" for o in findings["orphans"][:20]) or "  (none)"

    user_prompt = (
        f"Proposed assembled tree ({findings['total_files']} files):\n"
        f"{tree_lines}\n\n"
        f"Root distribution: {findings['root_distribution']}\n\n"
        f"Unresolved local imports ({len(findings['unresolved_imports'])} total, "
        f"showing up to 30):\n{unresolved_text}\n\n"
        f"Orphans ({len(findings['orphans'])} total, showing up to 20):\n{orphans_text}\n\n"
        "Is this assembly acceptable? Decide pass/severity/feedback per the system prompt."
    )

    response = generate_text(
        client, QA_ASSEMBLY_SYSTEM, user_prompt,
        max_tokens=1024, temperature=0.1,
    )
    result = parse_json_response(response)

    severity = result.get("severity", "minor")
    passed = bool(result.get("passed", severity != "critical"))
    feedback = result.get("feedback", "")
    remappings = result.get("specific_remappings", [])

    # Force-pass on the last iteration (always materialize what we have rather
    # than failing — the report will flag any open issues for manual fix).
    if qa_iteration + 1 >= MAX_QA_ITERATIONS and not passed:
        print(f"\n  Max QA iterations reached — proceeding to materialize with current mapping")
        passed = True
        severity = "minor"

    elapsed = time.time() - t0
    print(f"\n  Decision: {'PASS' if passed else 'RETRY'} (severity: {severity})")
    if feedback:
        print(f"  Feedback: {feedback[:200]}")
    if remappings and not passed:
        print(f"  Specific remappings ({len(remappings)}):")
        for r in remappings[:5]:
            print(f"    {r.get('folder')}: {r.get('from')} → {r.get('to')}")
    print(f"  QA in {elapsed:.1f}s")

    # Build the qa_feedback string injected into the next inference round
    qa_feedback_str = feedback
    if remappings:
        qa_feedback_str += "\n\nSpecific remappings to apply:\n" + "\n".join(
            f"- folder '{r.get('folder')}': change dst_rel from '{r.get('from')}' "
            f"to '{r.get('to')}'"
            for r in remappings
        )

    return {
        "qa_passed": passed,
        "qa_findings": findings,
        "qa_feedback": qa_feedback_str,
        "qa_iteration": qa_iteration + 1,
        "stage": "qa_reflected",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), f"qa_{qa_iteration + 1}": round(elapsed, 2)},
    }


def _build_assembly_report(state: AssembleState, timestamp: str) -> str:
    """Build the human-readable ASSEMBLY_REPORT.md content."""
    mappings = state.get("path_mappings", [])
    clusters = state.get("clusters", {})
    findings = state.get("qa_findings", {})
    qa_iters = state.get("qa_iteration", 0)

    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for m in mappings:
        conf_counts[m.get("confidence", "low")] = conf_counts.get(m.get("confidence", "low"), 0) + 1

    lines: list[str] = []
    lines.append(f"# Assembly Report — {timestamp}\n")
    lines.append(f"- **Source**: `{state.get('data_dir')}`")
    lines.append(f"- **Output**: `OUTPUT/{timestamp}/`")
    lines.append(f"- **Total artifacts assembled**: {len(mappings)}")
    lines.append(f"- **Confidence distribution**: high={conf_counts['high']}, "
                 f"medium={conf_counts['medium']}, low={conf_counts['low']}")
    lines.append(f"- **QA iterations used**: {qa_iters}")
    lines.append(f"- **QA passed**: {state.get('qa_passed', False)}")
    lines.append("")

    lines.append("## Cluster summary\n")
    for root in sorted(clusters.keys()):
        display = "(root)" if root == "" else root
        lines.append(f"- **{display}**: {len(clusters[root])} file(s)")
    lines.append("")

    if findings:
        lines.append("## QA findings\n")
        lines.append(f"- Unresolved imports: {len(findings.get('unresolved_imports', []))}")
        lines.append(f"- Orphans: {len(findings.get('orphans', []))} "
                     f"(note: relative imports are not analyzed, so this number "
                     f"is inflated by services that only import each other relatively)")
        if findings.get("unresolved_imports"):
            lines.append("\n### Sample unresolved imports\n")
            for f, mod in findings["unresolved_imports"][:10]:
                lines.append(f"- `{f}` → `{mod}`")
        lines.append("")

    flagged = [m for m in mappings if m.get("confidence") != "high"]
    if flagged:
        lines.append("## Mappings to review (non-high confidence)\n")
        for m in flagged:
            lines.append(f"- **{m['confidence']}** — `{m['folder']}` → `{m['dst_rel']}`")
            if m.get("reasoning"):
                lines.append(f"  - {m['reasoning']}")
        lines.append("")

    lines.append("## Full mapping\n")
    lines.append("See `MANIFEST.json` for the complete source-folder → "
                 "destination-path mapping. The MANIFEST is structurally "
                 "identical to a `ratita_mapping.json` and can be hand-edited "
                 "and re-applied via `python -m src.cli assemble --mapping <path>`.")

    return "\n".join(lines) + "\n"


def materialize_node(state: AssembleState) -> dict:
    """Write the assembled tree to OUTPUT/<timestamp>/.

    Always creates parent directories before write_text (the lesson from
    reconstruct.save_node — nested dst_rel paths like
    'asr-graph-compliance-api/src/standalone_graph_api/models.py' need
    intermediate dirs created or write_text crashes).
    """
    t0 = time.time()
    output_root = Path(state["output_dir"])
    timestamp = state.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = output_root / timestamp

    mappings = state.get("path_mappings", [])
    artifacts = state.get("artifacts", [])
    artifacts_by_key = {(a["folder"], a["src_rel"]): a for a in artifacts}

    print(f"\n{'='*60}")
    print(f"[8/8] MATERIALIZE — WRITE ASSEMBLED TREE")
    print(f"      Target: {target_dir}")
    print(f"      Files:  {len(mappings)}")
    print(f"{'='*60}")

    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped_missing: list[str] = []
    skipped_collision: list[str] = []
    seen_dst: set[str] = set()
    import shutil

    for m in mappings:
        key = (m["folder"], m["src_rel"])
        artifact = artifacts_by_key.get(key)
        if not artifact:
            skipped_missing.append(f"{m['folder']}/{m['src_rel']}")
            continue
        src_abs = Path(artifact["src_abs"])
        if not src_abs.is_file():
            skipped_missing.append(str(src_abs))
            continue
        dst_rel = m["dst_rel"].lstrip("/")
        if dst_rel in seen_dst:
            # Last writer wins, but log it
            skipped_collision.append(dst_rel)
        seen_dst.add(dst_rel)
        dst_abs = target_dir / dst_rel
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_abs, dst_abs)
        written.append(str(dst_abs))

    # MANIFEST.json — full mapping in the same shape as ratita_mapping.json,
    # so it can be hand-edited and re-applied via --mapping.
    manifest_path = target_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(mappings, indent=2))

    # ASSEMBLY_REPORT.md — human-readable summary.
    report_path = target_dir / "ASSEMBLY_REPORT.md"
    report_path.write_text(_build_assembly_report(state, timestamp))

    elapsed = time.time() - t0
    print(f"\n  Wrote {len(written)} file(s) in {elapsed:.1f}s")
    if skipped_missing:
        print(f"  Skipped (missing source): {len(skipped_missing)}")
    if skipped_collision:
        print(f"  Collisions (last writer won): {len(skipped_collision)}")
        for c in skipped_collision[:5]:
            print(f"    - {c}")
    print(f"  MANIFEST: {manifest_path}")
    print(f"  REPORT:   {report_path}")

    return {
        "materialized_files": written,
        "stage": "materialized",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "materialize": round(elapsed, 2)},
    }


def end_with_explanation_node(state: AssembleState) -> dict:
    """Terminal node when the gate decides this isn't a coding project."""
    print(f"\n{'='*60}")
    print(f"GATE FAILED — pipeline halted")
    print(f"{'='*60}")
    print(f"  Reasoning: {state.get('gate_reasoning', '?')}")
    return {"stage": "gate_failed"}


def load_override_mapping_node(state: AssembleState) -> dict:
    """Load a hand-edited MANIFEST.json from --mapping and skip LLM inference.

    Sets path_mappings directly. Cluster + QA still run on the loaded mapping
    so the user gets validation feedback even on a manually-curated mapping.
    """
    t0 = time.time()
    mapping_path = state.get("mapping_override")
    print(f"\n{'='*60}")
    print(f"[BYPASS] LOADING MAPPING OVERRIDE")
    print(f"      Source: {mapping_path}")
    print(f"{'='*60}")

    if not mapping_path or not Path(mapping_path).is_file():
        print(f"  ERROR: mapping file not found: {mapping_path}")
        return {"path_mappings": [], "stage": "override_failed",
                "elapsed_seconds": {**state.get("elapsed_seconds", {}), "override": 0.0}}

    try:
        loaded = json.loads(Path(mapping_path).read_text())
    except Exception as e:
        print(f"  ERROR: failed to parse mapping JSON: {e}")
        return {"path_mappings": [], "stage": "override_failed"}

    if not isinstance(loaded, list):
        print(f"  ERROR: mapping must be a JSON array, got {type(loaded).__name__}")
        return {"path_mappings": [], "stage": "override_failed"}

    cleaned = []
    for entry in loaded:
        if not isinstance(entry, dict):
            continue
        if "folder" not in entry or "src_rel" not in entry or "dst_rel" not in entry:
            continue
        cleaned.append({
            "folder": entry["folder"],
            "src_rel": entry["src_rel"],
            "dst_rel": entry["dst_rel"],
            "confidence": entry.get("confidence", "manual"),
            "reasoning": entry.get("reasoning", "from override mapping"),
        })

    elapsed = time.time() - t0
    print(f"  Loaded {len(cleaned)} mapping(s) in {elapsed:.1f}s")

    return {
        "path_mappings": cleaned,
        # Skip the corpus-classification phase entirely — the user has done
        # the inference work. We still want cluster + QA to validate.
        "is_code_project": True,
        "project_roots": [],
        "stage": "override_loaded",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "override": round(elapsed, 2)},
    }


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_discover(state: AssembleState) -> str:
    """If --mapping was provided, bypass all LLM inference and load it directly."""
    return "load_override_mapping" if state.get("mapping_override") else "gate"


def route_after_gate(state: AssembleState) -> str:
    return "classify_corpus" if state.get("is_code_project") else "end_with_explanation"


def route_after_qa(state: AssembleState) -> str:
    """After QA:
    - In --dry-run, always terminate (never write, never retry)
    - In --mapping override mode, never retry (the user supplied the mapping;
      retrying would discard it and re-run LLM inference)
    - Otherwise: retry on failure, materialize on pass
    """
    if state.get("dry_run"):
        return "end_dry_run"
    if state.get("mapping_override"):
        return "materialize"
    return "materialize" if state.get("qa_passed") else "plan_paths"


def end_dry_run_node(state: AssembleState) -> dict:
    """Terminal node for --dry-run mode. Prints summary and dumps the full
    mapping JSON to ./data/.assemble_dry_run.json for inspection."""
    print(f"\n{'='*60}")
    print(f"DRY RUN — stopping before materialize")
    print(f"{'='*60}")
    mappings = state.get("path_mappings", [])
    clusters = state.get("clusters", {})
    collisions = state.get("collisions", [])
    findings = state.get("qa_findings", {})

    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for m in mappings:
        c = m.get("confidence", "low")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    print(f"  Mappings produced: {len(mappings)}")
    print(f"  Confidence:        high={conf_counts['high']} medium={conf_counts['medium']} low={conf_counts['low']}")
    print(f"  Clusters:          {len(clusters)}")
    print(f"  Collisions:        {len(collisions)}")
    if findings:
        print(f"  QA passed:         {state.get('qa_passed')}")
        print(f"  Unresolved imports: {len(findings.get('unresolved_imports', []))}")
        print(f"  Orphans:            {len(findings.get('orphans', []))}")

    flagged = [m for m in mappings if m.get("confidence") != "high"]
    if flagged:
        print(f"\n  Non-high-confidence mappings ({len(flagged)}) — review:")
        for m in flagged:
            print(f"    [{m['confidence']:>6}] {m['folder'][:50]:<52} → {m['dst_rel']}")
            if m.get("reasoning"):
                print(f"             {m['reasoning'][:120]}")

    # Dump the full mapping to a sidecar file for inspection / diffing.
    # data/ is gitignored, so this won't pollute commits.
    dump_path = Path(state.get("data_dir", "./data")) / ".assemble_dry_run.json"
    try:
        dump_path.write_text(json.dumps(mappings, indent=2))
        print(f"\n  Full mapping dumped to: {dump_path}")
    except Exception as e:
        print(f"\n  Could not dump mapping: {e}")

    return {"stage": "dry_run_complete"}


# ── Graph Construction ───────────────────────────────────────────────────────

def build_assemble_graph():
    """Build the corpus assembly pipeline.

    Topology (step 2 — only the first three nodes do real work, rest are stubs):
        START → discover → gate → (classify_corpus | end_with_explanation)
                                       ↓
                           (end_dry_run | plan_paths → ... → materialize → END)
    """
    graph = StateGraph(AssembleState)

    graph.add_node("discover", discover_node)
    graph.add_node("load_override_mapping", load_override_mapping_node)
    graph.add_node("gate", gate_node)
    graph.add_node("classify_corpus", classify_corpus_node)
    graph.add_node("end_with_explanation", end_with_explanation_node)
    graph.add_node("end_dry_run", end_dry_run_node)
    graph.add_node("plan_paths", plan_paths_node)
    graph.add_node("infer_paths_sequential", infer_paths_sequential)
    graph.add_node("cluster", cluster_node)
    graph.add_node("qa_reflect", qa_reflect_node)
    graph.add_node("materialize", materialize_node)

    graph.add_edge(START, "discover")
    graph.add_conditional_edges(
        "discover", route_after_discover,
        {"load_override_mapping": "load_override_mapping", "gate": "gate"},
    )
    # Override path: load mapping → cluster (skip all LLM inference)
    graph.add_edge("load_override_mapping", "cluster")

    # Standard path: gate → classify → plan → infer → cluster
    graph.add_conditional_edges(
        "gate", route_after_gate,
        {"classify_corpus": "classify_corpus", "end_with_explanation": "end_with_explanation"},
    )
    graph.add_edge("classify_corpus", "plan_paths")
    graph.add_edge("plan_paths", "infer_paths_sequential")
    graph.add_edge("infer_paths_sequential", "cluster")
    graph.add_edge("cluster", "qa_reflect")
    graph.add_edge("end_with_explanation", END)
    graph.add_edge("end_dry_run", END)

    # Reflection loop: qa_reflect → plan_paths (retry), materialize (proceed),
    # or end_dry_run (--dry-run termination — runs through QA but never writes)
    graph.add_conditional_edges(
        "qa_reflect", route_after_qa,
        {"plan_paths": "plan_paths", "materialize": "materialize", "end_dry_run": "end_dry_run"},
    )
    # Stub: materialize → END will become real in step 5
    graph.add_edge("materialize", END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def assemble_corpus(
    data_dir: str,
    output_dir: str,
    config: ScreenLensConfig,
    mapping_override: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Run the assembly pipeline against an existing data/ directory."""
    pipeline = build_assemble_graph()
    initial_state: AssembleState = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "config": config.model_dump(),
        "mapping_override": mapping_override,
        "dry_run": dry_run,
        "path_mappings": [],
        "qa_iteration": 0,
        "elapsed_seconds": {},
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    return pipeline.invoke(initial_state)
