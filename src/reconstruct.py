"""
Artifact Reconstruction Pipeline — LangGraph Deep Agents.

Analyzes ingested video captions and reconstructs the original artifacts
(Python code, Markdown docs, PDFs, or GUI demo documentation) shown in recordings.

Architecture:
  1. Classify  — Determine content type from captions (code/doc/pdf/demo)
  2. Plan      — Generate tailored prompts + decompose into reconstruction tasks
  3. Execute   — Fan-out to parallel sub-agents via LangGraph Send (when safe)
                 OR sequential execution when ordering/coherence matters
  4. Reflect   — QA review with reflection agents (max 3 iterations)
  5. Save      — Write reconstructed artifacts to output folder

Uses LangGraph's Send API for conditional parallel dispatch and
Annotated reducers for collecting sub-agent outputs.
"""
import json
import logging
import operator
import re
import time
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .config import ScreenLensConfig
from .omlx_client import InferenceClient, InferenceTruncatedError
from .session import text_role_captioning_config
# Reuse the chunk-strategy math from the summarization pipeline. It's the same
# token-budget problem (fit a long caption stream into a fixed model context),
# so we deliberately share the helper rather than duplicate the constants.
from .pipeline import _chunk_captions_by_budget, _compute_chunk_strategy

logger = logging.getLogger("screenlens.reconstruct")


# ── Constants ────────────────────────────────────────────────────────────────

CONTENT_TYPES = {
    "python_code": "Python source code being written or edited in an IDE/editor",
    "markdown_document": "A Markdown or text document being authored or edited",
    "pdf_document": "A PDF document being viewed, reviewed, or presented",
    "gui_demo": "A GUI application walkthrough or demonstration",
}

MAX_QA_ITERATIONS = 3

# Fidelity cap in addition to the serialized-size budget. On a model with a
# very large context window, hundreds of short captions may technically fit,
# but one oversized extraction prompt would leave too little of that window
# for a detailed response and force the model to compress away specifics.
MAX_CAPTIONS_PER_CHUNK = 50

# Long-form calls may consume the client's full configured completion ceiling.
# Pass 2 separately reserves part of the context while planning its input, and
# intermediate condensation requests a smaller target while retaining the full
# server context as headroom; recursion is guarded by measured size reduction.
MIN_RECONSTRUCTION_OUTPUT_TOKENS = 256
MIN_SYNTHESIS_INPUT_TOKENS = 2048
SYNTHESIS_OUTPUT_RESERVE_RATIO = 0.25
SEGMENT_NOTE_TARGET_TOKENS = 1400
INTERMEDIATE_TARGET_RATIO = 0.3
INTERMEDIATE_TARGET_MAX_TOKENS = 4096
MAX_SYNTHESIS_DEPTH = 8
SYNTHESIS_OVERHEAD_TOKENS = 2548


# ── System Prompts ───────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = (
    "You are a content classifier for screen recordings. Based on frame-by-frame "
    "captions from a screen recording, determine what type of content is primarily "
    "being shown.\n\n"
    "Categories:\n"
    "- python_code: Python code being written, edited, debugged, or reviewed in an IDE or editor\n"
    "- markdown_document: A Markdown, RST, or text document being authored or edited\n"
    "- pdf_document: A PDF or formatted document being viewed, reviewed, or discussed\n"
    "- gui_demo: A GUI application being demonstrated — navigating menus, clicking buttons, "
    "configuring settings\n\n"
    "Respond with ONLY a valid JSON object (no markdown fences):\n"
    '{"type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}'
)

PLAN_PYTHON_SYSTEM = (
    "You are a reconstruction planner for screen recordings of Python coding sessions. "
    "Based on frame captions, identify ALL distinct Python files visible in the recording.\n\n"
    "For each file, provide:\n"
    "- filename: the file name as visible in the editor tab/title\n"
    "- description: what the file contains/does\n"
    "- key_content: notable imports, classes, functions visible\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{\n'
    '  "files": [{"filename": "...", "description": "...", "key_content": "..."}],\n'
    '  "parallel_safe": true/false,\n'
    '  "reasoning": "why parallel is or isn\'t safe"\n'
    '}\n\n'
    "Set parallel_safe to true ONLY if the files are independent (no cross-imports "
    "between them). If you can only identify one file, just list that one."
)

RECONSTRUCT_PYTHON_SYSTEM = (
    "You are an expert Python developer reconstructing source code from a screen recording. "
    "You are given frame-by-frame descriptions showing Python code in an editor.\n\n"
    "CRITICAL RULES:\n"
    "1. Reconstruct the COMPLETE, FINAL version of the code as it appears at the end\n"
    "2. Include ALL imports, function/class definitions, constants, and logic\n"
    "3. If the recording shows iterative edits, produce the FINAL state only\n"
    "4. Reproduce the code EXACTLY — do not add, remove, or 'improve' anything\n"
    "5. Use proper Python formatting, indentation, and style as shown\n"
    "6. Output ONLY the raw Python code — no markdown fences, no explanations"
)

RECONSTRUCT_MARKDOWN_SYSTEM = (
    "You are a document reconstruction specialist. You are given frame-by-frame "
    "descriptions of a screen recording showing a Markdown document being written or edited.\n\n"
    "CRITICAL RULES:\n"
    "1. Reconstruct the COMPLETE, FINAL version of the document\n"
    "2. Preserve ALL headings, lists, code blocks, tables, links, and formatting\n"
    "3. Reproduce text EXACTLY as shown — do not paraphrase or summarize\n"
    "4. If the recording shows edits, produce the FINAL state only\n"
    "5. Output ONLY the raw Markdown content — no wrapping or explanations"
)

RECONSTRUCT_PDF_SYSTEM = (
    "You are a document reconstruction specialist. Based on frame descriptions of a PDF "
    "document being viewed, reconstruct the document's full content in Markdown format.\n\n"
    "CRITICAL RULES:\n"
    "1. Preserve the document's structure — sections, subsections, numbered items\n"
    "2. Reproduce ALL text content as accurately as possible\n"
    "3. Render tables as Markdown tables\n"
    "4. Describe figures, charts, or diagrams in [Figure: ...] blocks\n"
    "5. Include page/slide numbers if visible\n"
    "6. Output ONLY the reconstructed Markdown content"
)

RECONSTRUCT_DEMO_WALKTHROUGH_SYSTEM = (
    "You are a technical writer producing a step-by-step walkthrough from a screen "
    "recording of a GUI application demonstration.\n\n"
    "Produce a structured Markdown document with:\n"
    "1. **Application Overview** — What application, version, and platform\n"
    "2. **Prerequisites** — What's needed before starting\n"
    "3. **Step-by-Step Walkthrough** — Numbered steps with:\n"
    "   - What to click/interact with\n"
    "   - What appears on screen after each action\n"
    "   - Any values entered or options selected\n"
    "4. **Key Observations** — Important settings, configurations, or behaviors noted\n\n"
    "Be specific about UI elements (button names, menu paths, field labels). "
    "Reference approximate timestamps where helpful."
)

RECONSTRUCT_DEMO_REFERENCE_SYSTEM = (
    "You are a technical writer producing a reference guide from a screen recording "
    "of a GUI application demonstration.\n\n"
    "Produce a structured Markdown document covering:\n"
    "1. **Application Architecture** — Components, panels, and navigation structure\n"
    "2. **Features Demonstrated** — Each feature with description and location in UI\n"
    "3. **Configuration Options** — Settings, preferences, and their effects\n"
    "4. **Keyboard Shortcuts / Controls** — Any shortcuts or special controls shown\n\n"
    "Focus on factual, reference-style documentation. No narrative."
)

EXTRACT_SEGMENT_SYSTEM = (
    "You are extracting raw content from a portion of a screen recording for "
    "later artifact reconstruction. The user prompt provides either frame "
    "captions for a video segment or extraction notes from a section of the "
    "recording.\n\n"
    "Your job: produce structured notes that preserve everything concrete. A "
    "separate synthesis pass will combine your notes with notes from other "
    "sections to build the final artifact, so anything you skip is "
    "permanently lost from the final output.\n\n"
    "CRITICAL RULES:\n"
    "1. EXTRACT, do not summarize. Preserve exact text, code, file names, UI "
    "labels, button names, paths, user inputs, system outputs, configuration "
    "values, error messages, numerical values.\n"
    "2. Reference timestamps (e.g. [00:01:23]) for concrete events whenever "
    "they appear in the input.\n"
    "3. Do not invent. If something is not in the input, do not include it.\n"
    "4. Be COMPACT in formatting: short bullet points, no narrative prose, "
    "no preamble, no commentary, no closing remarks.\n"
    "5. Group repeated observations: if the same UI element or status appears "
    "across many frames, note it once with the relevant timestamp range.\n"
    "6. Output ONLY the notes — no headings like 'EXTRACTION NOTES:'."
)


QA_REFLECT_SYSTEM = (
    "You are a quality assurance specialist reviewing reconstructed artifacts against "
    "the original screen recording frame captions.\n\n"
    "Evaluate the artifact on:\n"
    "1. **Completeness** (0-10): Does it capture ALL content shown in the recording?\n"
    "2. **Accuracy** (0-10): Is the content faithfully reproduced (not paraphrased/invented)?\n"
    "3. **Structure** (0-10): Is formatting, hierarchy, and organization correct?\n"
    "4. **Fidelity** (0-10): Would someone comparing this to the original find it faithful?\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    "{\n"
    '  "passed": true/false,\n'
    '  "scores": {"completeness": N, "accuracy": N, "structure": N, "fidelity": N},\n'
    '  "overall": <average 0-10>,\n'
    '  "feedback": "specific issues to address if not passed",\n'
    '  "missing_elements": ["list of specific things missing from the artifact"]\n'
    "}\n\n"
    "Pass threshold: overall >= 7.0. Be rigorous but fair."
)


# ── State Definitions ────────────────────────────────────────────────────────

class ReconstructState(TypedDict, total=False):
    """Main graph state for the reconstruction pipeline."""
    # Input
    folder_path: str
    folder_name: str
    captions: list[dict]
    config: dict

    # Classification
    content_type: str
    classification_confidence: float
    classification_reasoning: str

    # Reconstruction plan
    system_prompt: str
    reconstruction_tasks: list[dict]
    parallel_safe: bool

    # Hierarchical Pass-1 cache. Populated once per recording on the first
    # reconstruction iteration; reused unchanged across QA retries because the
    # captions don't change between iterations — only the QA feedback does, and
    # that flows through Pass 2 (synthesis), not Pass 1 (extraction).
    segment_notes: list[str]

    # Sub-agent output — uses add reducer for parallel fan-out collection
    artifacts: Annotated[list[dict], operator.add]

    # QA reflection
    qa_feedback: str
    qa_passed: bool
    qa_iteration: int
    qa_scores: dict

    # Output
    saved_paths: list[str]
    stage: str
    error: str
    elapsed_seconds: dict


# ── Model Cache ──────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _reconstruction_captioning_config(config: ScreenLensConfig):
    """Return the TEXT-role client config for reconstruction planning/QA.

    Reconstruction never looks at frames — it reasons over captions — so it runs
    on ``config.reconstruction`` (the text model), not the vision model that
    produced those captions. That also applies the reconstruction timeout, which
    artifact synthesis needs: it is a far longer generation than one caption.
    On DGX Spark both roles resolve to the same served vLLM checkpoint, so the
    model choice is a no-op there.
    """
    return text_role_captioning_config(config)


def get_inference_client(config: ScreenLensConfig) -> InferenceClient:
    """Create/cache the configured direct client for reuse across all nodes."""
    direct_config = _reconstruction_captioning_config(config)
    client = InferenceClient(direct_config)
    key = (
        client.backend.value,
        client.base_url,
        client.model,
        client.api_key,
        client.timeout,
    )
    if key not in _MODEL_CACHE:
        print(f"Using {client.backend.value} model: {client.model} at {client.base_url}")
        _MODEL_CACHE[key] = client
    return _MODEL_CACHE[key]


def generate_text(
    client: InferenceClient,
    system: str,
    user: str,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> str:
    """Generate text using the configured direct inference client."""
    return client.chat(
        system,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        extra={"chat_template_kwargs": {"enable_thinking": False}},
        require_complete=True,
    )


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
    return {}


def _build_caption_block(captions: list[dict], max_chars: int = 80000) -> str:
    """Build a formatted caption block for LLM consumption."""
    parts = []
    total_chars = 0
    for c in captions:
        ts = c.get("timestamp_str", "?")
        text = c.get("caption", "")
        entry = f"[{ts}]\n{text}"
        if total_chars + len(entry) > max_chars:
            parts.append("[... additional frames truncated for context limit ...]")
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n\n---\n\n".join(parts)


def _stratified_sample(items: list, n: int) -> list:
    """Pick ``n`` items spread evenly across ``items``, preserving order.

    Used by the QA reflector and ``plan_node``'s python file-identification
    step to give visibility into the entire recording's timeline rather than
    only the opening frames. Always includes the first and last items when
    ``n >= 2`` so endpoint context is preserved.
    """
    total = len(items)
    if total <= n:
        return list(items)
    if n <= 1:
        return [items[total // 2]]
    step = (total - 1) / (n - 1)
    return [items[int(round(i * step))] for i in range(n)]


def _estimated_text_tokens(text: str) -> int:
    """Estimate serialized note cost, including a separator allowance."""
    return max(1, (len(text) + 1) // 2) + 50


def _chunk_texts_by_budget(items: list[str], token_budget: int) -> list[list[str]]:
    """Greedily group notes by size and split an individually oversized note."""
    if token_budget <= 50:
        raise ValueError(f"Text token budget is too small: {token_budget}")
    max_text_chars = max(1, (token_budget - 50) * 2)
    units = [
        text[start : start + max_text_chars]
        for text in items
        for start in range(0, max(len(text), 1), max_text_chars)
    ]
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in units:
        text_tokens = _estimated_text_tokens(text)
        if current and current_tokens + text_tokens > token_budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += text_tokens
    if current:
        groups.append(current)
    return groups


def _get_model_context_size(client) -> int:
    """Return the configured context window used for chunk planning."""
    return client.context_size


def _long_form_output_ceiling(client, model_context: int) -> int:
    """Use the server's full context as the reconstruction completion ceiling.

    Passing the context ceiling to ``InferenceClient`` deliberately makes its
    vLLM path omit a literal ``max_tokens`` field, so vLLM assigns the exact
    space remaining after each rendered prompt. This follows a server upgrade
    automatically (for example, from 32K to Qwen3.6's native 262K context)
    without retaining the captioning path's smaller configured output default.
    """
    return max(1, int(model_context))


def _synthesis_output_reserve(output_ceiling: int, model_context: int) -> int:
    """Reserve output room for prompt planning without capping final output."""
    context_reserve = int(model_context * SYNTHESIS_OUTPUT_RESERVE_RATIO)
    return max(
        MIN_RECONSTRUCTION_OUTPUT_TOKENS,
        min(output_ceiling, max(MIN_RECONSTRUCTION_OUTPUT_TOKENS, context_reserve)),
    )


def _extract_segment_notes(
    captions: list[dict],
    client,
    model_context: int,
) -> list[str]:
    """Pass 1 of hierarchical reconstruction: extract content notes per chunk.

    Uses ``_compute_chunk_strategy`` for the safe input budget, then greedily
    packs each serialized caption by size. Each chunk gets a single inference
    call producing structured extraction notes (raw content, no synthesis).
    The result is a list of notes — one per chunk — that downstream synthesis
    passes consume.

    Task-agnostic by design: the same extraction is reused across all tasks
    in a single iteration AND across QA retries (cached in state). This
    avoids re-paying the dominant cost of the pipeline when only the
    synthesis prompt changes.
    """
    strategy = _compute_chunk_strategy(captions, model_context)
    output_ceiling = _long_form_output_ceiling(client, model_context)
    chunks = _chunk_captions_by_budget(
        captions,
        strategy["safe_context_tokens"],
        max_captions=MAX_CAPTIONS_PER_CHUNK,
    )
    extraction_strategy = "single_pass" if len(chunks) <= 1 else "hierarchical"
    max_chunk_size = max((len(chunk) for chunk in chunks), default=0)

    print(f"    [Pass 1] strategy={extraction_strategy} "
          f"max_chunk_size={max_chunk_size} "
          f"chunks={len(chunks)} "
          f"note_target<={SEGMENT_NOTE_TARGET_TOKENS:,} "
          f"context_ceiling={output_ceiling:,}")

    segment_notes: list[str] = []

    def extract_chunk(chunk: list[dict], label: str) -> list[str]:
        """Extract one bounded note, splitting the input if the model overruns."""
        start_ts = chunk[0].get("timestamp_str", "?")
        end_ts = chunk[-1].get("timestamp_str", "?")

        chunk_block = "\n\n---\n\n".join(
            f"[{c.get('timestamp_str', '?')}]\n{c.get('caption', '')}"
            for c in chunk
        )
        user = (
            f"Segment {label} of {len(chunks)} from a recording "
            f"(timestamps {start_ts} — {end_ts}, {len(chunk)} frames). "
            "Produce dense evidence notes for later artifact reconstruction. "
            "Preserve exact visible source text, filenames, values, and final-state "
            "changes. Consolidate repeated observations and omit routine UI chrome, "
            "navigation, and descriptions that add no artifact content. "
            f"Keep the response at or below {SEGMENT_NOTE_TARGET_TOKENS:,} tokens.\n\n"
            f"SEGMENT CAPTIONS:\n\n{chunk_block}"
        )
        t0 = time.time()
        try:
            notes = generate_text(
                client, EXTRACT_SEGMENT_SYSTEM, user,
                max_tokens=output_ceiling, temperature=0.1,
            )
        except InferenceTruncatedError as exc:
            if len(chunk) <= 1:
                raise RuntimeError(
                    "Segment-note extraction could not produce a complete bounded "
                    f"response for the frame at {start_ts}."
                ) from exc
            midpoint = len(chunk) // 2
            print(f"    [Pass 1] segment {label} exhausted the full "
                  f"{output_ceiling:,}-token context; retrying as two smaller "
                  "caption groups")
            return (
                extract_chunk(chunk[:midpoint], f"{label}a")
                + extract_chunk(chunk[midpoint:], f"{label}b")
            )
        elapsed = time.time() - t0
        heading = (
            "[Full recording]"
            if len(chunks) == 1 and label == "1"
            else f"[Segment {label}: {start_ts} — {end_ts}]"
        )
        print(f"    [Pass 1] segment {label}/{len(chunks)} done "
              f"({len(notes)} chars, {elapsed:.1f}s)")
        return [f"{heading}\n{notes}"]

    for i, chunk in enumerate(chunks, 1):
        segment_notes.extend(extract_chunk(chunk, str(i)))

    return segment_notes


def _intermediate_target_tokens(
    group_tokens: int,
) -> int:
    """Return the requested task-focused note target for a source group."""
    return max(
        MIN_RECONSTRUCTION_OUTPUT_TOKENS * 2,
        min(
            INTERMEDIATE_TARGET_MAX_TOKENS,
            int(group_tokens * INTERMEDIATE_TARGET_RATIO),
        ),
    )


def _hierarchical_synthesize(
    notes: list[str],
    task_user_prefix: str,
    task_system_prompt: str,
    client,
    model_context: int,
    max_output_tokens: int | None = None,
    *,
    _depth: int = 0,
) -> str:
    """Pass 2 of hierarchical reconstruction: synthesize segment notes into a
    final artifact, recursing if the notes don't all fit in one call.

    Single pass when the notes fit in the model's safe input budget.
    Otherwise: group notes into super-chunks, run an intermediate synthesis
    on each (using EXTRACT_SEGMENT_SYSTEM to preserve detail rather than
    finalize), then recurse on the intermediate notes. The recursion is
    guarded against non-termination by requiring each condensation pass to
    reduce the estimated note size and by enforcing a maximum recursion depth.
    """
    if _depth >= MAX_SYNTHESIS_DEPTH:
        raise RuntimeError(
            "Reconstruction synthesis exceeded its maximum condensation depth; "
            "the model is not reducing the intermediate notes enough to fit."
        )

    if max_output_tokens is None:
        max_output_tokens = _long_form_output_ceiling(client, model_context)
    else:
        max_output_tokens = max(1, min(max_output_tokens, model_context))

    # This reserve only controls how much input a synthesis prompt may pack.
    # The actual final call still receives the full configured ceiling; vLLM
    # then allocates the exact context remaining after the prompt.
    output_reserve = _synthesis_output_reserve(max_output_tokens, model_context)
    safe_input_tokens = max(
        MIN_SYNTHESIS_INPUT_TOKENS,
        int((model_context - SYNTHESIS_OVERHEAD_TOKENS - output_reserve) * 0.85),
    )

    estimated_tokens = sum(_estimated_text_tokens(note) for note in notes)

    if estimated_tokens <= safe_input_tokens:
        # Single synthesis pass — all notes fit
        notes_block = "\n\n---\n\n".join(notes)
        user = (
            f"{task_user_prefix}\n\n"
            f"You are working from segment-by-segment extraction notes covering "
            f"the entire recording in chronological order. Each segment's notes "
            f"contain raw extracted content with timestamps. Combine information "
            f"across all segments to produce the final artifact.\n\n"
            f"SEGMENT NOTES:\n\n{notes_block}"
        )
        return generate_text(
            client, task_system_prompt, user,
            max_tokens=max_output_tokens, temperature=0.1,
        )

    # Recursive group-and-condense. Budget each serialized note independently;
    # a fixed count derived from the average fails when one note is an outlier.
    groups = _chunk_texts_by_budget(notes, safe_input_tokens)
    max_group_size = max((len(group) for group in groups), default=0)

    print(f"    [Pass 2] {len(notes)} notes (~{estimated_tokens:,} tok) "
          f"exceed budget — recursing in {len(groups)} size-budgeted groups "
          f"(max {max_group_size} notes)")

    def condense_group(
        group: list[str],
        section_label: str,
        retry_depth: int = 0,
    ) -> list[str]:
        """Produce task-focused notes, splitting and retrying any overrun."""
        group_tokens = sum(_estimated_text_tokens(note) for note in group)
        target_tokens = _intermediate_target_tokens(group_tokens)
        group_block = "\n\n---\n\n".join(group)
        intermediate_user = (
            f"You are filtering section {section_label} of extraction notes for "
            "one reconstruction task. Produce dense intermediate notes containing "
            "every concrete fact needed for the task below. Consolidate duplicates "
            "and discard material solely about other files/artifacts, routine UI "
            "navigation, and repeated descriptions. Do not finalize the artifact.\n\n"
            f"TASK FOCUS:\n{task_user_prefix}\n\n"
            f"TARGET LENGTH: at most {target_tokens:,} tokens.\n\n"
            f"SECTION NOTES:\n\n{group_block}"
        )
        t0 = time.time()
        try:
            result = generate_text(
                client, EXTRACT_SEGMENT_SYSTEM, intermediate_user,
                max_tokens=max_output_tokens, temperature=0.1,
            )
        except InferenceTruncatedError as exc:
            retry_budget = max(MIN_SYNTHESIS_INPUT_TOKENS, group_tokens // 2)
            subgroups = _chunk_texts_by_budget(group, retry_budget)
            if retry_depth >= MAX_SYNTHESIS_DEPTH or len(subgroups) <= 1:
                raise RuntimeError(
                    "Task-focused reconstruction notes still exceeded their "
                    f"completion cap after input splitting (section {section_label})."
                ) from exc
            print(f"    [Pass 2] intermediate {section_label} exhausted the "
                  f"full {max_output_tokens:,}-token context; retrying in "
                  f"{len(subgroups)} smaller groups")
            retried: list[str] = []
            for index, subgroup in enumerate(subgroups, 1):
                retried.extend(condense_group(
                    subgroup,
                    f"{section_label}.{index}",
                    retry_depth + 1,
                ))
            return retried
        elapsed = time.time() - t0
        print(f"    [Pass 2] intermediate {section_label} done "
              f"({len(result)} chars, target<={target_tokens:,}, {elapsed:.1f}s)")
        return [f"[Section {section_label}]\n{result}"]

    intermediate_notes: list[str] = []
    for i, group in enumerate(groups, 1):
        intermediate_notes.extend(condense_group(group, f"{i}/{len(groups)}"))

    next_estimated_tokens = sum(
        _estimated_text_tokens(note) for note in intermediate_notes
    )
    if next_estimated_tokens >= estimated_tokens:
        raise RuntimeError(
            "Reconstruction condensation made no progress "
            f"(~{estimated_tokens:,} input tokens -> "
            f"~{next_estimated_tokens:,} output tokens). The model did not "
            "follow the dense-intermediate-notes instruction."
        )

    return _hierarchical_synthesize(
        intermediate_notes, task_user_prefix, task_system_prompt,
        client, model_context, max_output_tokens, _depth=_depth + 1,
    )


# ── Pipeline Nodes ───────────────────────────────────────────────────────────

def classify_node(state: ReconstructState) -> dict:
    """Classify the content type from captions."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)
    captions = state["captions"]

    print(f"\n{'='*60}")
    print(f"[1/5] CLASSIFYING CONTENT — {state['folder_name']}")
    print(f"      Analyzing {len(captions)} frame captions")
    print(f"{'='*60}")

    # Build a summary of captions for classification (cap at 50 frames)
    caption_texts = []
    for c in captions[:50]:
        ts = c.get("timestamp_str", "?")
        text = c.get("caption", "")[:500]
        caption_texts.append(f"[{ts}] {text}")

    user_prompt = (
        f"Screen recording: {len(captions)} frames total.\n\n"
        f"Frame captions (first {min(len(captions), 50)}):\n\n"
        + "\n\n---\n\n".join(caption_texts)
    )

    response = generate_text(client, CLASSIFY_SYSTEM, user_prompt,
                              max_tokens=512, temperature=0.1)
    result = parse_json_response(response)

    content_type = result.get("type", "gui_demo")
    if content_type not in CONTENT_TYPES:
        content_type = "gui_demo"

    confidence = result.get("confidence", 0.5)
    reasoning = result.get("reasoning", "No reasoning provided")

    elapsed = time.time() - t0
    print(f"\n  Content type: {content_type} (confidence: {confidence:.0%})")
    print(f"  Reasoning: {reasoning}")
    print(f"  Classified in {elapsed:.1f}s")

    return {
        "content_type": content_type,
        "classification_confidence": confidence,
        "classification_reasoning": reasoning,
        "stage": "classified",
        "elapsed_seconds": {"classify": round(elapsed, 2)},
    }


def plan_node(state: ReconstructState) -> dict:
    """Generate reconstruction plan: system prompt, task list, parallelism decision."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)
    content_type = state["content_type"]
    captions = state["captions"]
    qa_feedback = state.get("qa_feedback", "")
    qa_iteration = state.get("qa_iteration", 0)

    print(f"\n{'='*60}")
    print(f"[2/5] PLANNING RECONSTRUCTION — {content_type}")
    if qa_iteration > 0:
        print(f"      Retry #{qa_iteration} — incorporating QA feedback")
    print(f"{'='*60}")

    tasks = []
    parallel_safe = False
    system_prompt = ""

    if content_type == "python_code":
        system_prompt = RECONSTRUCT_PYTHON_SYSTEM

        # File identification needs visibility into the whole recording, so
        # use a stratified sample rather than the linear opening slice.
        sampled = _stratified_sample(captions, 60)
        sample_block = _build_caption_block(sampled, max_chars=80000)
        file_id_prompt = f"Frame captions from a Python coding session:\n\n{sample_block}"
        # The file list is unbounded — one entry per Python file visible, each
        # with a description and key_content — so it cannot share the classifier's
        # small fixed cap. Let the model run to its natural stop.
        plan_ceiling = _long_form_output_ceiling(client, _get_model_context_size(client))
        response = generate_text(client, PLAN_PYTHON_SYSTEM,
                                  file_id_prompt, max_tokens=plan_ceiling,
                                  temperature=0.1)
        plan = parse_json_response(response)

        files = plan.get("files", [{"filename": "reconstructed.py",
                                     "description": "Main script"}])
        parallel_safe = plan.get("parallel_safe", False) and len(files) > 1

        for f in files:
            task_prompt = (
                f"Reconstruct the file '{f['filename']}' ({f.get('description', '')}).\n\n"
            )
            if qa_feedback and qa_iteration > 0:
                task_prompt += (
                    f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"
                )

            tasks.append({
                "filename": f["filename"],
                "description": f.get("description", ""),
                "prompt": task_prompt,
                "output_type": "python",
            })

        print(f"  Identified {len(files)} Python file(s):")
        for f in files:
            print(f"    - {f['filename']}: {f.get('description', '')}")

    elif content_type == "markdown_document":
        system_prompt = RECONSTRUCT_MARKDOWN_SYSTEM
        parallel_safe = False

        task_prompt = "Reconstruct the complete Markdown document.\n\n"
        if qa_feedback and qa_iteration > 0:
            task_prompt += f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"

        tasks.append({
            "filename": "document.md",
            "description": "Reconstructed Markdown document",
            "prompt": task_prompt,
            "output_type": "markdown",
        })

    elif content_type == "pdf_document":
        system_prompt = RECONSTRUCT_PDF_SYSTEM
        parallel_safe = False

        task_prompt = "Reconstruct the PDF document content in Markdown format.\n\n"
        if qa_feedback and qa_iteration > 0:
            task_prompt += f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"

        tasks.append({
            "filename": "document.md",
            "description": "Reconstructed PDF content",
            "prompt": task_prompt,
            "output_type": "markdown",
        })

    elif content_type == "gui_demo":
        # GUI demos produce independent documents → parallel-safe
        parallel_safe = True

        base_context = ""
        if qa_feedback and qa_iteration > 0:
            base_context = f"PREVIOUS QA FEEDBACK — address these issues:\n{qa_feedback}\n\n"

        tasks.append({
            "filename": "walkthrough.md",
            "description": "Step-by-step walkthrough",
            "prompt": f"Generate a detailed step-by-step walkthrough.\n\n{base_context}",
            "output_type": "markdown",
            "system_override": RECONSTRUCT_DEMO_WALKTHROUGH_SYSTEM,
        })
        tasks.append({
            "filename": "reference.md",
            "description": "Technical reference guide",
            "prompt": f"Generate a technical reference guide.\n\n{base_context}",
            "output_type": "markdown",
            "system_override": RECONSTRUCT_DEMO_REFERENCE_SYSTEM,
        })

    print(f"  Tasks: {len(tasks)} | Parallel dispatch: {parallel_safe}")
    elapsed = time.time() - t0
    print(f"  Planned in {elapsed:.1f}s")

    return {
        "system_prompt": system_prompt,
        "reconstruction_tasks": tasks,
        "parallel_safe": parallel_safe,
        "stage": "planned",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "plan": round(elapsed, 2)},
    }


def reconstruct_worker(state: dict) -> dict:
    """Execute a single reconstruction task via LangGraph Send for parallel fan-out.

    Currently unreachable: ``route_to_workers`` always dispatches sequentially.
    Kept in sync with ``reconstruct_sequential`` so that re-enabling parallel
    dispatch would not silently regress.
    """
    task = state["task"]
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)
    captions = state.get("captions", [])
    segment_notes = state.get("segment_notes") or []
    model_context = _get_model_context_size(client)

    if not segment_notes and captions:
        segment_notes = _extract_segment_notes(captions, client, model_context)

    task_system = task.get("system_override", state.get("system_prompt", ""))
    task_user_prefix = task["prompt"]

    print(f"    [sub-agent] Reconstructing: {task['filename']}")
    t0 = time.time()

    content = _hierarchical_synthesize(
        segment_notes, task_user_prefix, task_system,
        client, model_context,
    )

    elapsed = time.time() - t0
    print(f"    [sub-agent] {task['filename']} done "
          f"({len(content)} chars, {elapsed:.1f}s)")

    return {
        "artifacts": [{
            "filename": task["filename"],
            "content": content,
            "type": task["output_type"],
            "description": task.get("description", ""),
            "iteration": state.get("qa_iteration", 0),
        }],
        "segment_notes": segment_notes,
    }


def reconstruct_sequential(state: ReconstructState) -> dict:
    """Process all reconstruction tasks sequentially via hierarchical synthesis.

    Two-pass design:
      Pass 1 — extract structured content notes from each chunk of captions
               (task-agnostic, shared across all tasks). Cached in state and
               reused on QA retries since the captions don't change between
               iterations.
      Pass 2 — for each task, hierarchically synthesize the segment notes
               into the final artifact, using the task's system prompt and
               any QA feedback embedded in the task user prompt.
    """
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)
    captions = state["captions"]
    tasks = state["reconstruction_tasks"]
    system_prompt = state.get("system_prompt", "")
    qa_iteration = state.get("qa_iteration", 0)

    model_context = _get_model_context_size(client)
    print(f"\n  Processing {len(tasks)} task(s) sequentially "
          f"(model context: {model_context:,} tokens)")

    # Pass 1: extract segment notes once per recording. Cached across QA
    # retries because the captions are stable — only the synthesis prompt
    # changes between iterations.
    segment_notes = state.get("segment_notes") or []
    if not segment_notes:
        print(f"  [Pass 1] Extracting segment notes from {len(captions)} captions...")
        t1 = time.time()
        segment_notes = _extract_segment_notes(
            captions, client, model_context,
        )
        print(f"  [Pass 1] Produced {len(segment_notes)} segment notes "
              f"in {time.time() - t1:.1f}s")
    else:
        print(f"  [Pass 1] Reusing {len(segment_notes)} cached segment notes "
              f"(QA iteration {qa_iteration})")

    # Pass 2: per-task synthesis from the shared segment notes.
    new_artifacts = []
    for i, task in enumerate(tasks, 1):
        task_system = task.get("system_override", system_prompt)
        task_user_prefix = task["prompt"]

        print(f"\n  [Pass 2] [{i}/{len(tasks)}] Synthesizing: {task['filename']}")
        t0 = time.time()

        content = _hierarchical_synthesize(
            segment_notes, task_user_prefix, task_system,
            client, model_context,
        )

        elapsed = time.time() - t0
        print(f"  [Pass 2] [{i}/{len(tasks)}] {task['filename']} done "
              f"({len(content)} chars, {elapsed:.1f}s)")

        new_artifacts.append({
            "filename": task["filename"],
            "content": content,
            "type": task["output_type"],
            "description": task.get("description", ""),
            "iteration": qa_iteration,
        })

    return {
        "artifacts": new_artifacts,
        "segment_notes": segment_notes,
    }


def qa_reflect_node(state: ReconstructState) -> dict:
    """Quality-check artifacts using a reflection agent. Routes to retry or save."""
    t0 = time.time()
    config = ScreenLensConfig(**state["config"])
    client = get_inference_client(config)
    qa_iteration = state.get("qa_iteration", 0)
    captions = state["captions"]

    # Get only artifacts from current iteration
    all_artifacts = state.get("artifacts", [])
    current_artifacts = [a for a in all_artifacts if a.get("iteration", 0) == qa_iteration]
    if not current_artifacts:
        # Fallback: take the most recent N artifacts
        n_tasks = len(state.get("reconstruction_tasks", []))
        current_artifacts = all_artifacts[-n_tasks:] if n_tasks else all_artifacts

    print(f"\n{'='*60}")
    print(f"[4/5] QA REFLECTION — iteration {qa_iteration + 1}/{MAX_QA_ITERATIONS}")
    print(f"      Reviewing {len(current_artifacts)} artifact(s)")
    print(f"{'='*60}")

    # Build QA context. Two important details:
    #   1. Captions are stratified-sampled across the *entire* recording so
    #      QA can validate content from any point in the timeline. A linear
    #      `captions[:30]` slice would only see the opening frames and would
    #      flag legitimate later content as "hallucinated".
    #   2. Artifacts get a generous per-document slice so QA actually sees
    #      the whole reconstruction. A short slice causes the reflector to
    #      mistake its own display window for real artifact truncation.
    sampled_captions = _stratified_sample(captions, 50)
    caption_summary = _build_caption_block(sampled_captions, max_chars=40000)

    ARTIFACT_SLICE_CHARS = 30000
    artifacts_text = ""
    for a in current_artifacts:
        artifacts_text += f"\n\n--- {a['filename']} ({a['type']}) ---\n"
        artifacts_text += a["content"][:ARTIFACT_SLICE_CHARS]
        if len(a["content"]) > ARTIFACT_SLICE_CHARS:
            artifacts_text += (
                f"\n[... truncated for QA display only, "
                f"{len(a['content'])} total chars in saved artifact ...]"
            )
        artifacts_text += "\n"

    user_prompt = (
        f"Content type: {state.get('content_type', 'unknown')}\n\n"
        f"ORIGINAL FRAME CAPTIONS — {len(sampled_captions)} frames sampled "
        f"evenly across the full {len(captions)}-frame recording "
        f"(timestamps span the entire video):\n{caption_summary}\n\n"
        f"RECONSTRUCTED ARTIFACTS:\n{artifacts_text}"
    )

    # feedback + per-artifact scores + missing_elements grow with the artifact
    # set, so this JSON has no small fixed bound either.
    qa_ceiling = _long_form_output_ceiling(client, _get_model_context_size(client))
    response = generate_text(client, QA_REFLECT_SYSTEM, user_prompt,
                              max_tokens=qa_ceiling, temperature=0.1)
    result = parse_json_response(response)

    passed = result.get("passed", True)
    overall = result.get("overall", 7.0)
    feedback = result.get("feedback", "")
    scores = result.get("scores", {})
    missing = result.get("missing_elements", [])

    # Force pass after max iterations
    if qa_iteration >= MAX_QA_ITERATIONS - 1 and not passed:
        passed = True
        feedback += " [Max iterations reached — accepting current output]"
        print(f"  Max QA iterations reached — accepting output.")

    elapsed = time.time() - t0
    print(f"\n  QA Score: {overall}/10")
    if scores:
        print(f"  Breakdown: {json.dumps(scores)}")
    print(f"  Passed: {'YES' if passed else 'NO'}")
    if not passed:
        print(f"  Feedback: {feedback}")
        if missing:
            print(f"  Missing: {', '.join(missing[:5])}")
    print(f"  Reflection completed in {elapsed:.1f}s")

    return {
        "qa_passed": passed,
        "qa_feedback": feedback,
        "qa_scores": scores,
        "qa_iteration": qa_iteration + 1 if not passed else qa_iteration,
        "stage": "qa_passed" if passed else "qa_retry",
        "elapsed_seconds": {
            **state.get("elapsed_seconds", {}),
            f"qa_{qa_iteration}": round(elapsed, 2),
        },
    }


def save_node(state: ReconstructState) -> dict:
    """Save reconstructed artifacts to the output folder."""
    t0 = time.time()
    folder_path = Path(state["folder_path"])
    output_dir = folder_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_iteration = state.get("qa_iteration", 0)

    # Get latest-iteration artifacts
    all_artifacts = state.get("artifacts", [])
    latest = [a for a in all_artifacts if a.get("iteration", 0) == qa_iteration]
    if not latest:
        n_tasks = len(state.get("reconstruction_tasks", []))
        latest = all_artifacts[-n_tasks:] if n_tasks else all_artifacts

    print(f"\n{'='*60}")
    print(f"[5/5] SAVING ARTIFACTS — {len(latest)} file(s)")
    print(f"      Output: {output_dir}")
    print(f"{'='*60}")

    saved = []
    for artifact in latest:
        filepath = output_dir / artifact["filename"]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(artifact["content"])
        saved.append(str(filepath))
        print(f"  saved {artifact['filename']} ({len(artifact['content']):,} chars)")

    # Save reconstruction metadata
    meta = {
        "content_type": state.get("content_type"),
        "classification_confidence": state.get("classification_confidence"),
        "classification_reasoning": state.get("classification_reasoning"),
        "qa_scores": state.get("qa_scores", {}),
        "qa_iterations_used": state.get("qa_iteration", 0) + 1,
        "max_qa_iterations": MAX_QA_ITERATIONS,
        "artifacts": [
            {
                "filename": a["filename"],
                "type": a["type"],
                "description": a.get("description", ""),
                "size_chars": len(a["content"]),
            }
            for a in latest
        ],
    }
    meta_path = output_dir / "reconstruction_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    saved.append(str(meta_path))

    elapsed = time.time() - t0
    print(f"\n  Saved {len(saved)} files in {elapsed:.1f}s")

    return {
        "saved_paths": saved,
        "stage": "saved",
        "elapsed_seconds": {**state.get("elapsed_seconds", {}), "save": round(elapsed, 2)},
    }


# ── Routing Functions ────────────────────────────────────────────────────────

def route_to_workers(state: ReconstructState):
    """Dispatch reconstruction tasks sequentially.

    Keep reconstruction deterministic by processing tasks in order. The
    expensive work is delegated to the inference server, which handles its own
    scheduling and batching.
    """
    tasks = state.get("reconstruction_tasks", [])
    print(f"\n  Sequential execution ({len(tasks)} task(s))")
    return "reconstruct_sequential"


def should_retry_or_save(state: ReconstructState) -> str:
    """After QA: retry reconstruction or proceed to save."""
    if state.get("qa_passed", False):
        return "save"
    return "plan"


# ── Graph Construction ───────────────────────────────────────────────────────

def build_reconstruct_graph():
    """Build the reconstruction pipeline with parallel sub-agents and reflection loop.

    Graph topology:
        START → classify → plan →[dispatch]→ worker(s)    → qa_reflect → save → END
                                  ↘ sequential ↗            ↓      ↑
                                                            plan ←─╯ (retry)
    """
    graph = StateGraph(ReconstructState)

    # Nodes
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("reconstruct_worker", reconstruct_worker)
    graph.add_node("reconstruct_sequential", reconstruct_sequential)
    graph.add_node("qa_reflect", qa_reflect_node)
    graph.add_node("save", save_node)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "plan")
    graph.add_conditional_edges(
        "plan", route_to_workers,
        ["reconstruct_worker", "reconstruct_sequential"],
    )
    graph.add_edge("reconstruct_worker", "qa_reflect")
    graph.add_edge("reconstruct_sequential", "qa_reflect")
    graph.add_conditional_edges(
        "qa_reflect", should_retry_or_save,
        {"plan": "plan", "save": "save"},
    )
    graph.add_edge("save", END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

def reconstruct_folder(folder_path: str, config: ScreenLensConfig) -> dict:
    """Run the full reconstruction pipeline on a single data folder.

    Args:
        folder_path: Path to a data/<video_name> folder containing captions/
        config: ScreenLensConfig instance

    Returns:
        Pipeline result dict with saved_paths, qa_scores, content_type, etc.
    """
    folder = Path(folder_path)
    captions_file = folder / "captions" / "all_captions.json"

    if not captions_file.exists():
        return {"error": f"No captions found at {captions_file}", "stage": "error"}

    with open(captions_file) as f:
        captions = json.load(f)

    if not captions:
        return {"error": "Captions file is empty", "stage": "error"}

    pipeline = build_reconstruct_graph()
    initial_state = {
        "folder_path": str(folder),
        "folder_name": folder.name,
        "captions": captions,
        "config": config.model_dump(),
        "artifacts": [],
        "qa_iteration": 0,
    }

    return pipeline.invoke(initial_state)
