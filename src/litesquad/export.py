"""Markdown export of one turn's final answer, for reading outside litesquad.

Plain text in, plain text out: no Dash or rich imports, so any frontend can offer
it. The file holds what a reader needs later and nothing more: a one-line header
saying when and how the answer was made, the question, and the answer itself.

The answer is copied verbatim. The judge already writes markdown, often with its
own headings, so the export frames it with a bold header line, a blockquote, and
a horizontal rule instead of adding headings that would clash with the answer's.
"""

from datetime import datetime
from pathlib import Path

from .models import TranscriptEvent, final_event


def answer_markdown(turn_events: list[TranscriptEvent]) -> str:
    """One turn's question and final answer as a standalone markdown document.

    Raises ValueError for a turn with no final answer (an aborted run), so callers
    decide explicitly what to do with a turn that has nothing worth saving.
    """
    final = final_event(turn_events)
    if final is None:
        raise ValueError("this turn has no final answer to export")
    return (
        f"{_header_line(turn_events, final)}\n\n"
        f"{_blockquote(final.task)}\n\n"
        "---\n\n"
        f"{final.output.strip()}\n"
    )


def answer_filename(transcript_name: str, turn_index: int) -> str:
    """The export's file name, built from the transcript's so a saved answer can always
    be traced back to its full record: ``2026-10-15_131301_web.jsonl`` with turn index 1
    (the second turn) gives ``2026-10-15_131301_web_turn2.md``."""
    return f"{Path(transcript_name).stem}_turn{turn_index + 1}.md"


def _header_line(turn_events: list[TranscriptEvent], final: TranscriptEvent) -> str:
    """When and how the answer was made, in one line, for example
    ``**litesquad** | 2026-10-15 13:13 | squad, 6 workers | judge: anthropic/claude-opus-5.5``.

    The time is when the answer finished, in local time. The worker count is the
    workers whose content actually reached the judge (a dropped worker has no
    successful extraction), so it describes this answer, not the configured roster.
    A follow-up turn says so: its question may lean on earlier turns the file
    does not include.
    """
    finished_at = datetime.fromisoformat(final.ts).astimezone().strftime("%Y-%m-%d %H:%M")
    if final.stage == "reply":
        mode = "quick, judge only"
    else:
        contributing_workers = sum(
            1 for event in turn_events if event.stage == "extract" and not event.error
        )
        noun = "worker" if contributing_workers == 1 else "workers"
        mode = f"squad, {contributing_workers} {noun}"
    parts = ["**litesquad**", finished_at, mode, f"judge: {final.model}"]
    if final.turn_index > 0:
        parts.append(f"follow-up, turn {final.turn_index + 1}")
    return " | ".join(parts)


def _blockquote(text: str) -> str:
    """Quote every line of ``text``. Blank lines become a bare ``>`` so a multi-paragraph
    question stays one continuous quote instead of breaking into several."""
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())
