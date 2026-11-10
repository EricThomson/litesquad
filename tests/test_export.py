"""Markdown export tests: one turn's question and final answer as a standalone document."""

from datetime import datetime

import pytest

from litesquad.export import answer_filename, answer_markdown, tidy_filename
from litesquad.models import TranscriptEvent

# Built in local time, so the header (which shows local time) reads 13:13 on any machine.
FINISHED_AT = datetime(2026, 10, 15, 13, 13).astimezone().isoformat()


def make_event(stage: str, role: str, **overrides) -> TranscriptEvent:
    fields = dict(turn_index=0, stage=stage, role=role, model=f"provider/{role}-model",
                  prompt="p", task="What should I name my cat?", output="output",
                  ts=FINISHED_AT)
    fields.update(overrides)
    return TranscriptEvent(**fields)


def squad_turn(workers: int = 2, **judge_overrides) -> list[TranscriptEvent]:
    events = [make_event("extract", f"extractor->worker_{n}") for n in range(1, workers + 1)]
    events.append(make_event("cluster", "clusterer"))
    judge_fields = {"output": "## Names\n\nTry Miso.", **judge_overrides}
    events.append(make_event("judge", "judge", **judge_fields))
    return events


def test_squad_answer_is_header_question_rule_answer():
    assert answer_markdown(squad_turn()) == (
        "**litesquad** | 2026-10-15 13:13 | squad, 2 workers | judge: provider/judge-model\n"
        "\n"
        "> What should I name my cat?\n"
        "\n"
        "---\n"
        "\n"
        "## Names\n\nTry Miso.\n"  # the judge's own headings survive untouched
    )


def test_worker_count_is_only_workers_whose_content_reached_the_judge():
    events = squad_turn(workers=3)
    events[1] = make_event("extract", "extractor->worker_2", output="", error="bad JSON")
    assert "| squad, 2 workers |" in answer_markdown(events)


def test_quick_answer_says_judge_only():
    events = [make_event("reply", "judge", output="Miso.")]
    assert "| quick, judge only |" in answer_markdown(events)


def test_follow_up_turn_says_so():
    header = answer_markdown(squad_turn(turn_index=2)).splitlines()[0]
    assert header.endswith("| follow-up, turn 3")
    assert "follow-up" not in answer_markdown(squad_turn()).splitlines()[0]


def test_multi_paragraph_question_stays_one_blockquote():
    events = squad_turn(task="First paragraph.\n\nSecond paragraph.")
    assert "> First paragraph.\n>\n> Second paragraph.\n" in answer_markdown(events)


def test_turn_without_a_final_answer_refuses():
    events = squad_turn(output="", error="judge timed out")
    with pytest.raises(ValueError):
        answer_markdown(events)


def test_filename_traces_back_to_the_transcript():
    assert answer_filename("2026-10-15_131301_web.jsonl", 1) == "2026-10-15_131301_web_turn2.md"


@pytest.mark.parametrize("typed, expected", [
    ("cat names", "cat names.md"),  # .md added when missing
    ("  cat names.md  ", "cat names.md"),  # spaces trimmed, no doubled suffix
    ("Cats.MD", "Cats.MD"),  # the suffix check ignores case
    ("notes/cats", "cats.md"),  # a typed folder is dropped: the browser picks where
    (r"C:\Users\me\cats.md", "cats.md"),
])
def test_tidy_filename_cleans_what_was_typed(typed, expected):
    assert tidy_filename(typed, "default.md") == expected


@pytest.mark.parametrize("typed", [None, "", "   ", ".", "..", "notes/"])
def test_tidy_filename_falls_back_when_nothing_usable_is_typed(typed):
    assert tidy_filename(typed, "default.md") == "default.md"
