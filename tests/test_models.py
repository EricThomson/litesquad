"""Transcript (de)serialization tests."""

from litesquad.models import TranscriptEvent, load_events, load_events_lenient


def make_event(**overrides) -> TranscriptEvent:
    fields = dict(turn_index=0, stage="propose", role="worker_1",
                  model="anthropic/worker-a", prompt="p", task="t", output="o")
    fields.update(overrides)
    return TranscriptEvent(**fields)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    original = make_event()
    path.write_text(original.to_jsonl() + "\n", encoding="utf-8")
    assert load_events(path) == [original]


def test_lenient_load_skips_foreign_lines(tmp_path):
    good = make_event()
    path = tmp_path / "t.jsonl"
    path.write_text(
        "\n".join([
            good.to_jsonl(),
            '{"stage": "not-a-real-stage", "role": "old"}',  # outdated format
            "not json at all",
            "",
            good.to_jsonl(),
        ]) + "\n",
        encoding="utf-8",
    )
    events, skipped = load_events_lenient(path)
    assert events == [good, good]
    assert skipped == 2
