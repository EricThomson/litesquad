"""In-memory transcript models. Each :class:`TranscriptEvent` is one JSONL row."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["propose", "critique", "revise", "synthesize", "reply"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranscriptEvent(BaseModel):
    turn_index: int
    stage: Stage
    role: str
    model: str
    prompt: str
    output: str = ""
    error: str | None = None
    ts: str = Field(default_factory=_now)

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class Turn(BaseModel):
    index: int
    task: str
    events: list[TranscriptEvent] = Field(default_factory=list)

    def add(self, event: TranscriptEvent) -> TranscriptEvent:
        self.events.append(event)
        return event

    @property
    def final_answer(self) -> str | None:
        # deep turns end in "synthesize", quick turns in "reply"
        for event in reversed(self.events):
            if event.stage in ("synthesize", "reply") and not event.error:
                return event.output
        return None


class Conversation(BaseModel):
    turns: list[Turn] = Field(default_factory=list)

    def new_turn(self, task: str) -> Turn:
        turn = Turn(index=len(self.turns), task=task)
        self.turns.append(turn)
        return turn

    def history_digest(self) -> str:
        """Compact prior message/answer pairs, for follow-up context."""
        parts: list[str] = []
        for turn in self.turns:
            if turn.final_answer:
                parts.append(f"User: {turn.task}\nAnswer: {turn.final_answer}")
        return "\n\n".join(parts)
