"""Orchestration.

Deep turn: each worker responds independently, the critic critiques each response, and the
worker revises. Then the revised answers are extracted into de-stylized content units,
clustered across responses into a content map, and the judge writes the single final answer
from that map (extract -> cluster -> judge). Quick turn: the judge answers directly, no
ensemble. Rendering-agnostic: progress is reported through a :class:`Reporter` so this module
never imports ``rich``.
"""

import random
from typing import Callable, Protocol

from . import prompts
from .config import AgentConfig, RunConfig, SquadConfig
from .llm import LLMError, call_model, json_list
from .models import Conversation, Stage, TranscriptEvent, Turn

# De-identified provenance labels for the content map: the clusterer and judge see "Draft A/B/C",
# never which model wrote what, so identity can't bias the grouping or the final answer.
_DRAFT_LABELS = ["Draft A", "Draft B", "Draft C", "Draft D", "Draft E"]


def _system(base: str, agent: AgentConfig) -> str:
    """Append an agent's optional per-config instructions to its base prompt."""
    return f"{base} {agent.instructions}" if agent.instructions else base


# A model caller: same signature as litellm-backed ``call_model``. Injectable so
# the CLI can run a mock (offline) squad without provider credentials.
Caller = Callable[..., str]


class Reporter(Protocol):
    def stage_start(self, stage: Stage, role: str, model: str) -> None: ...
    def stage_done(self, event: TranscriptEvent) -> None: ...


class NullReporter:
    """Reporter that does nothing (default for tests)."""

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        pass

    def stage_done(self, event: TranscriptEvent) -> None:
        pass


def _run_stage(
    turn: Turn,
    reporter: Reporter,
    run_cfg: RunConfig,
    caller: Caller,
    *,
    stage: Stage,
    role: str,
    model: str,
    system: str,
    prompt: str,
    structured: bool = False,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Run one model call, record an event, report it. Re-raises on failure.

    ``transform`` post-processes the raw model output before it is stored/returned (used to
    render the clusterer's JSON into the readable content map). A transform that raises aborts
    the turn like any stage failure.
    """
    reporter.stage_start(stage, role, model)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        output = caller(model, messages, run_cfg, role=role, structured=structured)
        if transform is not None:
            output = transform(output)
    except Exception as exc:  # noqa: BLE001 - record then re-raise to abort the turn
        event = turn.add(
            TranscriptEvent(
                turn_index=turn.index, stage=stage, role=role, model=model,
                prompt=prompt, task=turn.task, error=str(exc),
            )
        )
        reporter.stage_done(event)
        raise
    event = turn.add(
        TranscriptEvent(
            turn_index=turn.index, stage=stage, role=role, model=model,
            prompt=prompt, task=turn.task, output=output,
        )
    )
    reporter.stage_done(event)
    return output


def _content_map(clusters: list[dict], idmap: dict[str, dict]) -> str:
    """Render clusters into the judge's content map: facts only (support counts + conflicts).

    Any unit the clusterer left out becomes its own singleton so nothing is lost, and clusters
    that reference only unknown ids are dropped. Sorted by cross-response support, high first.
    """
    clustered = {mid for c in clusters for mid in c.get("member_ids", [])}
    for uid in idmap:
        if uid not in clustered:
            clusters.append({"label": idmap[uid]["text"], "member_ids": [uid], "conflict": None})

    def support(cluster: dict) -> list[str]:
        return sorted({idmap[m]["source"] for m in cluster.get("member_ids", []) if m in idmap})

    clusters = [c for c in clusters if support(c)]
    clusters.sort(key=lambda c: -len(support(c)))
    lines = []
    for cluster in clusters:
        tag = f"backed by {len(support(cluster))} response(s)"
        if cluster.get("conflict"):
            tag += f"; CONFLICT: {cluster['conflict']}"
        lines.append(f"- [{tag}] {cluster.get('label', '')}")
    return "\n".join(lines)


def run_turn(
    conversation: Conversation,
    task: str,
    config: SquadConfig,
    reporter: Reporter | None = None,
    *,
    caller: Caller = call_model,
) -> Turn:
    """Run one full ensemble turn, appending it to ``conversation``.

    ``caller`` defaults to the real LiteLLM-backed model call; inject a stand-in
    (e.g. ``mock_call_model``) to run offline.
    """
    reporter = reporter or NullReporter()
    run_cfg = config.run
    history = conversation.history_digest()
    turn = conversation.new_turn(task)

    # Each worker responds independently (blind to the others), the critic critiques that one
    # response, and the worker revises against its own critique.
    revised: list[str] = []
    for i, worker in enumerate(config.workers):
        role = f"worker_{i + 1}"
        proposal = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="propose", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.propose_prompt(task, history),
        )
        critique = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="critique", role=f"critic->{role}", model=config.critic.model,
            system=_system(prompts.CRITIC_SYSTEM, config.critic),
            prompt=prompts.critique_prompt(task, proposal),
        )
        revision = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="revise", role=role, model=worker.model,
            system=_system(prompts.WORKER_SYSTEM, worker),
            prompt=prompts.revise_prompt(task, proposal, critique),
        )
        revised.append(revision)

    # Synthesis (extract -> cluster -> judge). Extract each revised response into de-stylized
    # content units, pool them under de-identified labels, cluster equivalents (flagging
    # conflicts) into a content map, and let the judge write the final answer from that map.
    # Nothing polished reaches the judge, so it cannot clone the most fluent draft.
    idmap: dict[str, dict] = {}
    for i, revision in enumerate(revised):
        role = f"extractor->worker_{i + 1}"
        raw = _run_stage(
            turn, reporter, run_cfg, caller,
            stage="extract", role=role, model=config.extractor.model,
            system=_system(prompts.EXTRACT_SYSTEM, config.extractor),
            prompt=prompts.extract_prompt(task, revision), structured=True,
        )
        try:
            units = json_list(raw, "units")
        except ValueError as exc:
            raise LLMError(role, config.extractor.model, f"could not parse JSON: {exc}") from exc
        for unit in units:
            uid = f"u{len(idmap)}"
            idmap[uid] = {
                "source": _DRAFT_LABELS[i], "text": unit.get("text", ""),
                "kind": unit.get("kind", "claim"),
            }

    # De-identified, order-shuffled unit pool (unless disabled), so the clusterer groups on
    # content alone and no response is permanently first (LLM primacy bias).
    pool = list(idmap.items())
    if run_cfg.shuffle:
        random.shuffle(pool)
    lines = [f"{uid}: {unit['text']}  ({unit['kind']})" for uid, unit in pool]

    def _to_map(raw_clusters: str) -> str:
        try:
            clusters = json_list(raw_clusters, "clusters")
        except ValueError as exc:
            raise LLMError(
                "clusterer", config.clusterer.model, f"could not parse JSON: {exc}"
            ) from exc
        return _content_map(clusters, idmap)

    # The cluster stage stores/returns the rendered content map (facts only), so it displays as
    # a readable panel and feeds straight into the judge.
    content_map = _run_stage(
        turn, reporter, run_cfg, caller,
        stage="cluster", role="clusterer", model=config.clusterer.model,
        system=_system(prompts.CLUSTER_SYSTEM, config.clusterer),
        prompt=prompts.cluster_prompt(task, lines), structured=True, transform=_to_map,
    )

    _run_stage(
        turn, reporter, run_cfg, caller,
        stage="judge", role="judge", model=config.judge.model,
        system=_system(prompts.JUDGE_SYSTEM, config.judge),
        prompt=prompts.judge_prompt(task, content_map, history),
    )

    return turn


def run_quick(
    conversation: Conversation,
    task: str,
    config: SquadConfig,
    reporter: Reporter | None = None,
    *,
    caller: Caller = call_model,
) -> Turn:
    """Run one quick turn: the judge model answers directly, no ensemble."""
    reporter = reporter or NullReporter()
    history = conversation.history_digest()
    turn = conversation.new_turn(task)

    _run_stage(
        turn, reporter, config.run, caller,
        stage="reply", role="judge", model=config.judge.model,
        system=_system(prompts.QUICK_SYSTEM, config.judge),
        prompt=prompts.quick_prompt(task, history),
    )

    return turn
