"""Orchestration.

Deep turn: each worker responds independently, the critic critiques each response, and the
worker revises. Then the revised answers are extracted into de-stylized content units,
clustered across responses into a content map, and the judge writes the single final answer
from that map (extract -> cluster -> judge). Quick turn: the judge answers directly, no
ensemble. Rendering-agnostic: progress is reported through a :class:`Reporter` so this module
never imports ``rich``.

The per-worker chains (and the per-worker extractions) are independent by design, so they run
in parallel on a thread pool -- model calls are I/O-bound, and bounded by ``run.max_parallel``.
Threading buys wall-clock speed only: workers stay blind to each other, and every bit of shared
bookkeeping (event recording, reporting, the transcript append inside reporters) is serialized
through one lock so the parallelism never leaks into the data.
"""

import random
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Callable, Protocol

from . import prompts
from .config import AgentConfig, RunConfig, SquadConfig
from .llm import LLMError, call_model, json_list
from .models import Conversation, Stage, TranscriptEvent, Turn


def _draft_label(index: int) -> str:
    """De-identified provenance label for worker ``index``: Draft A..Z, AA, AB, ...

    The clusterer and judge see these labels in the content map, never which model wrote
    what, so identity can't bias the grouping or the final answer. Spreadsheet-style
    letters, so any roster size works.
    """
    letters = ""
    n = index
    while n >= 0:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
    return f"Draft {letters}"


def _system(base: str, agent: AgentConfig) -> str:
    """Append an agent's optional per-config instructions to its base prompt."""
    return f"{base} {agent.instructions}" if agent.instructions else base


def _agent_run_cfg(run_cfg: RunConfig, agent: AgentConfig) -> RunConfig:
    """The run config for one agent's calls: its own max_tokens wins if set.

    Roles need different headroom -- a reasoning model clustering a wide roster's
    units burns far more of the budget than a worker writing one answer.
    """
    if agent.max_tokens is None:
        return run_cfg
    return run_cfg.model_copy(update={"max_tokens": agent.max_tokens})


# A model caller: same signature as litellm-backed ``call_model``. Injectable so
# the CLI can run a mock (offline) squad without provider credentials.
Caller = Callable[..., str]


class Reporter(Protocol):
    """Progress sink for a turn.

    Contract: calls are serialized (run_turn holds a lock around each call, so two never
    run concurrently), but with parallel worker chains they interleave across roles --
    another role's ``stage_start`` may arrive between one role's ``stage_start`` and its
    ``stage_done``. Implementations should track in-flight work keyed by (role, stage)
    and must not assume stages nest.
    """

    def stage_start(self, stage: Stage, role: str, model: str) -> None: ...
    def stage_done(self, event: TranscriptEvent) -> None: ...


class NullReporter:
    """Reporter that does nothing (default for tests)."""

    def stage_start(self, stage: Stage, role: str, model: str) -> None:
        pass

    def stage_done(self, event: TranscriptEvent) -> None:
        pass


class _CancelledChain(Exception):
    """A parallel task was skipped because an earlier task already failed (fail-fast)."""


def _run_parallel(tasks: list[Callable[[], object]], max_parallel: int) -> list:
    """Run tasks on a thread pool; return their results in task order.

    Fail-fast: the first failure marks the run aborted, so tasks that have not started yet
    are skipped entirely (no model calls, no events), tasks already in flight finish and
    their events still land in the transcript, and the failure with the lowest task index
    is re-raised (deterministic, unlike first-by-wall-clock). The wait loop polls instead
    of blocking so Ctrl-C reaches the main thread on Windows; an interrupt cancels the
    queued tasks the same way before propagating, so only in-flight calls run out.

    With ``max_parallel=1`` tasks run one at a time in order, reproducing serial behavior
    exactly (including a failure stopping everything after it).
    """
    aborted = threading.Event()

    def guarded(task: Callable[[], object]) -> Callable[[], object]:
        def run() -> object:
            if aborted.is_set():
                raise _CancelledChain()
            try:
                return task()
            except BaseException:
                aborted.set()
                raise

        return run

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(guarded(task)) for task in tasks]
        try:
            pending = set(futures)
            while pending:
                _, pending = wait(pending, timeout=0.5, return_when=FIRST_EXCEPTION)
        except BaseException:  # KeyboardInterrupt: drop the queue, let in-flight finish
            aborted.set()
            for future in futures:
                future.cancel()
            raise

    failures = [
        future.exception()
        for future in futures
        if not future.cancelled() and future.exception() is not None
    ]
    real_failures = [exc for exc in failures if not isinstance(exc, _CancelledChain)]
    if real_failures:
        raise real_failures[0]
    return [future.result() for future in futures]


def _run_stage(
    turn: Turn,
    reporter: Reporter,
    run_cfg: RunConfig,
    caller: Caller,
    *,
    lock: threading.Lock,
    stage: Stage,
    role: str,
    model: str,
    system: str,
    prompt: str,
    structured: bool = False,
    transform: Callable[[str], str] | None = None,
) -> str:
    """Run one model call, record an event, report it. Re-raises on failure.

    The model call runs outside ``lock`` so parallel stages genuinely overlap; everything
    shared -- reporting and recording the event (which is also where reporters append the
    transcript JSONL) -- happens inside it, on the success and error paths alike, so the
    transcript and the reporter always see one whole event at a time.

    ``transform`` post-processes the raw model output before it is stored/returned (used to
    render the clusterer's JSON into the readable content map). A transform that raises aborts
    the turn like any stage failure.
    """
    with lock:
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
        with lock:
            event = turn.add(
                TranscriptEvent(
                    turn_index=turn.index, stage=stage, role=role, model=model,
                    prompt=prompt, task=turn.task, error=str(exc),
                )
            )
            reporter.stage_done(event)
        raise
    with lock:
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

    Each cluster carries its member units' FULL de-stylized texts under the label -- a label
    alone starves the judge of the specifics the extractor preserved, and a starved judge
    invents them (measured: it confabulated another concept's palette onto the winning one).
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
        label = cluster.get("label", "")
        lines.append(f"- [{tag}] {label}")
        members = [idmap[m] for m in cluster.get("member_ids", []) if m in idmap]
        if len(members) == 1 and members[0]["text"] == label:
            continue  # synthetic singleton: the label already is the unit's full text
        lines.extend(f"    * {unit['text']}" for unit in members)
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
    lock = threading.Lock()
    max_parallel = max(1, run_cfg.max_parallel)

    # Each worker responds independently (blind to the others), the critic critiques that one
    # response, and the worker revises against its own critique. The chains share nothing, so
    # they run as one parallel wave -- independence is what makes the parallelism safe.
    def worker_chain(index: int, worker: AgentConfig) -> Callable[[], str]:
        def chain() -> str:
            role = f"worker_{index + 1}"
            worker_cfg = _agent_run_cfg(run_cfg, worker)
            proposal = _run_stage(
                turn, reporter, worker_cfg, caller, lock=lock,
                stage="propose", role=role, model=worker.model,
                system=_system(prompts.WORKER_SYSTEM, worker),
                prompt=prompts.propose_prompt(task, history),
            )
            critique = _run_stage(
                turn, reporter, _agent_run_cfg(run_cfg, config.critic), caller, lock=lock,
                stage="critique", role=f"critic->{role}", model=config.critic.model,
                system=_system(prompts.CRITIC_SYSTEM, config.critic),
                prompt=prompts.critique_prompt(task, proposal),
            )
            return _run_stage(
                turn, reporter, worker_cfg, caller, lock=lock,
                stage="revise", role=role, model=worker.model,
                system=_system(prompts.WORKER_SYSTEM, worker),
                prompt=prompts.revise_prompt(task, proposal, critique),
            )

        return chain

    revised: list[str] = _run_parallel(
        [worker_chain(i, worker) for i, worker in enumerate(config.workers)], max_parallel
    )

    # Synthesis (extract -> cluster -> judge). Extract each revised response into de-stylized
    # content units, pool them under de-identified labels, cluster equivalents (flagging
    # conflicts) into a content map, and let the judge write the final answer from that map.
    # Nothing polished reaches the judge, so it cannot clone the most fluent draft.
    # Each extraction sees only its own revision, so this is a second parallel wave.
    def extract_units(index: int, revision: str) -> Callable[[], list[dict]]:
        def extract() -> list[dict]:
            role = f"extractor->worker_{index + 1}"
            raw = _run_stage(
                turn, reporter, _agent_run_cfg(run_cfg, config.extractor), caller, lock=lock,
                stage="extract", role=role, model=config.extractor.model,
                system=_system(prompts.EXTRACT_SYSTEM, config.extractor),
                prompt=prompts.extract_prompt(task, revision), structured=True,
            )
            try:
                return json_list(raw, "units")
            except ValueError as exc:
                raise LLMError(
                    role, config.extractor.model, f"could not parse JSON: {exc}"
                ) from exc

        return extract

    units_per_worker: list[list[dict]] = _run_parallel(
        [extract_units(i, revision) for i, revision in enumerate(revised)], max_parallel
    )

    # Pool the units on the main thread, in worker order, so unit ids and draft labels are
    # deterministic no matter how the parallel waves interleaved.
    idmap: dict[str, dict] = {}
    for index, units in enumerate(units_per_worker):
        for unit in units:
            uid = f"u{len(idmap)}"
            idmap[uid] = {
                "source": _draft_label(index), "text": unit.get("text", ""),
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
        turn, reporter, _agent_run_cfg(run_cfg, config.clusterer), caller, lock=lock,
        stage="cluster", role="clusterer", model=config.clusterer.model,
        system=_system(prompts.CLUSTER_SYSTEM, config.clusterer),
        prompt=prompts.cluster_prompt(task, lines), structured=True, transform=_to_map,
    )

    _run_stage(
        turn, reporter, _agent_run_cfg(run_cfg, config.judge), caller, lock=lock,
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
        turn, reporter, _agent_run_cfg(config.run, config.judge), caller, lock=threading.Lock(),
        stage="reply", role="judge", model=config.judge.model,
        system=_system(prompts.QUICK_SYSTEM, config.judge),
        prompt=prompts.quick_prompt(task, history),
    )

    return turn
