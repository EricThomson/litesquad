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
from concurrent.futures import ThreadPoolExecutor, wait
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
    """The run config for one agent's calls: its own max_tokens and reasoning_effort win if set.

    Roles need different headroom -- a reasoning model clustering a wide roster's
    units burns far more of the budget than a worker writing one answer.
    """
    agent_overrides = {
        "max_tokens": agent.max_tokens,
        "reasoning_effort": agent.reasoning_effort,
    }
    set_overrides = {key: value for key, value in agent_overrides.items() if value is not None}
    if not set_overrides:
        return run_cfg
    return run_cfg.model_copy(update=set_overrides)


# A model caller: same signature as the OpenRouter-backed ``call_model``. Injectable so
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


# A turn carries on when some workers fail, but not with fewer than this many surviving
# (or all of them, for a roster smaller than this): below it there is too little for
# cross-response recurrence to mean anything.
MIN_SURVIVING_WORKERS = 3


class TooFewWorkersError(RuntimeError):
    """Too many workers failed for the turn to continue."""


def _run_parallel(tasks: list[Callable[[], object]], max_parallel: int) -> list:
    """Run tasks on a thread pool; return each task's outcome, in task order.

    A task that raised an ``Exception`` is represented in the result list by that exception
    instead of a value, so one failure never stops the others -- the caller decides what
    to do with the survivors. The wait loop polls instead of blocking so Ctrl-C reaches the
    main thread on Windows; an interrupt cancels the queued tasks before propagating, so
    only in-flight calls run out.

    With ``max_parallel=1`` tasks run one at a time, in order.
    """
    interrupted = threading.Event()

    def guarded(task: Callable[[], object]) -> Callable[[], object]:
        def run() -> object:
            if interrupted.is_set():
                return None  # skipped after Ctrl-C; the interrupt propagates below
            return task()

        return run

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(guarded(task)) for task in tasks]
        try:
            pending = set(futures)
            while pending:
                _, pending = wait(pending, timeout=0.5)
        except BaseException:  # KeyboardInterrupt: drop the queue, let in-flight finish
            interrupted.set()
            for future in futures:
                future.cancel()
            raise

    outcomes = []
    for future in futures:
        failure = future.exception()
        if failure is not None and not isinstance(failure, Exception):
            raise failure  # e.g. KeyboardInterrupt inside a task: never swallow it
        outcomes.append(failure if failure is not None else future.result())
    return outcomes


def _survivors(
    outcomes_by_worker: list[tuple[int, object]], stage_name: str, roster_size: int
) -> list[tuple[int, object]]:
    """Keep the (worker index, result) pairs whose task succeeded.

    ``roster_size`` is the full roster, so workers dropped at an earlier stage still count
    against the minimum.

    Raises :class:`TooFewWorkersError` if fewer than :data:`MIN_SURVIVING_WORKERS` survive
    (or fewer than the whole roster, when the roster is smaller than that). The failures
    themselves were already recorded as error events by the stage that failed.

    Only model failures (:class:`LLMError`) drop a worker. Any other exception is a bug in
    litesquad, so it is re-raised with its original traceback instead of being hidden
    behind a dropped worker.
    """
    for _, outcome in outcomes_by_worker:
        if isinstance(outcome, Exception) and not isinstance(outcome, LLMError):
            raise outcome
    survivors = [(index, outcome) for index, outcome in outcomes_by_worker
                 if not isinstance(outcome, LLMError)]
    required = min(MIN_SURVIVING_WORKERS, roster_size)
    if len(survivors) < required:
        failures = "\n".join(
            f"  {outcome}" for _, outcome in outcomes_by_worker if isinstance(outcome, LLMError)
        )
        raise TooFewWorkersError(
            f"only {len(survivors)} of {roster_size} workers survived {stage_name} "
            f"(need {required}):\n{failures}"
        )
    return survivors


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
    validate the extractor's JSON, and to render the clusterer's JSON into the readable
    content map). A transform that raises fails the stage exactly like a failed call.
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

    ``caller`` defaults to the real OpenRouter-backed model call; inject a stand-in
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

    # A worker whose chain fails (any of its three calls) is dropped and the turn carries on
    # with the rest; its error is already in the transcript.
    roster_size = len(config.workers)
    chain_outcomes = _run_parallel(
        [worker_chain(i, worker) for i, worker in enumerate(config.workers)], max_parallel
    )
    revised: list[tuple[int, str]] = _survivors(
        list(enumerate(chain_outcomes)), "propose/critique/revise", roster_size
    )

    # Synthesis (extract -> cluster -> judge). Extract each revised response into de-stylized
    # content units, pool them under de-identified labels, cluster equivalents (flagging
    # conflicts) into a content map, and let the judge write the final answer from that map.
    # Nothing polished reaches the judge, so it cannot clone the most fluent draft.
    # Each extraction sees only its own revision, so this is a second parallel wave.
    def extract_units(index: int, revision: str) -> Callable[[], list[dict]]:
        def extract() -> list[dict]:
            role = f"extractor->worker_{index + 1}"

            # Checked inside the stage (as its transform), so an unparseable reply is recorded
            # as this stage's error and drops the worker visibly, rather than failing after a
            # success event was already recorded.
            def check_parses(raw: str) -> str:
                try:
                    json_list(raw, "units")
                except ValueError as exc:
                    raise LLMError(
                        role, config.extractor.model, f"could not parse JSON: {exc}"
                    ) from exc
                return raw

            raw = _run_stage(
                turn, reporter, _agent_run_cfg(run_cfg, config.extractor), caller, lock=lock,
                stage="extract", role=role, model=config.extractor.model,
                system=_system(prompts.EXTRACT_SYSTEM, config.extractor),
                prompt=prompts.extract_prompt(task, revision), structured=True,
                transform=check_parses,
            )
            return json_list(raw, "units")

        return extract

    extract_outcomes = _run_parallel(
        [extract_units(index, revision) for index, revision in revised], max_parallel
    )
    # A failed extraction drops its worker too: without units, nothing of it reaches the judge.
    surviving_indices = [index for index, _ in revised]
    units_per_worker: list[tuple[int, list[dict]]] = _survivors(
        list(zip(surviving_indices, extract_outcomes)), "extraction", roster_size
    )

    # Pool the units on the main thread, in worker order, so unit ids and draft labels are
    # deterministic no matter how the parallel waves interleaved.
    idmap: dict[str, dict] = {}
    for index, units in units_per_worker:
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
