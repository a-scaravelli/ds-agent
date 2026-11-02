"""The end-to-end flow of a cycle: start, rounds, holdout, report and suggestions (design step 11).

``run_pipeline`` runs the unchanged ``CycleMachine`` and installs one hook, ``Hooks.report``. At
REPORTING that hook, in order: chooses the bundle (``CycleView.reference()``), persists its
model, links the persisted bundle to the chosen attempt (the ``pipeline:<ref>`` artifact the web
interface reads, so there is a path from the attempt to the corresponding pipeline), evaluates the
holdout once, derives and stores the suggestions, builds the report and stores it. Every step
reads its state from the registry or the persisted folder and is idempotent, so a crash in
REPORTING resumes without a second holdout read and without duplicate records.

Not here: the orchestrator that starts the next cycle by itself (step 12). A cycle started
from a suggestion is linked to its origin only; the best model and the history of the origin are
not copied into the new cycle.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kaggle_agent.budget import CycleLimits
from kaggle_agent.bundle import BundleError, BundleRunError, load_bundle
from kaggle_agent.evaluation import EvaluationError
from kaggle_agent.loop import CycleContext, CycleMachine, CycleOutcome, CycleView, Hooks
from kaggle_agent.loop.states import PLANNING
from kaggle_agent.proposals import ShortPlan, ValidationError
from kaggle_agent.registry import Cycle, Registry

from .errors import PipelineError
from .holdout import evaluate_chosen_holdout, one_line, run_reason
from .persisted import ModelStore, PersistedBundle, persist_bundle
from .report import FinalReport, build_report, store_report
from .suggestions import derive_suggestions, store_suggestions

PIPELINE_ARTIFACT = "pipeline:"  # the convention of kaggle_agent.web.views.PIPELINE_PREFIX


@dataclass(frozen=True)
class PipelineContext:
    """What the report step needs besides the cycle context.

    ``out_dir`` receives ``persisted/<bundle>/`` and ``reports/<cycle>.md``.
    ``model_store_factory(loaded)`` builds the ``ModelStore`` of a loaded bundle (a real
    ``BundleRunner`` in production, ``FakeModelStore`` in tests). ``clock`` defaults to the
    clock of the cycle context. ``db_path`` and ``data_path`` only fill the reproduce command
    in the report.
    """

    out_dir: Path
    model_store_factory: Callable[..., ModelStore]
    clock: Callable[[], datetime] | None = None
    db_path: str | None = None
    data_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "out_dir", Path(self.out_dir).resolve())


@dataclass(frozen=True)
class PipelineResult:
    """How the cycle ended and the stored report (None if the cycle did not reach REPORTING)."""

    outcome: CycleOutcome
    report: FinalReport | None = field(default=None)


def start_cycle(registry: Registry, cycle_id: str, plan: ShortPlan, limits: CycleLimits,
                project_id: str | None = None, from_suggestion: str | None = None) -> Cycle:
    """Create a cycle in PLANNING with the rounds, stagnation and fix caps of ``limits``.

    With ``from_suggestion`` the suggestion must exist (``NotFoundError`` otherwise, and nothing
    is created) and the new cycle is linked to it with ``link_cycle``. That link only records
    the origin: the best model and the history of the suggestion's cycle are NOT carried into
    the new cycle.
    """
    if from_suggestion is not None:
        registry.get_suggestion(from_suggestion)
    cycle = Cycle(
        id=cycle_id, plan=plan, iterations=limits.rounds, stagnation_k=limits.stagnation_rounds,
        state=PLANNING, max_fix_retries=limits.fix_retries, project_id=project_id,
    )
    registry.add_cycle(cycle)
    if from_suggestion is not None:
        registry.link_cycle(from_suggestion, cycle_id)
    return cycle


def link_pipeline(registry: Registry, attempt_id: str, persisted_dir: Path | str) -> None:
    """Attach ``pipeline:<persisted_dir>`` to the last execution of the attempt; idempotent."""
    registry.add_execution_artifact(attempt_id, f"{PIPELINE_ARTIFACT}{persisted_dir}")


def _persist(ctx: CycleContext, pctx: PipelineContext, chosen):
    """Persist the chosen bundle; returns (persisted, store, failure kind, error text)."""
    registry = ctx.registry
    try:
        last = chosen.executions[-1]
        bundle_id = last.config.get("bundle_id") or chosen.attempt.bundle_id
        record = registry.get_bundle(bundle_id)
        loaded = load_bundle(Path(record.reference))
        if loaded.folder_hash != last.code_hash:
            raise PipelineError("the bundle folder differs from the code that was scored")
        store = pctx.model_store_factory(loaded)
        metric = next(m for m in last.metrics if m.partition == "validation")
        persisted = persist_bundle(
            pctx.out_dir, loaded, store, attempt_id=chosen.attempt.id, cycle_id=ctx.cycle_id,
            engine=ctx.engine, frame=ctx.frame, data_hash=ctx.data_hash,
            validation_mean=metric.mean, validation_dispersion=metric.dispersion,
        )
        return persisted, store, None, None
    except BundleRunError as error:
        return None, None, error.failure_kind, run_reason(error)
    except EvaluationError as error:  # the protocol hash changed
        return None, None, "protocol", one_line(error)
    except (PipelineError, BundleError, ValidationError, OSError) as error:
        return None, None, "persist", one_line(error)


def make_report_hook(pctx: PipelineContext) -> Callable[[CycleContext, str], None]:
    """The callable for ``Hooks.report(ctx, stop_reason)``: the whole REPORTING step."""

    def report_hook(ctx: CycleContext, stop_reason: str) -> None:
        registry = ctx.registry
        now = pctx.clock or ctx.now
        view = CycleView(registry, ctx.cycle_id, ctx.engine)
        chosen = view.reference()
        persisted: PersistedBundle | None = None
        store = None
        error_kind = error = None
        if chosen is not None:
            persisted, store, error_kind, error = _persist(ctx, pctx, chosen)
            if persisted is not None:
                link_pipeline(registry, chosen.attempt.id, persisted.root)
        outcome = evaluate_chosen_holdout(
            registry, ctx.engine, ctx.frame, chosen_id=None if chosen is None else chosen.attempt.id,
            persisted=persisted, store=store, now=now, persist_error=error,
            persist_kind=error_kind or "persist",
        )
        store_suggestions(registry, derive_suggestions(registry, view, outcome, stop_reason))
        stored = registry.get_final_report(ctx.cycle_id)
        if stored is not None:
            report = FinalReport.from_dict(stored.content)
        else:
            report = build_report(
                registry, view, stop_reason=stop_reason, now=now(), outcome=outcome,
                persisted_path=None if persisted is None else str(persisted.root),
                db_path=pctx.db_path, data_path=pctx.data_path,
            )
        store_report(registry, report, pctx.out_dir)

    return report_hook


def run_pipeline(ctx: CycleContext, pctx: PipelineContext) -> PipelineResult:
    """Run the cycle to its end with the pipeline report hook; returns the outcome and the report.

    The hook is installed with ``dataclasses.replace`` unless ``ctx.hooks.report`` is already set,
    in which case the caller's hook is used. The ``CycleMachine`` is not modified.
    """
    if ctx.hooks.report is None:
        ctx = dataclasses.replace(ctx, hooks=dataclasses.replace(ctx.hooks, report=make_report_hook(pctx)))
    outcome = CycleMachine(ctx).run()
    stored = ctx.registry.get_final_report(ctx.cycle_id)
    return PipelineResult(outcome, None if stored is None else FinalReport.from_dict(stored.content))
