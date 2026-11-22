"""The final model chosen by the DS: the report hook and the ``finalize`` job (ticket 30).

``make_ds_report_hook`` has the same ordered, idempotent steps as ``pipeline.flow.make_report_hook`` (persist,
link, one holdout read, suggestions, build the report, store it), but the chosen attempt is the one given by
``overrides.resolve_choice``: a valid ``choose_model`` record of the DS wins over ``view.reference()``, and the
program's own pick is kept and shown. It is passed through ``Hooks(report=...)`` (``run_pipeline`` honours a
caller's hook), so ``pipeline/`` is not edited. The registry report stays write-once: the DS choice is added to
the ``chosen`` object of the report (extra keys: ``chosen_by``, ``program_choice``, ``reason``) and to the marked
section of the Markdown file.

For a pick made AFTER the report, ``finalize_job`` persists the DS-chosen bundle, links the ``pipeline:``
artifact, writes ``out/chosen/<cycle>.json`` and refreshes the section. It reads NO holdout: the one session was
used on the program's pick, and the report says the holdout figure belongs to that pick.
"""

import json
from collections.abc import Callable
from pathlib import Path

from kaggle_agent.loop import CycleView
from kaggle_agent.overrides import records_for_cycle, resolve_choice
from kaggle_agent.pipeline import (
    FinalReport,
    build_report,
    derive_suggestions,
    evaluate_chosen_holdout,
    link_pipeline,
    store_report,
    store_suggestions,
)
from kaggle_agent.pipeline.flow import _persist

from .applied import applied_outcomes, mark
from .cockpit import CockpitFiles, read_jsonl
from .project import write_atomic
from .report_labels import collect_labels, override_comparisons, refresh_ds_section

CHOSEN_DIR = "chosen"
HOLDOUT_NOTE = ("the holdout figure of this report belongs to the program's own pick: the engine allows one "
                "holdout session per cycle")


def chosen_path(out: Path, cycle_id: str) -> Path:
    return Path(out) / CHOSEN_DIR / f"{cycle_id}.json"


def choice_records(files: CockpitFiles, cycle_id: str, first_cycle: str) -> list[dict]:
    records, _ = read_jsonl(files.decisions)
    return [r for r in records_for_cycle(records, cycle_id, first_cycle) if r.get("kind") == "choose_model"
            and isinstance(r.get("id"), str)]


def _chosen_dict(view: CycleView, detail) -> dict:
    last = detail.executions[-1]
    metric = next(m for m in last.metrics if m.partition == "validation")
    return {"attempt_id": detail.attempt.id,
            "bundle_id": last.config.get("bundle_id", detail.attempt.bundle_id),
            "hypothesis": detail.attempt.hypothesis, "validation_mean": metric.mean,
            "validation_dispersion": metric.dispersion, "executions": len(detail.executions)}


def usable_ids(view: CycleView, registry, cycle_id: str) -> set[str]:
    """Closed, succeeded, scored attempts that have a bundle (the ones a DS pick may name)."""
    out = set()
    for summary in registry.list_attempts(cycle_id):
        detail = view.detail(summary.attempt_id)
        if view.usable(detail) and (detail.executions[-1].config.get("bundle_id") or detail.attempt.bundle_id):
            out.add(detail.attempt.id)
    return out


def make_ds_report_hook(pctx, files: CockpitFiles, emitter, first_cycle: str, folder: Path) -> Callable:
    """``Hooks.report(ctx, stop_reason)`` with the DS choice (see the module docstring)."""

    def report_hook(ctx, stop_reason: str) -> None:
        registry = ctx.registry
        now = pctx.clock or ctx.now
        view = CycleView(registry, ctx.cycle_id, ctx.engine)
        program = view.reference()
        program_id = None if program is None else program.attempt.id
        chosen_id, record = resolve_choice(program_id, choice_records(files, ctx.cycle_id, first_cycle),
                                           usable_ids(view, registry, ctx.cycle_id),
                                           applied_outcomes(files))
        chosen = program if chosen_id == program_id else view.detail(chosen_id)
        persisted = store = None
        error_kind = error = None
        if chosen is not None:
            persisted, store, error_kind, error = _persist(ctx, pctx, chosen)
            if persisted is not None:
                link_pipeline(registry, chosen.attempt.id, persisted.root)
        outcome = evaluate_chosen_holdout(
            registry, ctx.engine, ctx.frame, chosen_id=None if chosen is None else chosen.attempt.id,
            persisted=persisted, store=store, now=now, persist_error=error,
            persist_kind=error_kind or "persist")
        store_suggestions(registry, derive_suggestions(registry, view, outcome, stop_reason))
        stored = registry.get_final_report(ctx.cycle_id)
        if stored is not None:
            report = FinalReport.from_dict(stored.content)
        else:
            report = build_report(
                registry, view, stop_reason=stop_reason, now=now(), outcome=outcome,
                persisted_path=None if persisted is None else str(persisted.root),
                db_path=pctx.db_path, data_path=pctx.data_path)
            if record is not None and chosen is not None:
                data = report.to_dict()
                data["chosen"] = {**_chosen_dict(view, chosen), "chosen_by": "the DS",
                                  "program_choice": program_id, "reason": record.get("reason", "")}
                report = FinalReport.from_dict(data)
        store_report(registry, report, pctx.out_dir)
        if record is not None:
            mark(files, emitter, record, "applied")
        labels = collect_labels(files, folder, registry, ctx.cycle_id, first_cycle)
        if labels or record is not None:
            refresh_ds_section(pctx.out_dir, ctx.cycle_id, labels,
                               None if record is None else {"attempt": chosen_id, "program_choice": program_id},
                               override_comparisons(registry, ctx.cycle_id))

    return report_hook


def read_chosen(out: Path, cycle_id: str) -> dict | None:
    path = chosen_path(out, cycle_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_chosen(out: Path, cycle_id: str, data: dict) -> None:
    write_atomic(chosen_path(out, cycle_id), json.dumps(data, indent=2, sort_keys=True))
