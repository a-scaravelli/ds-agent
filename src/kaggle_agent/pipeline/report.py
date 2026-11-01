"""The final report of a cycle (design sections 7 and 15, step 11).

``FinalReport`` is a frozen record built only from the registry and the derived ``CycleView``: the
chosen bundle with its frozen-protocol metric, the holdout outcome, the history of rounds with their
candidates, comparisons, reviewer text and guidance, what worked and what failed, the suggestions
for the next cycle, the persisted bundle and the command that reproduces the reported metric.
``render_text`` gives a Markdown view; ``store_report`` writes the record to the registry and the
Markdown to ``<out>/reports/<cycle>.md``, once.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kaggle_agent.loop import CycleView
from kaggle_agent.proposals import ValidationError
from kaggle_agent.proposals.errors import check_keys, loads_object, require_text
from kaggle_agent.registry import HoldoutOutcome, Registry, ReportRecord

REPORTS_DIR = "reports"
_FIELDS = (
    "cycle_id", "created_at", "stop_reason", "metric", "direction", "protocol_hash", "chosen",
    "baseline_score", "holdout", "rounds", "worked", "failures", "suggestions", "persisted_path",
    "reproduce_command",
)


def _check_json(name: str, value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError(name, f"must be JSON-compatible: {error}") from None


@dataclass(frozen=True)
class FinalReport:
    """The report of a finished cycle; every part is JSON-compatible.

    ``chosen`` (attempt_id, bundle_id, hypothesis, validation_mean, validation_dispersion,
    executions) is None when no attempt was usable. ``holdout`` holds the status, score, rows,
    failure kind and reason. Each item of ``rounds`` has the round id, index, tasks, reviewer
    text, candidates, comparison, improved flag and guidance. ``failures`` counts failed attempts
    by failure kind; ``worked`` lists the improving rounds' best candidates; ``suggestions`` lists
    ``{id, text}``.
    """

    cycle_id: str
    created_at: str
    stop_reason: str
    metric: str
    direction: str
    protocol_hash: str
    holdout: dict
    chosen: dict | None = None
    baseline_score: float | None = None
    rounds: list = field(default_factory=list)
    worked: list = field(default_factory=list)
    failures: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    persisted_path: str | None = None
    reproduce_command: str | None = None

    def __post_init__(self) -> None:
        for name in ("cycle_id", "created_at", "stop_reason", "metric", "direction", "protocol_hash"):
            require_text(name, getattr(self, name))
        for name, kind in (("holdout", dict), ("rounds", list), ("worked", list), ("failures", dict),
                           ("suggestions", list)):
            if not isinstance(getattr(self, name), kind):
                raise ValidationError(name, f"must be a {kind.__name__}")
        if self.chosen is not None and not isinstance(self.chosen, dict):
            raise ValidationError("chosen", "must be an object or None")
        for name in ("persisted_path", "reproduce_command"):
            if getattr(self, name) is not None:
                require_text(name, getattr(self, name))
        for name in _FIELDS:
            _check_json(name, getattr(self, name))

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "FinalReport":
        check_keys("report", data, _FIELDS)
        kwargs = {name: data[name] for name in _FIELDS if name in data}
        for name in ("cycle_id", "created_at", "stop_reason", "metric", "direction", "protocol_hash"):
            kwargs.setdefault(name, None)
        kwargs.setdefault("holdout", None)
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FinalReport":
        return cls.from_dict(loads_object("report", text))


def holdout_dict(outcome: HoldoutOutcome) -> dict:
    """The part of a ``HoldoutOutcome`` that the report keeps."""
    return {name: getattr(outcome, name) for name in
            ("status", "score", "rows", "failure_kind", "reason", "attempt_id", "bundle_id")}


def reproduce_command(persisted_path: str, db_path: str | None, data_path: str | None) -> str:
    """The one command that reproduces the reported frozen-protocol metric from the persisted bundle."""
    return (f"python -m kaggle_agent.pipeline reproduce --persisted {persisted_path} "
            f"--db {db_path or '<registry.db>'} --data {data_path or '<data.csv>'}")


def _candidate(view: CycleView, detail) -> dict:
    return {
        "id": detail.attempt.id,
        "hypothesis": detail.attempt.hypothesis,
        "outcome": detail.outcome or "open",
        "score": view.score(detail),
        "failure_kind": detail.failure_kind,
    }


def _rounds(registry: Registry, view: CycleView) -> list:
    reports = registry.list_reviewer_reports(view.cycle_id)
    guidance = registry.list_guidance(view.cycle_id)
    out = []
    for state in view.round_states():
        round_id = state.round_id
        texts = [r.text for r in reports if r.round_id == round_id]
        comparison = state.comparison
        out.append({
            "round_id": round_id,
            "index": state.index,
            "tasks": list(state.round.round.tasks),
            "reviewer": texts[-1] if texts else None,
            "candidates": [_candidate(view, d) for d in state.details],
            "comparison": None if comparison is None else {
                "difference": comparison.difference, "threshold": comparison.threshold,
                "is_real": comparison.is_real,
            },
            "improved": state.improved,
            "guidance": [{"id": g.id, "text": g.text} for g in guidance if g.round_id == round_id],
        })
    return out


def build_report(registry: Registry, view: CycleView, *, stop_reason: str, now: datetime,
                 outcome: HoldoutOutcome, persisted_path: str | None = None,
                 db_path: str | None = None, data_path: str | None = None) -> FinalReport:
    """Build the report from the registry and the derived view; nothing is written.

    The chosen bundle is ``view.reference()`` (the best so far after the complete rounds).
    """
    cycle = registry.get_cycle(view.cycle_id)
    chosen_detail = view.reference()
    chosen = None
    if chosen_detail is not None:
        last = chosen_detail.executions[-1]
        metric = next(m for m in last.metrics if m.partition == "validation")
        chosen = {
            "attempt_id": chosen_detail.attempt.id,
            "bundle_id": last.config.get("bundle_id", chosen_detail.attempt.bundle_id),
            "hypothesis": chosen_detail.attempt.hypothesis,
            "validation_mean": metric.mean,
            "validation_dispersion": metric.dispersion,
            "executions": len(chosen_detail.executions),
        }
    baseline = view.baseline()
    rounds = _rounds(registry, view)
    mine = {s.attempt_id for s in registry.list_attempts(view.cycle_id)}
    counts: dict = {}
    for failure in registry.list_failures():
        if failure.attempt_id in mine:
            counts[failure.failure_kind] = counts.get(failure.failure_kind, 0) + 1
    worked = []
    for item, state in zip(rounds, view.round_states()):
        if state.complete and state.improved and state.best is not None:
            worked.append({"round_id": item["round_id"], **_candidate(view, state.best)})
    suggestions = [{"id": s.id, "text": s.text} for s in registry.list_suggestions()
                   if s.origin_cycle == view.cycle_id]
    return FinalReport(
        cycle_id=view.cycle_id, created_at=now.isoformat(), stop_reason=stop_reason,
        metric=view.protocol.metric, direction=view.protocol.direction,
        protocol_hash=cycle.protocol_hash or "not frozen", holdout=holdout_dict(outcome),
        chosen=chosen, baseline_score=None if baseline is None else view.score(baseline),
        rounds=rounds, worked=worked, failures=dict(sorted(counts.items())), suggestions=suggestions,
        persisted_path=persisted_path,
        reproduce_command=(reproduce_command(persisted_path, db_path, data_path)
                           if persisted_path else None),
    )


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_text(report: FinalReport) -> str:
    """A readable Markdown form of the report."""
    lines = [f"# Final report of cycle {report.cycle_id}", "",
             f"- Created: {report.created_at}",
             f"- Stop reason: {report.stop_reason}",
             f"- Metric: {report.metric} ({report.direction}), protocol hash {report.protocol_hash[:12]}",
             f"- Baseline score: {_number(report.baseline_score)}", "", "## Chosen bundle", ""]
    chosen = report.chosen
    if chosen is None:
        lines.append("No usable attempt: no bundle was chosen.")
    else:
        lines += [f"- Bundle: {chosen['bundle_id']} (attempt {chosen['attempt_id']})",
                  f"- Hypothesis: {chosen['hypothesis']}",
                  f"- Frozen-protocol validation {report.metric}: {_number(chosen['validation_mean'])} "
                  f"(dispersion {_number(chosen['validation_dispersion'])})",
                  f"- Executions: {chosen['executions']}"]
    holdout = report.holdout
    lines += ["", "## Holdout", "", f"- Status: {holdout['status']}"]
    if holdout["status"] == "succeeded":
        lines += [f"- Holdout {report.metric}: {_number(holdout['score'])} on {holdout['rows']} rows"]
    else:
        lines += [f"- Reason: {holdout['reason']}"]
        if holdout.get("failure_kind"):
            lines += [f"- Failure kind: {holdout['failure_kind']}"]
    lines += ["", "## Rounds", ""]
    for item in report.rounds:
        comparison = item["comparison"]
        verdict = "no comparison" if comparison is None else (
            f"difference {comparison['difference']:.4f} vs threshold {comparison['threshold']:.4f}: "
            + ("real improvement" if comparison["is_real"] else "not a real improvement"))
        lines += [f"### Round {item['index']} ({item['round_id']})", "",
                  f"- Tasks: {'; '.join(item['tasks'])}", f"- Comparison: {verdict}",
                  f"- Reviewer: {item['reviewer'] or 'none'}"]
        for candidate in item["candidates"]:
            kind = f" [{candidate['failure_kind']}]" if candidate["failure_kind"] else ""
            lines.append(f"- Candidate {candidate['id']}: {candidate['outcome']}{kind}, score "
                         f"{_number(candidate['score'])} - {candidate['hypothesis']}")
        for guidance in item["guidance"]:
            lines.append(f"- Guidance of the lead: {guidance['text']}")
        lines.append("")
    lines += ["## What worked", ""]
    lines += [f"- {w['round_id']}: {w['id']} ({_number(w['score'])}) - {w['hypothesis']}"
              for w in report.worked] or ["- nothing beat the baseline by a real margin"]
    lines += ["", "## What failed", ""]
    lines += [f"- {kind}: {count}" for kind, count in report.failures.items()] or ["- no failed attempt"]
    lines += ["", "## Suggestions for the next cycle", ""]
    lines += [f"- {s['id']}: {s['text']}" for s in report.suggestions] or ["- none"]
    lines += ["", "## Persisted bundle and reproduction", ""]
    if report.persisted_path:
        lines += [f"- Persisted bundle: {report.persisted_path}",
                  f"- Reproduce the frozen-protocol metric: `{report.reproduce_command}`"]
    else:
        lines += ["- No persisted bundle."]
    return "\n".join(lines) + "\n"


def report_path(out_dir: Path | str, cycle_id: str) -> Path:
    """Where the Markdown report of a cycle is written."""
    return Path(out_dir) / REPORTS_DIR / f"{cycle_id}.md"


def store_report(registry: Registry, report: FinalReport, out_dir: Path | str) -> FinalReport:
    """Write the report to the registry and ``<out>/reports/<cycle>.md``; a no-op when it exists.

    If the registry already holds a report it is returned and nothing is rewritten (a missing
    Markdown file is recreated from the stored one).
    """
    stored = registry.get_final_report(report.cycle_id)
    if stored is not None:
        report = FinalReport.from_dict(stored.content)
    else:
        registry.add_final_report(ReportRecord(report.cycle_id, report.created_at, report.to_dict()))
    path = report_path(out_dir, report.cycle_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_text(report), encoding="utf-8")
    return report
