"""Read-only data layer of the cockpit: plain dicts for the JSON API and the HTML pages.

Both the ``/api`` routes and the pages call the functions of this module, so a different front
end could replace the pages and keep the data. Every function takes a ``Source`` and returns
JSON-safe values; only read methods of the registry are called and nothing is computed that a
score depends on (no holdout number is ever produced here, only what the registry recorded).

Old registry files: every section goes through ``Source.section``. A missing table or column
(``sqlite3.Error`` or ``RegistryError``) gives the empty default and a ``warnings`` entry, so the
page renders with a "not available in this registry file" note instead of failing. An unknown id
still raises ``NotFoundError`` (404) and an invalid filter raises ``ValidationError`` (400).

The only function of the web package that opens a file is ``read_project_file`` (the structural
guard of the tests relies on it). The web never reads the bundle folders: a bundle is shown only
as the record the registry holds.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from kaggle_agent.proposals import ValidationError, validate_role
from kaggle_agent.registry import OUTCOMES, NotFoundError, RegistryError

from . import store, views
from .serialize import jsonable

PROJECT_FILE = "project.json"
PARTITION = "validation"
DIRECTIONS = ("max", "min")
ROLE_NAMES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer")
_ROLE_OUTPUT_ORDER = ("data", "feature_engineering", "modeling", "coordinator", "reviewer")


def read_project_file(project_dir) -> dict | None:
    """The parsed ``project.json`` of the project folder, or None when missing or unreadable.

    The one place of the web package that opens a file. ``project_dir`` may be None.
    """
    if project_dir is None:
        return None
    try:
        text = (Path(project_dir) / PROJECT_FILE).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Source:
    """A registry, the project folder (for ``project.json`` only) and the warnings of one request."""

    def __init__(self, registry, project_dir=None) -> None:
        self.registry = registry
        self.project_dir = project_dir
        self.warnings: list[dict] = []
        self._cache: dict = {}

    def section(self, name: str, fn, default):
        """``fn()``, or ``default`` plus a warning when the registry file lacks what it needs."""
        try:
            return fn()
        except NotFoundError:
            raise
        except (sqlite3.Error, RegistryError) as exc:
            if not any(w["section"] == name for w in self.warnings):
                self.warnings.append(
                    {"section": name, "reason": f"not available in this registry file ({exc})"}
                )
            return default

    def notes(self) -> dict:
        """Notes and tags of the project, latest record wins (read from ``notes.jsonl``)."""
        return self.cached("notes", lambda: notes_index(self.project_dir))

    def cached(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]


# -- small helpers ------------------------------------------------------------------------


def _number(name: str, value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        raise ValidationError(name, "must be a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValidationError(name, "must be a finite number")
    return number


def _direction(value) -> str | None:
    text = str(value).lower() if value is not None else ""
    if text in ("max", "maximize", "maximise"):
        return "max"
    if text in ("min", "minimize", "minimise"):
        return "min"
    return None


def setup_round_id(cycle_id: str) -> str:
    return f"{cycle_id}-R0"


def _status(state: str, stop_reason: str | None, chosen: bool) -> str:
    """Design token of the state: succeeded, failed, running, waiting or unknown."""
    if state == "STOPPED":
        if stop_reason == "consecutive_failures" or (stop_reason is None and not chosen):
            return "failed"
        return "succeeded"
    if state.startswith("AWAITING"):
        return "waiting"
    if state in ("PLANNING", "SPLITTING", "BASELINE", "ITERATING", "REPORTING"):
        return "running"
    return "unknown"


# -- run core (shared by the list, the summary and the chart) --------------------------------


def _core(src: Source, run_id: str, best_param: str | None = None) -> dict:
    def build():
        reg = src.registry
        cycle = reg.get_cycle(run_id)
        sec = src.section
        attempts = sec("attempts", lambda: reg.list_attempts(run_id), [])
        rounds = sec("rounds", lambda: reg.list_rounds_with_candidates(run_id), [])
        points = sec("metrics", lambda: reg.metric_series(run_id, PARTITION), [])
        comparisons = sec("comparisons", lambda: reg.list_comparisons(run_id), [])
        report = sec("final_report", lambda: reg.get_final_report(run_id), None)
        outcome = sec("holdout", lambda: reg.get_holdout_outcome(run_id), None)
        reads = sec("holdout", lambda: reg.list_holdout_reads(run_id), [])
        submissions = sec("submissions", lambda: reg.list_submissions(run_id), [])
        mean_of = {p.attempt_id: p for p in points}
        protocol = _protocol(src, cycle, comparisons, mean_of, best_param)
        direction = protocol["direction"]
        outcome_of = {a.attempt_id: a.outcome for a in attempts}
        succeeded = [
            p for p in points if outcome_of.get(p.attempt_id) == "succeeded"
        ]
        best = None
        for p in succeeded:
            if best is None or (p.mean > best.mean if direction == "max" else p.mean < best.mean):
                best = p
        baseline = _baseline(run_id, rounds, comparisons, mean_of, outcome_of)
        chosen_id, chosen_source = None, None
        content = report.content if report else {}
        reported = content.get("chosen") if isinstance(content.get("chosen"), dict) else None
        if reported and reported.get("attempt_id"):
            chosen_id, chosen_source = reported["attempt_id"], "final report"
        elif outcome is not None and outcome.attempt_id:
            chosen_id, chosen_source = outcome.attempt_id, "holdout outcome"
        elif best is not None:
            chosen_id, chosen_source = best.attempt_id, "best score so far"
        return {
            "cycle": cycle, "attempts": attempts, "rounds": rounds, "points": points,
            "comparisons": comparisons, "report": report, "outcome": outcome, "reads": reads,
            "submissions": submissions, "protocol": protocol, "mean_of": mean_of,
            "outcome_of": outcome_of, "best": best, "baseline": baseline,
            "chosen_id": chosen_id, "chosen_source": chosen_source,
        }

    return src.cached(("core", run_id, best_param), build)


def _baseline(run_id, rounds, comparisons, mean_of, outcome_of):
    """The baseline attempt: first scored attempt of the setup round, else the first reference."""
    setup = next((r for r in rounds if r.round.id == setup_round_id(run_id)), None)
    if setup is not None:
        for c in setup.candidates:
            if c.attempt_id in mean_of and outcome_of.get(c.attempt_id) == "succeeded":
                return mean_of[c.attempt_id]
    if comparisons:
        return mean_of.get(comparisons[0].reference_attempt_id)
    return None


def _find_limits(value) -> dict | None:
    """The first dict holding ``threshold_k`` and ``validation_scheme`` (a stored RunLimits)."""
    if isinstance(value, dict):
        if "threshold_k" in value and "validation_scheme" in value:
            return value
        for item in value.values():
            found = _find_limits(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_limits(item)
            if found:
                return found
    return None


def _infer_direction(comparisons, mean_of) -> str | None:
    """Max if a positive stored difference goes with a higher raw score, min if with a lower one."""
    for c in comparisons:
        cand, ref = mean_of.get(c.candidate_attempt_id), mean_of.get(c.reference_attempt_id)
        if cand is None or ref is None or c.difference == 0 or cand.mean == ref.mean:
            continue
        same_sign = (cand.mean - ref.mean > 0) == (c.difference > 0)
        return "max" if same_sign else "min"
    return None


def _protocol(src: Source, cycle, comparisons, mean_of, best_param) -> dict:
    """Metric, direction, scheme, repeats, threshold_k, seed and hash, each with its source."""
    project = read_project_file(src.project_dir)
    proto = project.get("protocol") if project and isinstance(project.get("protocol"), dict) else {}
    values: dict = {}
    sources: dict = {}

    def put(name, value, source):
        if name not in values and value is not None:
            values[name], sources[name] = value, source

    if project:
        put("metric", proto.get("metric", project.get("metric")), PROJECT_FILE)
        put("scheme", proto.get("validation_scheme"), PROJECT_FILE)
        put("repeats", proto.get("repetitions"), PROJECT_FILE)
        put("threshold_k", proto.get("threshold_k"), PROJECT_FILE)
        put("seed", proto.get("seed"), PROJECT_FILE)
    limits = None
    if "scheme" not in values or "repeats" not in values or "threshold_k" not in values:
        traces = src.section("role_traces", lambda: src.registry.list_role_traces(cycle.id), [])
        for trace in traces:
            limits = _find_limits(trace.input)
            if limits:
                break
    if limits:
        put("scheme", limits.get("validation_scheme"), "role trace limits")
        put("repeats", limits.get("repetitions"), "role trace limits")
        put("threshold_k", limits.get("threshold_k"), "role trace limits")
    put("metric", cycle.plan.metric, "plan")
    put("scheme", cycle.plan.validation, "plan")
    put("hash", cycle.protocol_hash, "registry")
    direction, direction_source = None, None
    if project:
        direction = _direction(proto.get("direction", project.get("direction")))
        direction_source = PROJECT_FILE if direction else None
    if direction is None:
        direction = _infer_direction(comparisons, mean_of)
        direction_source = "inferred from comparisons" if direction else None
    if direction is None:
        direction = best_param if best_param in DIRECTIONS else "max"
        direction_source = "best parameter" if best_param in DIRECTIONS else "default (max)"
    out = {name: values.get(name) for name in
           ("metric", "scheme", "repeats", "threshold_k", "seed", "hash")}
    out["direction"] = direction
    out["direction_source"] = direction_source
    out["sources"] = {name: sources.get(name) for name in out if name in sources}
    out["sources"]["direction"] = direction_source
    out["project_file"] = project is not None
    return out


def notes_index(project_dir) -> dict:
    """``{(type, id): {"notes": [...], "tags": [...]}}``: add, edit and delete (a tombstone) by note id."""
    if project_dir is None:
        return {}
    records, _ = store.read_jsonl(store.cockpit_path(project_dir, "notes.jsonl"))
    notes: dict[str, dict] = {}
    tags: dict[tuple, list] = {}
    for r in records:
        op, nid = r.get("op"), r.get("note_id") or r.get("id")
        target = (r.get("target_type"), r.get("target_id"))
        if op == "add" and isinstance(r.get("text"), str):
            notes[nid] = {"id": nid, "target": target, "text": r["text"], "who": r.get("who"), "at": r.get("at")}
        elif op == "edit" and nid in notes and isinstance(r.get("text"), str):
            notes[nid].update(text=r["text"], edited_at=r.get("at"))
        elif op == "delete":
            notes.pop(nid, None)
        elif op == "tags" and isinstance(r.get("tags"), list):
            tags[target] = [str(t) for t in r["tags"]]
    index: dict = {}
    for note in notes.values():
        index.setdefault(note["target"], {"notes": [], "tags": []})["notes"].append(
            {k: v for k, v in note.items() if k != "target"})
    for target, values in tags.items():
        index.setdefault(target, {"notes": [], "tags": []})["tags"] = values
    return index


def _annotations(src: Source, kind: str, target_id: str) -> dict:
    return src.notes().get((kind, target_id), {"notes": [], "tags": []})


def _note_text(annotations: dict) -> str:
    return " ".join([n["text"] for n in annotations["notes"]] + annotations["tags"])


def overlay_limits(base: dict, records: list[dict]) -> dict:
    """Effective limits: the base overlaid by the change records in file order, last value wins.

    The orchestrator has its own copy of this rule (``orchestrator.cockpit.overlay_cycle``); a test checks
    that both give the same values.
    """
    out = dict(base)
    for record in records:
        values = record.get("values")
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in LIMIT_KEYS and isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                out[key] = value
    return out


LIMIT_KEYS = ("rounds", "candidates", "stagnation")


def limits_view(src: Source, cycle) -> dict:
    """The base limits of the cycle, each change record and the effective values."""
    params = cycle.parameters or {}
    base = {"rounds": cycle.iterations, "candidates": params.get("candidates"),
            "stagnation": cycle.stagnation_k}
    records, _ = (store.read_jsonl(store.cockpit_path(src.project_dir, "limit-changes.jsonl"))
                  if src.project_dir is not None else ([], 0))
    effective = overlay_limits(base, records) if cycle.parameters is not None else dict(base)
    return {"base": base, "changes": records, "effective": effective,
            "applies": cycle.parameters is not None}


def _limits(cycle) -> dict:
    params = cycle.parameters or {}
    return {
        "rounds": cycle.iterations,
        "stagnation_k": cycle.stagnation_k,
        "max_fix_retries": cycle.max_fix_retries,
        "candidates": params.get("candidates"),
        "pause_every": params.get("pause_every"),
        "max_consecutive_failures": params.get("max_consecutive_failures"),
        "max_rejections": params.get("max_rejections"),
    }


def _rounds_done(run_id: str, rounds) -> int:
    done = 0
    for r in rounds:
        if r.round.id == setup_round_id(run_id) or not r.candidates:
            continue
        if all(c.outcome is not None for c in r.candidates):
            done += 1
    return done


def _holdout_view(core: dict) -> dict:
    """Sealed unless the registry recorded a read or an outcome; never a computed number."""
    outcome, reads = core["outcome"], core["reads"]
    if outcome is None and not reads:
        return {"status": "sealed", "score": None, "rows": None, "attempt_id": None,
                "reason": None, "reads": [], "metric": None, "recorded_at": None}
    out = {
        "status": outcome.status if outcome else "read",
        "score": outcome.score if outcome else None,
        "rows": outcome.rows if outcome else None,
        "attempt_id": outcome.attempt_id if outcome else reads[-1].attempt_id,
        "reason": outcome.reason if outcome else None,
        "metric": outcome.metric if outcome else None,
        "recorded_at": outcome.recorded_at if outcome else None,
        "reads": jsonable(reads),
    }
    return out


def _chosen_view(core: dict) -> dict | None:
    chosen_id = core["chosen_id"]
    if chosen_id is None:
        return None
    report = core["report"].content if core["report"] else {}
    reported = report.get("chosen") if isinstance(report.get("chosen"), dict) else {}
    bundle_id = reported.get("bundle_id")
    point = core["mean_of"].get(chosen_id)
    return {
        "attempt_id": chosen_id,
        "source": core["chosen_source"],
        "bundle_id": bundle_id,
        # the registry stores no model name: the model is the bundle that holds it
        "model": bundle_id,
        "hypothesis": reported.get("hypothesis"),
        "mean": point.mean if point else reported.get("validation_mean"),
        "dispersion": point.dispersion if point else reported.get("validation_dispersion"),
    }


def _point_view(p) -> dict | None:
    return None if p is None else {"attempt_id": p.attempt_id, "mean": p.mean,
                                   "dispersion": p.dispersion}


def _rule_view(core: dict, chosen: dict | None) -> dict:
    """The improvement rule with the numbers: a change is real above threshold_k x the baseline noise."""
    k = core["protocol"]["threshold_k"]
    baseline = core["baseline"]
    noise = baseline.dispersion if baseline else None
    comparison = None
    if chosen is not None:
        comparison = next((c for c in reversed(core["comparisons"])
                           if c.candidate_attempt_id == chosen["attempt_id"]), None)
    return {
        "threshold_k": k, "baseline_noise": noise, "bar": (k * noise) if (k is not None and noise is not None) else None,
        "difference": comparison.difference if comparison else None,
        "threshold": comparison.threshold if comparison else None,
        "is_real": comparison.is_real if comparison else None,
        "text": ("The program picks a model only when it beats the best one so far by more than "
                 "threshold_k x the baseline noise (mean paired difference over the validation partitions). "
                 "'Best seen' is just the highest score observed, even if it did not clear that bar."),
    }


def run_row(src: Source, run_id: str, best_param=None) -> dict:
    core = _core(src, run_id, best_param)
    cycle = core["cycle"]
    submissions = core["submissions"]
    stop_reason = (core["report"].content.get("stop_reason") if core["report"] else None)
    chosen = _chosen_view(core)
    limits = limits_view(src, cycle)
    notes = _annotations(src, "run", cycle.id)
    return {
        "notes": notes["notes"], "tags": notes["tags"],
        "best_seen": _point_view(core["best"]), "improvement_rule": _rule_view(core, chosen),
        "limits_effective": limits["effective"],
        "id": cycle.id,
        "project_id": cycle.project_id,
        "state": cycle.state,
        "status": _status(cycle.state, stop_reason, chosen is not None),
        "plan": jsonable(cycle.plan),
        "rounds_done": _rounds_done(run_id, core["rounds"]),
        "rounds_planned": limits["effective"]["rounds"],
        "attempts_total": len(core["attempts"]),
        "best": _point_view(core["best"]),
        "chosen": chosen,
        "holdout": _holdout_view(core),
        "submission": (jsonable({
            "submission_id": submissions[-1].submission_id, "status": submissions[-1].status,
            "public_score": submissions[-1].public_score,
            "submitted_at": submissions[-1].submitted_at}) if submissions else None),
        "direction": core["protocol"]["direction"],
        "protocol_hash": cycle.protocol_hash,
    }


# -- runs ---------------------------------------------------------------------------------------


def runs(src: Source, q: str | None = None) -> list[dict]:
    """Every cycle (a project may have several) with its summary row; ``q`` searches id, plan, notes, tags."""
    cycles = src.section("runs", lambda: views.list_cycles(src.registry), [])
    rows = [run_row(src, c.id) for c in cycles]
    q = (q or "").strip().lower()
    if not q:
        return rows
    return [r for r in rows if q in " ".join([
        r["id"], json.dumps(r["plan"]), _note_text({"notes": r["notes"], "tags": r["tags"]})]).lower()]


def run_summary(src: Source, run_id: str, best_param: str | None = None) -> dict:
    """The run row plus protocol, limits, stop reason, what worked and what failed."""
    core = _core(src, run_id, best_param)
    cycle = core["cycle"]
    row = run_row(src, run_id, best_param)
    content = core["report"].content if core["report"] else {}
    attempts = {a.attempt_id: a for a in core["attempts"]}
    worked = []
    for c in core["comparisons"]:
        if c.is_real and c.candidate_attempt_id in attempts:
            point = core["mean_of"].get(c.candidate_attempt_id)
            worked.append({
                "attempt_id": c.candidate_attempt_id, "difference": c.difference,
                "threshold": c.threshold, "mean": point.mean if point else None,
            })
    failures = src.section("failures", src.registry.list_failures, [])
    failed = [
        {"attempt_id": f.attempt_id, "what_changed": f.what_changed, "outcome": f.outcome,
         "failure_kind": f.failure_kind, "reason": f.reason}
        for f in failures if f.attempt_id in attempts
    ]
    counts = {"succeeded": 0, "failed": 0, "inconclusive": 0, "open": 0}
    for a in core["attempts"]:
        counts[a.outcome or "open"] = counts.get(a.outcome or "open", 0) + 1
    row.update({
        "created_at": core["report"].created_at if core["report"] else None,
        "stop_reason": content.get("stop_reason"),
        "baseline": _point_view(core["baseline"]),
        "worked": worked,
        "failed": failed,
        "counts": counts,
        "protocol": core["protocol"],
        "limits": _limits(cycle),
        "limits_view": limits_view(src, cycle),
        "has_report": core["report"] is not None,
    })
    return row


def chart(src: Source, run_id: str, best_param: str | None = None) -> dict:
    """Score per attempt with mean and dispersion, the baseline and the improvement bar."""
    core = _core(src, run_id, best_param)
    position_of = {}
    for r in core["rounds"]:
        for c in r.candidates:
            position_of[c.attempt_id] = (r.round.id, r.position)
    iteration_of = {a.attempt_id: a.iteration for a in core["attempts"]}
    comparison_of = {c.candidate_attempt_id: c for c in core["comparisons"]}
    baseline = core["baseline"]
    direction = core["protocol"]["direction"]
    threshold = core["comparisons"][0].threshold if core["comparisons"] else None
    bar = None
    if baseline is not None and threshold is not None:
        bar = baseline.mean + threshold if direction == "max" else baseline.mean - threshold
    points = []
    for p in sorted(core["points"], key=lambda p: p.iteration):
        round_id, position = position_of.get(p.attempt_id, (None, None))
        comparison = comparison_of.get(p.attempt_id)
        points.append({
            "attempt_id": p.attempt_id, "iteration": p.iteration, "round_id": round_id,
            "round_position": position, "mean": p.mean, "dispersion": p.dispersion,
            "outcome": core["outcome_of"].get(p.attempt_id),
            "chosen": p.attempt_id == core["chosen_id"],
            "is_baseline": baseline is not None and p.attempt_id == baseline.attempt_id,
            "is_real": comparison.is_real if comparison else None,
        })
    return {
        "run_id": run_id, "partition": PARTITION, "direction": direction,
        "direction_source": core["protocol"]["direction_source"],
        "metric": core["protocol"]["metric"],
        "baseline": _point_view(baseline), "threshold": threshold, "bar": bar,
        "chosen_attempt_id": core["chosen_id"], "points": points,
    }


# -- rounds ----------------------------------------------------------------------------------


def _comparison_view(c) -> dict:
    return {
        "candidate_attempt_id": c.candidate_attempt_id,
        "reference_attempt_id": c.reference_attempt_id,
        "difference": c.difference, "threshold": c.threshold, "is_real": c.is_real,
        "vs_bar": c.difference - c.threshold,
        "verdict": "real improvement" if c.is_real else "not a real improvement",
    }


def comparisons(src: Source, run_id: str) -> list[dict]:
    core = _core(src, run_id)
    return [_comparison_view(c) for c in core["comparisons"]]


def rounds(src: Source, run_id: str) -> list[dict]:
    core = _core(src, run_id)
    out = []
    for r in core["rounds"]:
        out.append({
            "id": r.round.id, "position": r.position, "tasks": list(r.round.tasks),
            "is_setup": r.round.id == setup_round_id(run_id),
            "candidates": [c.attempt_id for c in r.candidates],
            "complete": bool(r.candidates) and all(c.outcome is not None for c in r.candidates),
        })
    return out


def _output_view(index: int, trace) -> dict:
    return {
        "n": index, "role": trace.role, "outcome": trace.outcome, "created_at": trace.created_at,
        "candidate_id": trace.candidate_id, "output_type": trace.output_type,
        "output": jsonable(trace.output),
    }


def round_detail(src: Source, run_id: str, round_id: str) -> dict:
    core = _core(src, run_id)
    detail = next((r for r in core["rounds"] if r.round.id == round_id), None)
    if detail is None:
        raise NotFoundError(f"round {round_id!r} does not exist in cycle {run_id!r}")
    reg = src.registry
    comparison_of = {c.candidate_attempt_id: c for c in core["comparisons"]}
    candidates = []
    for s in detail.candidates:
        attempt = reg.get_attempt(s.attempt_id)
        point = core["mean_of"].get(s.attempt_id)
        comparison = comparison_of.get(s.attempt_id)
        candidates.append({
            "attempt_id": s.attempt_id, "iteration": s.iteration, "parent_id": s.parent_id,
            "phase": s.phase, "outcome": s.outcome, "failure_kind": s.failure_kind,
            "what_changed": attempt.attempt.what_changed, "runs": len(attempt.executions),
            "closed": attempt.closed,
            "mean": point.mean if point else None,
            "dispersion": point.dispersion if point else None,
            "comparison": _comparison_view(comparison) if comparison else None,
        })
    reports = src.section(
        "reviewer_reports", lambda: reg.list_reviewer_reports(run_id, round_id), [])
    given = src.section(
        "guidance", lambda: [g for g in reg.list_guidance(run_id) if g.round_id == round_id], [])
    all_traces = src.section("role_traces", lambda: reg.list_role_traces(run_id), [])
    outputs: dict = {role: [] for role in _ROLE_OUTPUT_ORDER}
    for index, trace in enumerate(all_traces, start=1):
        if trace.round_id == round_id:
            outputs.setdefault(trace.role, []).append(_output_view(index, trace))
    return {
        "id": detail.round.id, "run_id": run_id, "position": detail.position,
        "tasks": list(detail.round.tasks),
        "is_setup": detail.round.id == setup_round_id(run_id),
        "candidates": candidates,
        "reviewer_reports": jsonable(reports),
        "guidance": jsonable(given),
        "role_outputs": outputs,
        "direction": core["protocol"]["direction"],
    }


# -- attempts and executions ---------------------------------------------------------------------


def _execution_view(sequence: int, e) -> dict:
    return {
        "id": e.id, "attempt_id": e.attempt_id, "sequence": sequence, "outcome": e.outcome,
        "failure_kind": e.failure_kind, "reason": e.reason, "seed": e.seed,
        "code_hash": e.code_hash, "data_hash": e.data_hash, "protocol_hash": e.protocol_hash,
        "duration_seconds": e.duration_seconds, "metrics": jsonable(e.metrics),
        "config": jsonable(e.config), "resources": jsonable(e.resources),
        "artifacts": list(e.artifacts),
        "library_versions": (e.resources or {}).get("library_versions") or {},
    }


def _attempt_row(detail, summary, round_of, point, chosen_id, annotations=None) -> dict:
    a = detail.attempt
    annotations = annotations or {"notes": [], "tags": []}
    return {
        "notes": annotations["notes"], "tags": annotations["tags"],
        "id": a.id, "run_id": detail.cycle_id, "iteration": detail.iteration,
        "round_id": round_of.get(a.id), "parent_id": a.parent_id, "phase": a.phase,
        "outcome": detail.outcome, "failure_kind": detail.failure_kind, "reason": detail.reason,
        "hypothesis": a.hypothesis, "what_changed": a.what_changed,
        "mean": point.mean if point else None,
        "dispersion": point.dispersion if point else None,
        "runs": len(detail.executions), "closed": detail.closed, "bundle_id": a.bundle_id,
        "chosen": a.id == chosen_id,
    }


def _matches(row: dict, detail, q: str) -> bool:
    a = detail.attempt
    hay = " ".join(str(x) for x in (
        a.id, a.hypothesis, a.rationale, a.what_changed, detail.reason, detail.failure_kind,
        a.expected_effect, " ".join(a.evidence),
        _note_text({"notes": row.get("notes", []), "tags": row.get("tags", [])})) if x).lower()
    return q.lower() in hay


def attempts(src: Source, run_id: str, filters: dict | None = None) -> list[dict]:
    """Attempts of a run; filters: outcome, round, min_score, max_score, q (all optional)."""
    filters = filters or {}
    outcome = filters.get("outcome") or None
    if outcome and outcome not in (*OUTCOMES, "open"):
        raise ValidationError("outcome", f"must be one of {', '.join((*OUTCOMES, 'open'))}")
    low = _number("min_score", filters.get("min_score"))
    high = _number("max_score", filters.get("max_score"))
    q = (filters.get("q") or "").strip()
    round_id = filters.get("round") or None
    core = _core(src, run_id)
    round_of = {c.attempt_id: r.round.id for r in core["rounds"] for c in r.candidates}
    out = []
    for summary in core["attempts"]:
        if outcome and (summary.outcome or "open") != outcome:
            continue
        if round_id and round_of.get(summary.attempt_id) != round_id:
            continue
        point = core["mean_of"].get(summary.attempt_id)
        if low is not None and (point is None or point.mean < low):
            continue
        if high is not None and (point is None or point.mean > high):
            continue
        detail = src.registry.get_attempt(summary.attempt_id)
        row = _attempt_row(detail, summary, round_of, point, core["chosen_id"],
                           _annotations(src, "attempt", summary.attempt_id))
        if q and not _matches(row, detail, q):
            continue
        out.append(row)
    return out


def executions(src: Source, run_id: str) -> list[dict]:
    core = _core(src, run_id)
    out = []
    for summary in core["attempts"]:
        detail = src.registry.get_attempt(summary.attempt_id)
        out.extend(_execution_view(i, e) for i, e in enumerate(detail.executions, start=1))
    return out


def attempt_executions(src: Source, attempt_id: str) -> list[dict]:
    detail = src.registry.get_attempt(attempt_id)
    return [_execution_view(i, e) for i, e in enumerate(detail.executions, start=1)]


def _bundle_view(b) -> dict:
    # the reference is plain text: the web never resolves it or opens it
    return {"id": b.id, "run_id": b.cycle_id, "folder_hash": b.folder_hash,
            "reference": b.reference, "schema": jsonable(b.schema)}


def attempt_detail(src: Source, attempt_id: str) -> dict:
    detail = src.registry.get_attempt(attempt_id)  # NotFoundError for unknown ids
    a = detail.attempt
    run_id = detail.cycle_id
    core = _core(src, run_id)
    round_id = next(
        (r.round.id for r in core["rounds"] if any(c.attempt_id == attempt_id for c in r.candidates)),
        None)
    bundle = None
    if a.bundle_id is not None:
        bundle = next((b for b in src.section(
            "bundles", lambda: src.registry.list_bundles(run_id), []) if b.id == a.bundle_id), None)
    traces = src.section("role_traces", lambda: src.registry.list_role_traces(run_id), [])
    point = core["mean_of"].get(attempt_id)
    execs = [_execution_view(i, e) for i, e in enumerate(detail.executions, start=1)]
    last = execs[-1] if execs else None
    versions = next((e["library_versions"] for e in reversed(execs) if e["library_versions"]), {})
    comparison = next((c for c in core["comparisons"] if c.candidate_attempt_id == attempt_id), None)
    annotations = _annotations(src, "attempt", attempt_id)
    return {
        "notes": annotations["notes"], "tags": annotations["tags"],
        "id": a.id, "run_id": run_id, "round_id": round_id, "iteration": detail.iteration,
        "phase": a.phase, "parent_id": a.parent_id, "closed": detail.closed,
        "conclusion": detail.conclusion, "fingerprint": detail.fingerprint,
        "hypothesis": a.hypothesis, "rationale": a.rationale, "evidence": list(a.evidence),
        "what_changed": a.what_changed, "expected_effect": a.expected_effect,
        "measure": a.measure, "risks": list(a.risks), "exploratory": a.exploratory,
        "estimated_cost": a.estimated_cost, "artifacts": list(a.artifacts),
        "outcome": detail.outcome, "failure_kind": detail.failure_kind, "reason": detail.reason,
        "mean": point.mean if point else None,
        "dispersion": point.dispersion if point else None,
        "chosen": attempt_id == core["chosen_id"],
        "comparison": _comparison_view(comparison) if comparison else None,
        "retries": max(len(execs) - 1, 0),
        "duration_seconds": sum(e["duration_seconds"] for e in execs),
        "seed": last["seed"] if last else None,
        "code_hash": last["code_hash"] if last else None,
        "data_hash": last["data_hash"] if last else None,
        "protocol_hash": last["protocol_hash"] if last else None,
        "library_versions": versions,
        "executions": execs,
        "bundle_id": a.bundle_id,
        "bundle": _bundle_view(bundle) if bundle else None,
        "pipeline": views.pipeline_reference([x for e in detail.executions for x in e.artifacts]),
        "role_trace_count": sum(1 for t in traces if t.candidate_id == attempt_id),
        "direction": core["protocol"]["direction"],
    }


# -- reports, guidance, submissions, holdout, bundles ----------------------------------------------


def reviewer_reports(src: Source, run_id: str) -> list[dict]:
    src.registry.get_cycle(run_id)
    return jsonable(src.section(
        "reviewer_reports", lambda: src.registry.list_reviewer_reports(run_id), []))


def guidance(src: Source, run_id: str) -> list[dict]:
    src.registry.get_cycle(run_id)
    items = src.section("guidance", lambda: views.guidance_page(src.registry, run_id)["items"], [])
    return [{"id": i["guidance"].id, "round_id": i["guidance"].round_id,
             "text": i["guidance"].text, "created_at": i["guidance"].created_at,
             "suggestion_id": i["suggestion"].id} for i in items]


def suggestions(src: Source) -> list[dict]:
    items = src.section("suggestions", lambda: views.suggestions_page(src.registry), [])
    return [{"id": i["suggestion"].id, "text": i["suggestion"].text,
             "origin_cycle": i["suggestion"].origin_cycle,
             "spawned_cycles": [c.id for c in i["cycles"]]} for i in items]


def submissions(src: Source, run_id: str) -> list[dict]:
    src.registry.get_cycle(run_id)
    return jsonable(src.section(
        "submissions", lambda: src.registry.list_submissions(run_id), []))


def holdout(src: Source, run_id: str) -> dict:
    return _holdout_view(_core(src, run_id))


def final_report(src: Source, run_id: str) -> dict:
    core = _core(src, run_id)
    report = core["report"]
    return {
        "available": report is not None,
        "created_at": report.created_at if report else None,
        "content": jsonable(report.content) if report else None,
    }


def bundles(src: Source, run_id: str) -> list[dict]:
    src.registry.get_cycle(run_id)
    items = src.section("bundles", lambda: src.registry.list_bundles(run_id), [])
    return [_bundle_view(b) for b in items]


# -- LLM calls and role traces --------------------------------------------------------------------


def llm_calls(src: Source, run_id: str, role=None, outcome=None, q=None) -> list[dict]:
    src.registry.get_cycle(run_id)
    if role:
        validate_role(role)
    entries = src.section(
        "llm_calls", lambda: src.registry.list_llm_calls(run_id, role or None), [])
    out = []
    for index, e in enumerate(entries, start=1):
        if outcome and e.outcome != outcome:
            continue
        if q and q.lower() not in f"{e.purpose} {e.model or ''} {e.rejection_reason or ''}".lower():
            continue
        out.append({
            "index": index, "role": e.role, "purpose": e.purpose,
            # the registry stores no prompt: this is a short hash of the purpose text
            "purpose_hash": hashlib.sha256(e.purpose.encode("utf-8")).hexdigest()[:12],
            "outcome": e.outcome, "started_at": e.started_at,
            "duration_seconds": e.duration_seconds, "input_tokens": e.input_tokens,
            "output_tokens": e.output_tokens, "attempt": e.attempt, "model": e.model,
            "rejection_reason": e.rejection_reason,
        })
    return out


def _trace_row(index: int, t) -> dict:
    return {
        "n": index, "role": t.role, "round_id": t.round_id, "candidate_id": t.candidate_id,
        "input_type": t.input_type, "output_type": t.output_type, "outcome": t.outcome,
        "llm_call_id": t.llm_call_id, "created_at": t.created_at,
    }


def role_traces(src: Source, run_id: str, role=None, round_id=None, candidate_id=None) -> list[dict]:
    core = _core(src, run_id)
    if role:
        validate_role(role)
    if round_id and not any(r.round.id == round_id for r in core["rounds"]):
        raise NotFoundError(f"round {round_id!r} does not exist in cycle {run_id!r}")
    everything = src.section("role_traces", lambda: src.registry.list_role_traces(run_id), [])
    return [
        _trace_row(i, t) for i, t in enumerate(everything, start=1)
        if (not role or t.role == role) and (not round_id or t.round_id == round_id)
        and (not candidate_id or t.candidate_id == candidate_id)
    ]


def role_trace(src: Source, run_id: str, n: int) -> dict:
    src.registry.get_cycle(run_id)
    everything = src.section("role_traces", lambda: src.registry.list_role_traces(run_id), [])
    if n < 1 or n > len(everything):
        raise NotFoundError(f"role trace {n} does not exist in cycle {run_id!r}")
    t = everything[n - 1]
    return {**_trace_row(n, t), "input": jsonable(t.input), "output": jsonable(t.output)}
