"""Performance analytics for the pages and the API (ticket 29): registry (read-only) joined with the analytics files.

Every section returns either data or a ``not_recorded`` object with a plain reason (``analytics.words``). The
chain is attempt -> last succeeded execution -> ``config.bundle_id`` -> ``analytics/index.jsonl`` -> files, all read
through ``store`` (paths from validated ids only, hashes checked). Per-partition scores come from the registry
metrics ``validation/<partition>`` (so old projects still show folds); the recorded copy is a cross-check.

DEVELOPMENT DATA ONLY: nothing here calls or reads anything about the holdout; the existing holdout views of
``data.py`` are not used.
"""

import threading
from collections import OrderedDict

from kaggle_agent.analytics import classification, compare, correlation, errors, flags, folds, hardest
from kaggle_agent.analytics import history as history_mod
from kaggle_agent.analytics import importance as importance_mod
from kaggle_agent.analytics import regression, segments, tasks
from kaggle_agent.analytics.describe import validate_describe
from kaggle_agent.analytics.families import feature_families
from kaggle_agent.analytics.oof_format import OofError, decode_oof, decode_snapshot
from kaggle_agent.analytics.words import is_not_recorded, not_recorded
from kaggle_agent.proposals import ValidationError
from kaggle_agent.registry import NotFoundError

from . import data, store

VALIDATION = "validation/"
_LRU: "OrderedDict" = OrderedDict()
_LRU_LOCK = threading.Lock()
_LRU_MAX = 8


def _lru(key, build):
    with _LRU_LOCK:
        if key in _LRU:
            _LRU.move_to_end(key)
            return _LRU[key]
    value = build()
    with _LRU_LOCK:
        _LRU[key] = value
        while len(_LRU) > _LRU_MAX:
            _LRU.popitem(last=False)
    return value


def _word(direction: str) -> str:
    return "maximize" if direction == "max" else "minimize"


# -- registry side ----------------------------------------------------------------------------------


def _detail(src, attempt_id):
    return src.registry.get_attempt(attempt_id)  # NotFoundError -> 404


def _last_succeeded(detail):
    done = [e for e in detail.executions if e.outcome == "succeeded"]
    return done[-1] if done else None


def _scores_of(execution) -> dict:
    return {m.partition[len(VALIDATION):]: m.mean for m in execution.metrics if m.partition.startswith(VALIDATION)}


def _bundle_of(detail):
    """(bundle_id, execution, not_recorded or None)."""
    last = _last_succeeded(detail)
    if last is None:
        return None, None, not_recorded("run_failed" if detail.executions else "no_bundle")
    bundle = (last.config or {}).get("bundle_id")
    if not isinstance(bundle, str) or not store.BUNDLE_ID.fullmatch(bundle):
        return None, last, not_recorded("no_bundle")
    return bundle, last, None


def _protocol(src, run_id) -> dict:
    core = data._core(src, run_id)
    proto = core["protocol"]
    return {"metric": proto.get("metric"), "direction": _word(proto["direction"]), "threshold_k": proto.get("threshold_k"),
            "core": core}


def _task(src, run_id) -> dict | None:
    metric = _protocol(src, run_id)["metric"]
    try:
        return {**tasks.task_of(metric), "metric": metric} if metric else None
    except ValidationError:
        return None


# -- files side -------------------------------------------------------------------------------------


def _index(src):
    """(latest lines, skipped) or None when the project has no analytics folder."""
    if src.project_dir is None:
        return None
    return src.cached("analytics-index", lambda: _read_index(src.project_dir))


def _read_index(project_dir):
    folder = store.Path(project_dir) / store.ANALYTICS_DIR
    if not folder.is_dir():
        return None
    return store.analytics_index(project_dir)


def _line_status(line, kind_text):
    status = line.get("status")
    if status == "ok":
        return None
    if status in ("too_large", "budget_exhausted"):
        return not_recorded(status)
    return not_recorded("recording_error", f"{kind_text}: {status}")


def _oof(src, run_id, bundle_id):
    """(OofTable, line, None) or (None, line|None, not_recorded)."""
    index = _index(src)
    if index is None:
        return None, None, not_recorded("older_project")
    line = index[0].get(("oof", bundle_id))
    if line is None:
        return None, None, not_recorded("not_in_index")
    bad = _line_status(line, "oof")
    if bad:
        return None, line, bad
    task = _task(src, run_id)

    def build():
        blob, reason = store.read_analytics_file(src.project_dir, line)
        if blob is None:
            return None, reason
        try:
            return decode_oof(blob, classification=bool(task and task["task"] == "classification")), None
        except OofError as exc:
            return None, str(exc)

    table, reason = _lru(("oof", line["sha256"]), build)
    if table is None:
        return None, line, not_recorded("not_readable", reason)
    return table, line, None


def _development(src):
    index = _index(src)
    if index is None:
        return None, not_recorded("older_project")
    line = index[0].get(("development", None))
    if line is None:
        return None, not_recorded("no_snapshot")
    bad = _line_status(line, "development")
    if bad:
        return None, bad

    def build():
        blob, reason = store.read_analytics_file(src.project_dir, line)
        if blob is None:
            return None, reason
        try:
            return decode_snapshot(blob), None
        except OofError as exc:
            return None, str(exc)

    frame, reason = _lru(("dev", line["sha256"]), build)
    return (frame, None) if frame is not None else (None, not_recorded("not_readable", reason))


def _describe(src, bundle_id):
    """{'status': valid|missing|invalid, 'content': dict|None, 'reason': text|None} or a not_recorded."""
    index = _index(src)
    if index is None:
        return not_recorded("older_project")
    line = index[0].get(("describe", bundle_id))
    if line is None:
        return not_recorded("not_in_index")
    status = line.get("status")
    if status == "missing":
        return {"status": "missing", "content": None, "reason": not_recorded("describe_missing")["reason"]}
    if status == "invalid":
        return {"status": "invalid", "content": None,
                "reason": "provided but invalid: " + str(line.get("reason") or "unknown reason")[:200]}
    blob, reason = store.read_analytics_file(src.project_dir, line)
    if blob is None:
        return not_recorded("not_readable", reason)
    result = validate_describe(blob)
    if result.status != "valid":
        return {"status": "invalid", "content": None, "reason": "provided but invalid: " + str(result.reason)}
    return {"status": "valid", "content": result.content, "reason": None, "warnings": line.get("warnings") or []}


# -- sections ---------------------------------------------------------------------------------------


def _attempt_context(src, attempt_id):
    detail = _detail(src, attempt_id)
    run_id = detail.cycle_id
    bundle, last, bad = _bundle_of(detail)
    return detail, run_id, bundle, last, bad


def overview(src, attempt_id: str) -> dict:
    """What is recorded for the attempt, with hashes and reasons."""
    detail, run_id, bundle, last, bad = _attempt_context(src, attempt_id)
    out = {"attempt_id": attempt_id, "run_id": run_id, "bundle_id": bundle, "task": _task(src, run_id),
           "development_only": True, "recorded": {}}
    if bad:
        out["recorded"] = {"oof": bad, "describe": bad, "development": bad}
        out["not_recorded"] = bad
        return out
    index = _index(src)
    if index is None:
        old = not_recorded("older_project")
        out["recorded"] = {"oof": old, "describe": old, "development": old}
        return out
    lines = index[0]
    for kind, key in (("oof", ("oof", bundle)), ("describe", ("describe", bundle)), ("development", ("development", None))):
        line = lines.get(key)
        if line is None:
            out["recorded"][kind] = not_recorded("not_in_index" if kind != "development" else "no_snapshot")
        else:
            bad_line = _line_status(line, kind)
            out["recorded"][kind] = bad_line or {k: line.get(k) for k in ("status", "sha256", "bytes", "rows", "partitions", "at")}
            if kind == "describe":
                out["recorded"][kind] = {**out["recorded"][kind], "status": line.get("status"), "reason": line.get("reason")}
    out["recorded"]["importance"] = (
        {"status": "ok"} if store.read_importance(src.project_dir, bundle) is not None else not_recorded("no_importance"))
    out["skipped_index_lines"] = index[1]
    return out


def _previous_best(src, run_id, attempt_id, direction):
    core = data._core(src, run_id)
    me = next((p for p in core["points"] if p.attempt_id == attempt_id), None)
    if me is None:
        return None, None
    ok = [p for p in core["points"] if p.iteration < me.iteration and core["outcome_of"].get(p.attempt_id) == "succeeded"]
    if not ok:
        return me.mean, None
    best = max(p.mean for p in ok) if direction == "maximize" else min(p.mean for p in ok)
    gain = (me.mean - best) if direction == "maximize" else (best - me.mean)
    return me.mean, gain


def _bar(src, run_id):
    rule = data._rule_view(data._core(src, run_id), None)
    return rule.get("bar")


def fold_scores(src, attempt_id: str) -> dict:
    """Per-partition scores, spread, worst fold and stability (F1, F10) from the registry metrics."""
    detail, run_id, bundle, last, bad = _attempt_context(src, attempt_id)
    if last is None:
        return not_recorded("run_failed" if detail.executions else "no_scores")
    scores = _scores_of(last)
    if not scores:
        return not_recorded("no_scores")
    direction = _protocol(src, run_id)["direction"]
    view = folds.fold_view(scores, direction)
    _, gain = _previous_best(src, run_id, attempt_id, direction)
    view["stability"] = folds.stability(scores, direction, gain, _bar(src, run_id))
    view["source"] = "registry"
    view["recorded_check"] = None
    index = _index(src)
    if bundle and index is not None:
        line = index[0].get(("oof", bundle))
        recorded = (line or {}).get("fold_scores")
        if isinstance(recorded, dict):
            diff = [p for p in scores if p in recorded and abs(scores[p] - recorded[p]) > 1e-6]
            view["recorded_check"] = {"matches": not diff and set(recorded) == set(scores), "differs": diff}
            if diff:
                view["warning"] = "The recorded copy differs from the registry for " + ", ".join(diff) + "; the registry value is shown."
    return view


def _oof_for(src, attempt_id):
    detail, run_id, bundle, last, bad = _attempt_context(src, attempt_id)
    if bad:
        return None, bad, run_id, None
    table, line, bad = _oof(src, run_id, bundle)
    return table, bad, run_id, bundle


def panels(src, attempt_id: str, rep: int = 0) -> dict:
    table, bad, run_id, _ = _oof_for(src, attempt_id)
    if bad:
        return bad
    task = _task(src, run_id)
    if task is None:
        return not_recorded("recording_error", "unknown metric")
    if task["task"] == "regression":
        return {"task": "regression", **regression.regression_panels(table.frame, rep)}
    return {"task": "classification", "input_kind": task["input_kind"],
            **classification.classification_panels(table.frame, table.classes, task["input_kind"], rep)}


def _errors_and_features(src, attempt_id):
    table, bad, run_id, _ = _oof_for(src, attempt_id)
    if bad:
        return None, None, bad
    task = _task(src, run_id)
    err = errors.row_errors(table.frame, task["task"], table.classes, task["input_kind"])
    if err is None:
        return None, None, not_recorded("label_only", "per-row errors need labels or probabilities")
    feats, bad = _development(src)
    if bad:
        return None, None, bad
    return err, feats, None


def segment_table(src, attempt_id: str, min_size: int = 1, interactions: bool = True, top: int = 20) -> dict:
    table, bad, run_id, _ = _oof_for(src, attempt_id)
    if bad:
        return bad
    err, feats, bad = _errors_and_features(src, attempt_id)
    if bad:
        return bad
    task = _task(src, run_id)["task"]
    line = _index(src)[0][("oof", _bundle_of(_detail(src, attempt_id))[0])]
    key = ("segments", line["sha256"], interactions)
    full = _lru(key, lambda: segments.build_segments(feats, err, task, interactions))
    return segments.filter_segments(full, min_size, top)


def hardest_rows(src, attempt_id: str, sort: str = "error", order: str = "desc", limit: int = 50) -> dict:
    err, feats, bad = _errors_and_features(src, attempt_id)
    if bad:
        return bad
    return hardest.hardest_rows(feats, err, sort, order, limit)


def sanity_flags(src, attempt_id: str) -> dict:
    detail, run_id, bundle, last, bad = _attempt_context(src, attempt_id)
    direction = _protocol(src, run_id)["direction"]
    core = data._core(src, run_id)
    history = [{"attempt": p.attempt_id, "score": p.mean} for p in sorted(core["points"], key=lambda p: p.iteration)
               if core["outcome_of"].get(p.attempt_id) == "succeeded"]
    out = {"flags": [], "checked": []}
    out["flags"] += flags.score_flags(history, attempt_id, direction, _bar(src, run_id))
    out["checked"].append("score jumps")
    err, feats, bad_err = _errors_and_features(src, attempt_id)
    if err is not None and flags.is_perfect(err):
        out["flags"] += flags.score_flags([], attempt_id, direction, None, perfect=True)
    out["checked"].append("perfect out-of-fold result" if err is not None else None)
    table, bad_oof, _, _ = _oof_for(src, attempt_id)
    dev, bad_dev = _development(src)
    if dev is not None and table is not None:
        target = table.frame.groupby("row")["truth"].first()
        out["flags"] += flags.column_flags(dev.loc[dev.index.intersection(target.index)], target.loc[dev.index.intersection(target.index)])
        out["checked"].append("ID-like columns and features that mirror the target")
    else:
        out["columns"] = bad_dev if dev is None else bad_oof
    out["checked"] = [c for c in out["checked"] if c]
    out["explanation"] = ("Quick checks for leakage and mistakes. They are heuristics with fixed thresholds: "
                          "they can flag a clean column or miss a real leak.")
    return out


def features(src, attempt_id: str) -> dict:
    """describe.json content, native importance, permutation importance and families."""
    detail, run_id, bundle, last, bad = _attempt_context(src, attempt_id)
    if bad:
        return bad
    desc = _describe(src, bundle)
    out = {"attempt_id": attempt_id, "bundle_id": bundle, "describe": desc, "native": None, "permutation": None}
    dev, _ = _development(src)
    known = [str(c) for c in dev.columns] if dev is not None else None
    content = desc.get("content") if isinstance(desc, dict) and not is_not_recorded(desc) else None
    if content:
        feats = list(content.get("features_used") or [])
        fam = feature_families(feats + list((content.get("native_importances") or {})), content.get("derived_features"), known)
        out["families"] = fam
        if content.get("native_importances"):
            out["native"] = {"features": content["native_importances"],
                             "groups": importance_mod.group_importance(content["native_importances"], fam)}
    perm = store.read_importance(src.project_dir, bundle) if src.project_dir is not None else None
    if perm:
        scores = {r["feature"]: r["importance"] for r in perm.get("features", [])}
        fam = feature_families(list(scores), (content or {}).get("derived_features"), known)
        out["permutation"] = {**perm, "groups": importance_mod.group_importance(scores, fam)}
    else:
        out["permutation"] = not_recorded("no_importance")
    return out


# -- run level --------------------------------------------------------------------------------------


def _valid_attempt(src, run_id, attempt_id):
    detail = _detail(src, attempt_id)
    if detail.cycle_id != run_id:
        raise NotFoundError(f"attempt {attempt_id!r} is not in run {run_id!r}")
    return detail


def paired(src, run_id: str, a: str, b: str, with_correlation: bool = True) -> dict:
    """A (reference) against B (candidate) fold by fold, the program's rule beside it, and the OOF correlation."""
    data._core(src, run_id)
    da, db = _valid_attempt(src, run_id, a), _valid_attempt(src, run_id, b)
    proto = _protocol(src, run_id)
    core = proto["core"]
    la, lb = _last_succeeded(da), _last_succeeded(db)
    out = {"run_id": run_id, "a": a, "b": b, "direction": proto["direction"], "threshold_k": proto["threshold_k"]}
    if la is None or lb is None:
        out["comparison"] = not_recorded("run_failed", "A" if la is None else "B")
    else:
        base = core["baseline"]
        baseline_scores = None
        if base is not None:
            bd = src.registry.get_attempt(base.attempt_id)
            be = _last_succeeded(bd)
            baseline_scores = _scores_of(be) if be else None
        out["comparison"] = compare.paired_comparison(_scores_of(la), _scores_of(lb), baseline_scores,
                                                      proto["direction"], proto["threshold_k"])
    if not with_correlation:
        return out
    ta, ba, _, _ = _oof_for(src, a)
    tb, bb, _, _ = _oof_for(src, b)
    if ta is None or tb is None:
        out["correlation"] = ba or bb
    else:
        task = _task(src, run_id)
        out["correlation"] = correlation.prediction_correlation(ta.frame, tb.frame, task["task"], ta.classes, task["input_kind"])
    return out


def _attempt_rows(src, run_id):
    core = data._core(src, run_id)
    round_of = {c.attempt_id: r.round.id for r in core["rounds"] for c in r.candidates}
    index = _index(src)
    rows = []
    for summary in sorted(core["attempts"], key=lambda s: s.iteration):
        detail = src.registry.get_attempt(summary.attempt_id)
        last = _last_succeeded(detail)
        point = core["mean_of"].get(summary.attempt_id)
        bundle = (last.config or {}).get("bundle_id") if last else None
        desc = None
        if last and bundle and index is not None and store.BUNDLE_ID.fullmatch(str(bundle)):
            d = _describe(src, bundle)
            desc = d.get("content") if isinstance(d, dict) and not is_not_recorded(d) else None
        rows.append({
            "attempt": summary.attempt_id, "round": round_of.get(summary.attempt_id), "iteration": summary.iteration,
            "status": detail.outcome or "open", "score": point.mean if point and last else None,
            "spread": point.dispersion if point and last else None,
            "family": (desc or {}).get("model_family"),
            "features": len((desc or {}).get("features_used") or []) if desc and "features_used" in desc else None,
            "time": sum(e.duration_seconds for e in detail.executions),
            "text": f"{detail.attempt.hypothesis} {detail.attempt.what_changed}",
            "hypothesis": detail.attempt.hypothesis, "describe": desc, "bundle_id": bundle,
            "phase": detail.attempt.phase,
        })
    return rows


def compare_table(src, run_id: str, sort="attempt", order="asc", q=None, round_id=None, family=None) -> dict:
    proto = _protocol(src, run_id)
    rows = _attempt_rows(src, run_id)
    shown = history_mod.all_attempts(rows, sort, order, q, round_id, family, proto["direction"])
    slim = [{k: v for k, v in r.items() if k not in ("describe", "text")} for r in shown]
    return {"run_id": run_id, "direction": proto["direction"], "rows": slim, "count": len(slim),
            "rounds": sorted({r["round"] for r in rows if r["round"]}),
            "families": sorted({r["family"] for r in rows if r["family"]})}


def history(src, run_id: str) -> dict:
    proto = _protocol(src, run_id)
    core = proto["core"]
    chart = data.chart(src, run_id)
    points = [{"attempt": p["attempt_id"], "iteration": p["iteration"], "round": p["round_id"], "score": p["mean"],
               "spread": p["dispersion"], "outcome": p["outcome"]} for p in chart["points"]
              if p["outcome"] == "succeeded"]
    return {"run_id": run_id, "direction": proto["direction"], "metric": proto["metric"],
            "points": history_mod.best_so_far(points, proto["direction"]),
            "baseline": chart["baseline"], "bar": chart["bar"]}


def side_by_side(src, run_id: str, ids: list[str]) -> dict:
    data._core(src, run_id)
    if not 2 <= len(ids) <= history_mod.MAX_SIDE or len(set(ids)) != len(ids):
        raise ValidationError("ids", f"give 2 to {history_mod.MAX_SIDE} different attempt ids")
    rows = {r["attempt"]: r for r in _attempt_rows(src, run_id)}
    details = {}
    for i in ids:
        _valid_attempt(src, run_id, i)
        r = rows[i]
        desc = r["describe"]
        last = _last_succeeded(src.registry.get_attempt(i))
        details[i] = {
            "hypothesis": r["hypothesis"], "what_changed": src.registry.get_attempt(i).attempt.what_changed,
            "features": (desc or {}).get("features_used") if desc else None,
            "derived_features": (desc or {}).get("derived_features") if desc else None,
            "hyperparameters": (desc or {}).get("hyperparameters") if desc else None,
            "folds": _scores_of(last) if last else None, "score": r["score"], "spread": r["spread"],
            "describe": "provided" if desc else "not provided"}
    return {"run_id": run_id, "attempts": history_mod.side_by_side(ids, details)}


def ablation(src, run_id: str) -> dict:
    """Attempts with and without each feature family (display only; no ablation candidate is started)."""
    rows = _attempt_rows(src, run_id)
    known = None  # no snapshot is decoded here: the table is cheap and works from describe.json alone
    feats, derived = set(), []
    for r in rows:
        d = r["describe"]
        if d:
            feats |= set(d.get("features_used") or [])
            derived += d.get("derived_features") or []
    fam = feature_families(sorted(feats), derived, known)
    items = [{"attempt": r["attempt"], "score": r["score"], "spread": r["spread"],
              "features": None if not r["describe"] or "features_used" not in r["describe"] else
              list(r["describe"]["features_used"]) + [x["name"] for x in r["describe"].get("derived_features") or []]}
             for r in rows if r["status"] == "succeeded"]
    return {"run_id": run_id, "families": fam, **importance_mod.ablation_table(items, fam)}
