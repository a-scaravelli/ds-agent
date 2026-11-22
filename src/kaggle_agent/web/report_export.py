"""The final report as the DS reads it: one context for the page, the Markdown export and the print view (ticket 30).

The context is built from the stored report with the holdout MASKED until the DS unsealed it (``data.final_report``),
so the page, the export and the print view show the same thing and none shows a number the DS has not asked for. The
Markdown is rendered here, not read from ``out/reports/<cycle>.md`` (the orchestrator writes the holdout figure there
because ``pipeline.render_text`` is not edited; that file is also where the marked DS section lives). No new dependency.
"""

from . import bundle_data, data, labels_data, submission_data


def _num(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def context(src, run_id: str) -> dict:
    summary = data.run_summary(src, run_id)
    report = data.final_report(src, run_id)
    return {
        "run": summary, "report": report, "holdout": data.holdout(src, run_id),
        "labels": labels_data.labels(src, run_id), "choice": labels_data.choice(src, run_id),
        "bundle": bundle_data.bundle_view(src, run_id),
        "submissions": submission_data.history(src, run_id)["submissions"],
        "suggestions": [s for s in data.suggestions(src) if s["origin_cycle"] == run_id],
        "decisions": [d for d in labels_data.override_statuses(src) if d["cycle"] == run_id],
    }


def _holdout_lines(ctx: dict) -> list[str]:
    h = ctx["holdout"]
    if h["state"] != "unsealed":
        return ["- Status: sealed", f"- {h['text']}"]
    content = (ctx["report"]["content"] or {}).get("holdout") or {}
    lines = [f"- Status: {h['status']}"]
    if h["status"] == "succeeded":
        lines.append(f"- Holdout {h['metric']}: {_num(h['score'])} on {h['rows']} rows")
    elif h["status"] == "waiting":
        lines.append("- The read was requested; the number shows when the run reports.")
    else:
        lines.append(f"- Reason: {h['reason'] or content.get('reason') or '-'}")
    if ctx["choice"]["ds"] and ctx["choice"]["ds"].get("when") == "after the report":
        lines.append("- This holdout figure belongs to the program's own pick, not to the model chosen by the DS.")
    return lines


def markdown(ctx: dict) -> str:
    report = ctx["report"]
    content = report["content"]
    run = ctx["run"]
    if not report["available"] or content is None:
        return f"# Final report of cycle {run['id']}\n\nThe cycle has not written its report yet.\n"
    lines = [f"# Final report of cycle {content.get('cycle_id', run['id'])}", "",
             f"- Created: {content.get('created_at')}", f"- Stop reason: {content.get('stop_reason')}",
             f"- Metric: {content.get('metric')} ({content.get('direction')}), protocol hash "
             f"{str(content.get('protocol_hash'))[:12]}",
             f"- Baseline score: {_num(content.get('baseline_score'))}", ""]
    for label in ctx["labels"]:
        lines.append(f"- **{label['label']}**" + (f": {label['reason']}" if label.get("reason") else ""))
    if ctx["labels"]:
        lines.append("")
    lines += ["## Chosen bundle", ""]
    chosen = content.get("chosen")
    if not chosen:
        lines.append("No usable attempt: no bundle was chosen.")
    else:
        lines += [f"- Bundle: {chosen['bundle_id']} (attempt {chosen['attempt_id']})",
                  f"- Hypothesis: {chosen['hypothesis']}",
                  f"- Frozen-protocol validation {content.get('metric')}: {_num(chosen['validation_mean'])} "
                  f"(dispersion {_num(chosen['validation_dispersion'])})"]
        choice = ctx["choice"]
        if choice["ds"]:
            lines.append(f"- Chosen by the DS: attempt {choice['ds']['attempt']} (reason: {choice['ds']['reason']}); "
                         f"the program's own choice was {choice['program'] or 'none'}")
    lines += ["", "## Holdout", "", *_holdout_lines(ctx), "", "## Rounds", ""]
    for item in content.get("rounds", []):
        comparison = item.get("comparison")
        verdict = "no comparison" if not comparison else (
            f"difference {comparison['difference']:.4f} vs threshold {comparison['threshold']:.4f}: "
            + ("real improvement" if comparison["is_real"] else "not a real improvement"))
        lines += [f"### Round {item['index']} ({item['round_id']})", "", f"- Tasks: {'; '.join(item['tasks'])}",
                  f"- Comparison: {verdict}", f"- Reviewer: {item.get('reviewer') or 'none'}"]
        for c in item["candidates"]:
            lines.append(f"- Candidate {c['id']}: {c['outcome']}, score {_num(c['score'])} - {c['hypothesis']}")
        for g in item.get("guidance", []):
            lines.append(f"- Guidance of the lead: {g['text']}")
        lines.append("")
    lines += ["## What worked", ""]
    lines += [f"- {w['round_id']}: {w['id']} ({_num(w['score'])}) - {w['hypothesis']}" for w in content.get("worked", [])] \
        or ["- nothing beat the baseline by a real margin"]
    lines += ["", "## What failed", ""]
    lines += [f"- {k}: {v}" for k, v in (content.get("failures") or {}).items()] or ["- no failed attempt"]
    lines += ["", "## Suggestions for the next cycle", ""]
    lines += [f"- {s['id']}: {s['text']}" for s in ctx["suggestions"]] or ["- none"]
    lines += ["", "## Persisted bundle and reproduction", ""]
    bundle = ctx["bundle"]
    if bundle.get("available"):
        h = bundle["hashes"]
        lines += [f"- Bundle: {bundle['bundle_id']}", f"- Folder hash: {h['folder']}", f"- Model hash: {h['model']}",
                  f"- Protocol hash: {h['protocol']}", f"- Data hash: {h['data']}"]
        versions = bundle["library_versions"]
        lines.append("- Library versions: " + (versions if isinstance(versions, str) else
                                                ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))))
    else:
        lines.append("- No persisted bundle.")
    lines += ["", "## Submissions", ""]
    lines += [f"- {s['submission_id']}: {s['status']}, public {_num(s['public_score'])}, CV {_num(s['cv'])}, "
              f"holdout {s['holdout'] if isinstance(s['holdout'], str) else _num(s['holdout'])}"
              for s in ctx["submissions"]] or ["- none"]
    lines += ["", "## Decisions of the DS", ""]
    lines += [f"- {d['summary']}: {d['status']}" + (f" ({d['reason']})" if d.get("reason") else "")
              for d in ctx["decisions"]] or ["- none: the program's own choices stand"]
    return "\n".join(lines) + "\n"
