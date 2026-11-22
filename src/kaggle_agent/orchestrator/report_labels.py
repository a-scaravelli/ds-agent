"""The labels of a run on the orchestrator side, and the marked 'DS decisions' section of the Markdown report.

The registry report is write-once, so the labels live in ``out/reports/<cycle>.md`` between
``<!-- ds-decisions:begin -->`` and ``<!-- ds-decisions:end -->`` (the ``refresh_report_file`` precedent of the
submission section): the section is rewritten, nothing else in the file is touched. It shows the label, the
reason the DS gave, the numbers the DS saw and, next to a choice, the program's own pick.
"""

from pathlib import Path

from kaggle_agent.overrides import records_for_cycle, run_labels
from kaggle_agent.pipeline import report_path

from .applied import applied_outcomes
from .cockpit import CockpitFiles, read_jsonl
from .cycles import entry_for, read_cycles

SECTION_BEGIN = "<!-- ds-decisions:begin -->"
SECTION_END = "<!-- ds-decisions:end -->"


def collect_labels(files: CockpitFiles, folder: Path, registry, cycle_id: str, first_cycle: str) -> list[dict]:
    decisions, _ = read_jsonl(files.decisions)
    mine = records_for_cycle(decisions, cycle_id, first_cycle)
    reads = records_for_cycle(read_jsonl(files.holdout_reads)[0], cycle_id, first_cycle)
    entry = entry_for(read_cycles(folder), cycle_id)
    facts: dict = {}
    if entry is not None:
        origin = entry.get("origin", {}).get("cycle_id")
        facts["origin_read"] = bool(origin) and registry.get_holdout_outcome(origin) is not None
        facts["protocol_hash"] = registry.get_cycle(cycle_id).protocol_hash
    facts["seed_reruns"] = _seed_reruns(registry, cycle_id)
    return run_labels(cycle_id, first_cycle, mine, applied_outcomes(files), reads, entry, facts)


def _seed_reruns(registry, cycle_id: str) -> list[str]:
    out = []
    for summary in registry.list_attempts(cycle_id):
        detail = registry.get_attempt(summary.attempt_id)
        if detail.executions and (detail.executions[-1].config or {}).get("seed_offset"):
            out.append(detail.attempt.id)
    return out


def section_lines(labels: list[dict], choice: dict | None, comparisons: list, note: str | None) -> list[str]:
    lines = [SECTION_BEGIN, "## Decisions of the DS", ""]
    if not labels:
        lines.append("- none: the program's own choices stand")
    for item in labels:
        text = f"- {item['label']}"
        if item.get("attempt"):
            text += f": attempt {item['attempt']}"
        if item.get("reason"):
            text += f" - reason: {item['reason']}"
        if item.get("status") not in (None, "applied"):
            text += f" ({item['status']})"
        lines.append(text)
    if choice:
        lines.append(f"- the program's own choice was {choice.get('program_choice') or 'none'}; the model "
                     f"of this report is {choice.get('attempt')} (chosen by the DS)")
    for comparison in comparisons:
        lines.append(f"- bar numbers for {comparison['candidate']}: difference {comparison['difference']:.5f}, "
                     f"bar {comparison['threshold']:.5f}, stored verdict "
                     f"{'real' if comparison['is_real'] else 'not real'}")
    if note:
        lines.append(f"- {note}")
    lines.append(SECTION_END)
    return lines


def refresh_ds_section(out: Path, cycle_id: str, labels: list[dict], choice: dict | None = None,
                       comparisons: list | None = None, note: str | None = None) -> None:
    """Rewrite the marked section of the Markdown report; a missing report file is left missing."""
    path = report_path(out, cycle_id)
    if not path.exists():
        return
    section = "\n".join(section_lines(labels, choice, comparisons or [], note)) + "\n"
    text = path.read_text(encoding="utf-8")
    if SECTION_BEGIN in text and SECTION_END in text:
        head, rest = text.split(SECTION_BEGIN, 1)
        tail = rest.split(SECTION_END, 1)[1].lstrip("\n")
        text = head + section + (("\n" + tail) if tail else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def override_comparisons(registry, cycle_id: str) -> list[dict]:
    """The numbers of the comparisons the DS overrode (the bar and the difference), from the STORED rows."""
    inner = getattr(registry, "_inner", registry)
    stored = inner.list_comparisons(cycle_id)
    effective = registry.list_comparisons(cycle_id)
    return [{"candidate": s.candidate_attempt_id, "difference": s.difference, "threshold": s.threshold,
             "is_real": s.is_real} for s, e in zip(stored, effective) if e.is_real and not s.is_real]
