"""The lead's decisions as a file: ``FileLeadInput`` implements ``review.LeadInput`` over ``lead.toml``.

The file lives in the project folder, is written by the lead and NEVER edited by the program. It has
at most these tables (any other key is a ``LeadFileError``)::

    [plan]                      # asked at AWAITING_PLAN_APPROVAL
    decision = "approve"        # or "revise"
    text = "what to change"     # revise only
    plan = 1                    # revise only: the number of the plan being revised (1, then 2, ...)

    [guidance]                  # asked after a round (AWAITING_GUIDANCE)
    round = 1                   # the round the answer is for
    action = "guidance"         # or "continue"
    text = "what to try next"   # guidance only

    [review]                    # asked at AWAITING_REVIEW, after the final report
    decision = "accept"

How a decision is applied once. The loop keeps plan revisions only in memory and writes no record for
"continue", so the entries are ADDRESSED: guidance to a round number, a revision to a plan number
(1 plus the revisions already applied). An entry for another question is stale and ignored (a notice
says so) and the cycle waits. Approve and accept need no number because they end their question.
The file is read again at every call, so an edit made while the cycle waits is honoured; a missing
file, a missing table or a table that is still empty answers None and the cycle waits.

Where each decision is recorded: guidance by the loop in the registry (``add_guidance``); the frozen
protocol and the STOPPED state by the loop; a revision by this class in ``lead-ledger.json`` BEFORE
the ``Revise`` is returned, because the plan hook must still know it after a suspension. A guidance
answer of "continue" is deliberately recorded nowhere: after a quota error in the guidance step the
resume asks the same round again and the entry addressed to that round is reused.

Maintainers: ``RoundPipeline._ask_lead`` (loop/round.py) asks again, in a ``while True`` loop, after
every answer it does not accept. A new action must therefore be answered once per request and then
return None, or the cycle would loop on it.

Notices (``notice()``, stale entries, "waiting for ...") go to ``NOTICES.txt``, each distinct text
once, and to ``emitted`` so that the command line can print them.
"""

import json
import tomllib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.agents import (
    CoordinatorInput,
    CoordinatorOutput,
    PendingTrace,
    RegistrySummary,
    TraceContext,
    record_trace,
)
from kaggle_agent.proposals import ShortPlan
from kaggle_agent.loop import ids, role_limits
from kaggle_agent.loop.states import AWAITING_GUIDANCE, AWAITING_PLAN_APPROVAL, AWAITING_REVIEW
from kaggle_agent.registry import Registry
from kaggle_agent.overrides import records_for_cycle
from kaggle_agent.review import Accept, Approve, LeadContext, LeadInput, LeadResponse, Revise

from .applied import applied_outcomes, mark
from .cockpit import read_jsonl
from .errors import LeadFileError
from .project import LEDGER_FILE, NOTICES_FILE, write_atomic

_TABLES = {"plan": ("decision", "text", "plan"), "guidance": ("round", "action", "text"),
           "review": ("decision",)}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# -- reading and validating the file -------------------------------------------------------------


def _text(table: dict, where: str, key: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LeadFileError(f"{where}.{key}: must be a non-empty string")
    return value


def _ordinal(table: dict, where: str, key: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LeadFileError(f"{where}.{key}: must be an integer of at least 1")
    return value


def normalize_entry(name: str, table: dict) -> dict | None:
    """The validated entry of one table (lead.toml or a web decision), or None while absent or empty."""
    if table is None or table == {}:
        return None
    if not isinstance(table, dict):
        raise LeadFileError(f"{name}: must be a table")
    unknown = sorted(set(table) - set(_TABLES[name]))
    if unknown:
        raise LeadFileError(f"{name}.{unknown[0]}: unknown key (allowed: {', '.join(_TABLES[name])})")
    if name == "plan":
        decision = table.get("decision")
        if decision == "approve":
            return {"decision": "approve"}
        if decision == "revise":
            return {"decision": "revise", "text": _text(table, name, "text"),
                    "plan": _ordinal(table, name, "plan")}
        raise LeadFileError("plan.decision: must be 'approve' or 'revise'")
    if name == "guidance":
        number = _ordinal(table, name, "round")
        action = table.get("action")
        if action == "continue":
            return {"round": number, "action": "continue", "text": ""}
        if action == "guidance":
            return {"round": number, "action": "guidance", "text": _text(table, name, "text")}
        raise LeadFileError("guidance.action: must be 'continue' or 'guidance'")
    if table.get("decision") != "accept":
        raise LeadFileError("review.decision: must be 'accept'")
    return {"decision": "accept"}


def read_lead(path: Path) -> dict:
    """Parse and validate the lead file; ``{}`` entries are None. A missing file is all None."""
    entries = {name: None for name in _TABLES}
    if not Path(path).exists():
        return entries
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise LeadFileError(f"{path}: not valid TOML: {error}") from None
    except OSError as error:
        raise LeadFileError(f"{path}: cannot be read: {error}") from None
    unknown = sorted(set(raw) - set(_TABLES))
    if unknown:
        raise LeadFileError(f"{unknown[0]}: unknown key (the lead file has only the tables "
                            f"{', '.join(_TABLES)})")
    for name in _TABLES:
        entries[name] = normalize_entry(name, raw.get(name))
    return entries


def _check(table: dict, name: str) -> dict | None:
    return normalize_entry(name, table)


# -- the ledger of applied revisions -------------------------------------------------------------


class Ledger:
    """Revisions already applied, in ``lead-ledger.json`` (written only by the orchestrator)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def revisions(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8"))["revisions"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise LeadFileError(f"{self.path}: unreadable ledger: {error}") from None

    def texts(self) -> tuple[str, ...]:
        """The revision texts for the coordinator; an own plan of the DS is not a revision text."""
        return tuple(item["text"] for item in self.revisions() if item.get("kind") != "own")

    def plan_number(self) -> int:
        return len(self.revisions()) + 1

    def append(self, plan: int, text: str, when: str, **extra) -> None:
        items = self.revisions()
        items.append({"plan": plan, "text": text, "at": when, **extra})
        write_atomic(self.path, json.dumps({"revisions": items}, indent=2))


OWN_TEXT = "plan written by the DS"
WEB_KINDS = ("plan_approve", "plan_revise", "plan_own", "guidance", "review_accept")


def read_web_decisions(path) -> list[dict]:
    """The valid decisions of ``cockpit/decisions.jsonl`` (written only by the web), in file order.

    A record has ``id``, ``kind`` (one of ``WEB_KINDS``), ``address`` (plan number or round number),
    ``text`` / ``plan`` / ``action`` as the kind needs, ``who`` and ``at``. Invalid records are skipped.
    """
    records, _ = read_jsonl(path)
    valid = []
    for record in records:
        if record.get("kind") in WEB_KINDS and isinstance(record.get("id"), str):
            valid.append(record)
    return valid


def web_entry(record: dict) -> dict | None:
    """A web decision as the same validated entry a lead.toml table gives (own plans: ``kind`` own)."""
    kind, address = record["kind"], record.get("address")
    try:
        if kind == "plan_approve":
            return normalize_entry("plan", {"decision": "approve"})
        if kind == "plan_revise":
            return normalize_entry("plan", {"decision": "revise", "text": record.get("text"), "plan": address})
        if kind == "plan_own":
            ShortPlan.from_dict(record["plan"]["plan"])
            tasks = record["plan"].get("tasks")
            if not isinstance(tasks, list) or not tasks:
                return None
            return {"decision": "own", "plan": address, "fields": record["plan"],
                    "reason": record.get("reason", "")}
        if kind == "guidance":
            return normalize_entry("guidance", {"round": address, "action": record.get("action"),
                                                "text": record.get("text")})
        return normalize_entry("review", {"decision": "accept"})
    except (LeadFileError, KeyError, TypeError, ValueError, AttributeError):
        return None


def make_plan_hook(ledger: Ledger):
    """``Hooks.plan``: the coordinator call of the default hook, with the revisions from the ledger.

    The loop passes the revisions it holds in memory, which a resumed run has lost; the ledger has
    them all, so a suspension while re-planning does not lose the lead's revision. When the last
    ledger item is an own plan of the DS, the coordinator is not called: the typed plan becomes the
    coordinator output and is traced with the same shape as a real one (``plan`` and ``tasks``), with
    ``author: lead`` in the trace input because the registry has no author field.
    """

    def plan(ctx, revisions):
        items = ledger.revisions()
        if items and items[-1].get("kind") == "own":
            return _own_plan(ctx, items)
        role_input = CoordinatorInput(
            data=ctx.data_summary, metric=ctx.engine.protocol.metric, history=RegistrySummary(),
            guidance=ledger.texts(), limits=role_limits(ctx, 0),
        )
        return ctx.roles.coordinator(TraceContext(ctx.cycle_id, ids.setup_round_id(ctx.cycle_id)),
                                     role_input)

    return plan


def _own_plan(ctx, items: list[dict]) -> CoordinatorOutput:
    round_id = ids.setup_round_id(ctx.cycle_id)
    done = [t for t in ctx.registry.list_role_traces(ctx.cycle_id, "coordinator", round_id)
            if t.output is not None]
    if len(done) >= len(items) + 1:  # a crash after the trace: do not trace twice
        return CoordinatorOutput.from_dict(done[-1].output)
    last = items[-1]
    fields = last["fields"]
    output = CoordinatorOutput(plan=ShortPlan.from_dict(fields["plan"]), tasks=tuple(fields["tasks"]))
    pending = PendingTrace(
        cycle_id=ctx.cycle_id, role="coordinator", round_id=round_id, input_type="LeadPlan",
        input={"author": "lead", "plan_number": last["plan"], "reason": last.get("reason", "")},
        outcome="ok", llm_call_id=None, output_type=type(output).__name__, output=output.to_dict(),
        created_at=ctx.now().isoformat())
    record_trace(ctx.registry, pending)
    return output


# -- the LeadInput -------------------------------------------------------------------------------


class FileLeadInput(LeadInput):
    """Reads the decisions from the lead file and from the web decisions; None until one answers.

    ONE implementation, one set of consumption rules. Precedence: a web decision addressed to the
    asked question wins; a different lead.toml entry for the same question is shadowed (a notice and a
    ``decision_stale`` event say so). A decision with an old address is stale and ignored.
    """

    def __init__(self, path: Path, project_dir: Path, registry: Registry, cycle_id: str,
                 clock: Callable[[], datetime] = _now, cockpit=None, emitter=None,
                 first_cycle: str | None = None, ledger_name: str = LEDGER_FILE) -> None:
        self.path = Path(path)
        self.notices_path = Path(project_dir) / NOTICES_FILE
        self.ledger = Ledger(Path(project_dir) / ledger_name)
        self.first_cycle = first_cycle or cycle_id
        self.registry = registry
        self.cycle_id = cycle_id
        self._clock = clock
        self.cockpit = cockpit
        self.emitter = emitter
        self.emitted: list[str] = []

    # -- notices --------------------------------------------------------------

    def _known(self) -> set[str]:
        if not self.notices_path.exists():
            return set()
        lines = self.notices_path.read_text(encoding="utf-8").splitlines()
        return {line.split(" ", 1)[1] for line in lines if " " in line}

    def notice(self, message: str) -> None:
        """Append a message to ``NOTICES.txt`` (each distinct text once) and remember it."""
        message = " ".join(str(message).split())
        if message in self._known():
            return
        self.notices_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.notices_path, "a", encoding="utf-8") as handle:
            handle.write(f"{self._clock().isoformat()} {message}\n")
        self.emitted.append(message)

    def _waiting(self, what: str) -> None:
        self.notice(f"waiting for {what} in {self.path}")
        if self.emitter is not None:
            self.emitter.emit("waiting", question=what)

    # -- the web source ---------------------------------------------------------

    def _web(self) -> list[dict]:
        if self.cockpit is None:
            return []
        return records_for_cycle(read_web_decisions(self.cockpit.decisions), self.cycle_id, self.first_cycle)

    def _applied_ids(self) -> dict[str, str]:
        return applied_outcomes(self.cockpit)

    def _mark(self, record: dict, outcome: str, reason: str = "") -> None:
        """Once per decision: ``decisions-applied.jsonl`` and the event (see ``applied.mark``)."""
        mark(self.cockpit, self.emitter, record, outcome, reason)

    def _conflict(self, what: str, record: dict, toml_entry: dict | None, web_entry_: dict | None) -> None:
        if toml_entry is None or web_entry_ is None:
            return
        if {k: v for k, v in toml_entry.items() if k != "fields"} == \
                {k: v for k, v in web_entry_.items() if k not in ("fields", "reason")}:
            return
        self.notice(f"conflict: the web decision {record['id']} for {what} wins over lead.toml, "
                    "which answers the same question differently")
        if self.emitter is not None:
            self.emitter.emit("decision_stale", decision=record["id"], kind=record["kind"],
                              reason="conflict", source="lead.toml")

    def _for(self, kinds: tuple, number: int) -> tuple[dict | None, dict | None]:
        """The latest web decision addressed to this question (and its entry); stale ones are marked."""
        found = None
        for record in self._web():
            if record["kind"] not in kinds:
                continue
            address = record.get("address")
            if kinds == ("review_accept",) or address == number:
                entry = web_entry(record)
                if entry is not None:
                    found = (record, entry)
            elif isinstance(address, int) and address < number:
                self._mark(record, "stale", f"addressed to {address}, the question is {number}")
        return found if found else (None, None)

    # -- the plan ---------------------------------------------------------------

    def _replan_pending(self) -> bool:
        """True if the last revision was written to the ledger but the coordinator did not re-plan yet."""
        done = [t for t in self.registry.list_role_traces(
            self.cycle_id, "coordinator", ids.setup_round_id(self.cycle_id)) if t.output is not None]
        return len(done) < len(self.ledger.revisions()) + 1

    def plan_decision(self, context: LeadContext) -> Approve | Revise | None:
        number = self.ledger.plan_number()
        items = self.ledger.revisions()
        if number > 1 and self._replan_pending():  # a crash after the ledger write: re-plan again
            return Revise(items[-1]["text"])
        if items and items[-1].get("kind") == "own":  # the typed plan counts as the DS's approval
            return Approve()
        entries = read_lead(self.path)
        record, wentry = self._for(("plan_approve", "plan_revise", "plan_own"), number)
        toml_entry = entries["plan"]
        if toml_entry is not None and toml_entry["decision"] == "revise" and toml_entry["plan"] != number:
            toml_entry_same = None
        else:
            toml_entry_same = toml_entry
        if record is not None:
            self._conflict(f"plan {number}", record, toml_entry_same, wentry)
            self._mark(record, "applied")
            if wentry["decision"] == "approve":
                return Approve()
            if wentry["decision"] == "own":
                self.ledger.append(number, OWN_TEXT, self._clock().isoformat(), kind="own",
                                   fields=wentry["fields"], reason=wentry["reason"])
                return Revise(OWN_TEXT)
            self.ledger.append(number, wentry["text"], self._clock().isoformat())
            return Revise(wentry["text"])
        entry = toml_entry
        if entry is not None and entry["decision"] == "approve":
            return Approve()
        if entry is not None:
            if entry["plan"] == number:
                self.ledger.append(number, entry["text"], self._clock().isoformat())
                return Revise(entry["text"])
            if entry["plan"] < number:
                self.notice(f"stale plan revision {entry['plan']} ignored; plan {number} is waiting")
        self._waiting(f"plan approval (plan {number})")
        return None

    # -- guidance ---------------------------------------------------------------

    def guidance(self, context: LeadContext) -> LeadResponse | None:
        entry = read_lead(self.path)["guidance"]
        asked = context.round_position
        record, wentry = self._for(("guidance",), asked or 0)
        if record is not None:
            if entry is not None and entry["round"] == asked:
                self._conflict(f"the guidance of round {asked}", record, entry, wentry)
            self._mark(record, "applied")
            return LeadResponse(wentry["action"], wentry["text"])
        if entry is not None:
            if entry["round"] == asked:
                return LeadResponse(entry["action"], entry["text"])
            if entry["round"] < (asked or 0):
                self.notice(f"stale guidance for round {entry['round']} ignored; round {asked} is waiting")
        self._waiting(f"guidance after round {asked}")
        return None

    # -- review -----------------------------------------------------------------

    def review_decision(self, context: LeadContext) -> Accept | None:
        record, wentry = self._for(("review_accept",), 0)
        if record is not None:
            self._mark(record, "applied")
            return Accept()
        if read_lead(self.path)["review"] is not None:
            return Accept()
        self._waiting("review of the final report")
        return None


# -- what the lead has to fill next --------------------------------------------------------------


def describe_pending(state: str, registry: Registry, cycle_id: str, lead_path: Path,
                     ledger: Ledger) -> list[str]:
    """Lines telling the lead exactly which table and keys to fill next ([] if nothing is awaited)."""
    where = f"Edit {lead_path} and add:"
    if state == AWAITING_PLAN_APPROVAL:
        number = ledger.plan_number()
        return [f"Waiting for plan approval (plan {number}). {where}", "",
                "    [plan]", '    decision = "approve"', "",
                "or, to ask for a new plan:", "",
                "    [plan]", '    decision = "revise"', f"    plan = {number}",
                '    text = "what to change"']
    if state == AWAITING_GUIDANCE:
        setup = ids.setup_round_id(cycle_id)
        position = len([r for r in registry.list_rounds_with_candidates(cycle_id) if r.round.id != setup])
        return [f"Waiting for guidance after round {position}. {where}", "",
                "    [guidance]", f"    round = {position}", '    action = "guidance"',
                '    text = "what to try next"', "",
                "or, to go on without guidance:", "",
                "    [guidance]", f"    round = {position}", '    action = "continue"']
    if state == AWAITING_REVIEW:
        return [f"Waiting for the review of the final report. Read the report, then {where[0].lower()}{where[1:]}", "",
                "    [review]", '    decision = "accept"']
    return []
