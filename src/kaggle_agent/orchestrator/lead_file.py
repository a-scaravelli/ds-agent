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

from kaggle_agent.agents import CoordinatorInput, RegistrySummary, TraceContext
from kaggle_agent.loop import ids
from kaggle_agent.loop.states import AWAITING_GUIDANCE, AWAITING_PLAN_APPROVAL, AWAITING_REVIEW
from kaggle_agent.registry import Registry
from kaggle_agent.review import Accept, Approve, LeadContext, LeadInput, LeadResponse, Revise

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


def _check(table: dict, name: str) -> dict | None:
    """The validated entry of one table, or None while the table is absent or still empty."""
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
        entries[name] = _check(raw.get(name), name)
    return entries


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
        return tuple(item["text"] for item in self.revisions())

    def plan_number(self) -> int:
        return len(self.revisions()) + 1

    def append(self, plan: int, text: str, when: str) -> None:
        items = self.revisions()
        items.append({"plan": plan, "text": text, "at": when})
        write_atomic(self.path, json.dumps({"revisions": items}, indent=2))


def make_plan_hook(ledger: Ledger):
    """``Hooks.plan``: the coordinator call of the default hook, with the revisions from the ledger.

    The loop passes the revisions it holds in memory, which a resumed run has lost; the ledger has
    them all, so a suspension while re-planning does not lose the lead's revision.
    """

    def plan(ctx, revisions):
        role_input = CoordinatorInput(
            data=ctx.data_summary, metric=ctx.engine.protocol.metric, history=RegistrySummary(),
            guidance=ledger.texts(),
        )
        return ctx.roles.coordinator(TraceContext(ctx.cycle_id, ids.setup_round_id(ctx.cycle_id)),
                                     role_input)

    return plan


# -- the LeadInput -------------------------------------------------------------------------------


class FileLeadInput(LeadInput):
    """Reads the decisions from the lead file; returns None until the file holds the asked one."""

    def __init__(self, path: Path, project_dir: Path, registry: Registry, cycle_id: str,
                 clock: Callable[[], datetime] = _now) -> None:
        self.path = Path(path)
        self.notices_path = Path(project_dir) / NOTICES_FILE
        self.ledger = Ledger(Path(project_dir) / LEDGER_FILE)
        self.registry = registry
        self.cycle_id = cycle_id
        self._clock = clock
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

    # -- the plan ---------------------------------------------------------------

    def _replan_pending(self) -> bool:
        """True if the last revision was written to the ledger but the coordinator did not re-plan yet."""
        done = [t for t in self.registry.list_role_traces(
            self.cycle_id, "coordinator", ids.setup_round_id(self.cycle_id)) if t.output is not None]
        return len(done) < len(self.ledger.revisions()) + 1

    def plan_decision(self, context: LeadContext) -> Approve | Revise | None:
        entries = read_lead(self.path)
        number = self.ledger.plan_number()
        if number > 1 and self._replan_pending():  # a crash after the ledger write: re-plan again
            return Revise(self.ledger.texts()[-1])
        entry = entries["plan"]
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
        if entry is not None:
            if entry["round"] == asked:
                return LeadResponse(entry["action"], entry["text"])
            if entry["round"] < (asked or 0):
                self.notice(f"stale guidance for round {entry['round']} ignored; round {asked} is waiting")
        self._waiting(f"guidance after round {asked}")
        return None

    # -- review -----------------------------------------------------------------

    def review_decision(self, context: LeadContext) -> Accept | None:
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
