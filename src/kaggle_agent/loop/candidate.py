"""One candidate end to end: propose, build a bundle, run it, fix it, close it.

Every step is idempotent. The state of a candidate is read back from the registry (attempt
missing, registered but not executed, failed and waiting for a fix, executed but not closed,
closed), so a resumed run continues where the previous one stopped and never repeats a finished
step. Ids are deterministic (``ids``).

The roles never call each other: this module passes the findings of the data role to feature
engineering and modeling, and the feature engineering attempt to modeling.
"""

import dataclasses
import hashlib
from pathlib import Path

from kaggle_agent.agents import (
    DataFindings,
    DataInput,
    FeatureEngineeringInput,
    ModelingInput,
    RoleRejectedError,
    TraceContext,
    record_trace,
)
from kaggle_agent.budget import CycleLimits
from kaggle_agent.bundle import (
    BundleError,
    BundleLayoutError,
    attempt_with_bundle,
    load_bundle,
    register_bundle,
    score_bundle,
)
from kaggle_agent.proposals import Attempt, ValidationError
from kaggle_agent.registry import (
    AttemptDetail,
    DuplicateAttemptError,
    ExecutionRecord,
    NotFoundError,
    compute_fingerprint,
)

from . import ids
from .context import CycleContext
from .derive import PLACEHOLDER_PREFIX, CycleView
from .writer import BundleRequest, FixRequest, RoleOutputs

# a proposal that goes back to the roles with the reason, instead of ending the cycle
REJECTIONS = (RoleRejectedError, ValidationError, BundleError, DuplicateAttemptError, NotFoundError)
MAX_REASON = 300


def clean(reason: object) -> str:
    """One line of at most ``MAX_REASON`` characters."""
    text = " ".join(str(reason).split())
    return text if len(text) <= MAX_REASON else text[: MAX_REASON - 3] + "..."


def content_hash(folder: Path) -> str:
    """Hash of the code of a bundle folder, without its manifest.

    The manifest holds the bundle id, which is unique per bundle, so it is left out: two bundles
    with the same code are then recognised as duplicates by the registry fingerprint.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        relative = path.relative_to(folder)
        if relative.name == "bundle.json" and len(relative.parts) == 1:
            continue
        if "__pycache__" in relative.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class CandidateRunner:
    """Runs the candidates of a cycle; holds the injected context, the limits and the derived view."""

    def __init__(self, ctx: CycleContext, limits: CycleLimits, view: CycleView) -> None:
        self.ctx = ctx
        self.limits = limits
        self.view = view
        self.registry = ctx.registry

    # -- entry point ----------------------------------------------------------

    def run(self, round_id: str, index: int, task: str) -> AttemptDetail:
        """Bring candidate ``index`` of the round to a closed attempt and return it."""
        cid = ids.candidate_id(round_id, index)
        detail = self.view.detail(cid)
        outputs = None
        if detail is None:
            detail, outputs = self._propose(round_id, cid, task)
        if not detail.closed:
            self._execute(round_id, cid, outputs)
            detail = self.view.detail(cid)
        return detail

    # -- propose ----------------------------------------------------------------

    def _propose(self, round_id: str, cid: str, task: str):
        ctx = self.ctx
        trace_ctx = TraceContext(ctx.cycle_id, round_id)
        plan = self.registry.get_cycle(ctx.cycle_id).plan
        findings = None
        reason = None
        rejections = 0
        while True:
            text = task
            if reason is not None:
                text = f"{task}\n\nYour previous proposal was rejected: {reason}. Propose a valid one."
            try:
                if findings is None:
                    findings = ctx.roles.data(trace_ctx, DataInput(ctx.data_summary, text))
                history = self.view.registry_summary()
                fe = ctx.roles.feature_engineering(
                    trace_ctx, FeatureEngineeringInput(cid, plan, findings, text, history)
                )
                modeling = ctx.roles.modeling(
                    trace_ctx, ModelingInput(cid, plan, findings, text, history, fe.output)
                )
                outputs = RoleOutputs(findings, fe.output, modeling.output)
                self._register(round_id, cid, outputs, (fe, modeling))
                return self.view.detail(cid), outputs
            except REJECTIONS as exc:
                reason = clean(exc.reason if isinstance(exc, RoleRejectedError) else exc)
                rejections += 1
                if rejections >= self.limits.max_rejections:
                    self._placeholder(round_id, cid, rejections, reason)
                    return self.view.detail(cid), None

    def _free_bundle_id(self, cid: str) -> str:
        taken = {b.id for b in self.registry.list_bundles(self.ctx.cycle_id)}
        number = 1
        while ids.bundle_id(cid, number) in taken:
            number += 1
        return ids.bundle_id(cid, number)

    def _load(self, request: BundleRequest):
        folder = self.ctx.writer.write(request)
        loaded = load_bundle(folder)
        if loaded.bundle_id != request.bundle_id:
            raise BundleLayoutError(
                f"the manifest has bundle_id {loaded.bundle_id!r}, expected {request.bundle_id!r}"
            )
        return loaded

    def _register(self, round_id: str, cid: str, outputs: RoleOutputs, results) -> None:
        ctx = self.ctx
        cycle = self.registry.get_cycle(ctx.cycle_id)
        bundle_id = self._free_bundle_id(cid)
        request = BundleRequest(
            attempt=outputs.modeling,
            outputs=outputs,
            bundle_id=bundle_id,
            folder=ctx.workspace / "bundles" / bundle_id,
        )
        loaded = self._load(request)
        config = {
            "feature_engineering": outputs.feature_engineering.what_changed,
            "modeling": outputs.modeling.what_changed,
        }
        fingerprint = compute_fingerprint(
            config, content_hash(loaded.folder), ctx.data_hash, cycle.protocol_hash
        )
        phase = "baseline" if round_id == ids.setup_round_id(ctx.cycle_id) else "iteration"
        attempt = attempt_with_bundle(dataclasses.replace(outputs.modeling, id=cid, phase=phase), bundle_id)
        register_bundle(self.registry, loaded, ctx.cycle_id, reference=str(loaded.folder))
        self.registry.add_attempt(ctx.cycle_id, attempt, fingerprint, round_id=round_id)
        # the pending traces of the candidate roles are written now, so candidate_id is set
        for result in results:
            if getattr(result, "trace", None) is not None:
                record_trace(self.registry, result.trace, candidate_id=cid)

    def _placeholder(self, round_id: str, cid: str, rejections: int, reason: str) -> None:
        """Record a candidate that never produced a valid proposal as an inconclusive attempt."""
        ctx = self.ctx
        cycle = self.registry.get_cycle(ctx.cycle_id)
        text = f"{PLACEHOLDER_PREFIX} {rejections} rejections: {reason}"
        attempt = Attempt(
            id=cid,
            phase="baseline" if round_id == ids.setup_round_id(ctx.cycle_id) else "iteration",
            hypothesis="No valid proposal was produced for this candidate",
            rationale=text,
            what_changed="nothing: the proposal was rejected every time",
            measure="not applicable",
            exploratory=True,
        )
        fingerprint = compute_fingerprint(
            {"placeholder": cid}, "placeholder", ctx.data_hash, cycle.protocol_hash
        )
        self.registry.add_attempt(ctx.cycle_id, attempt, fingerprint, round_id=round_id)
        self.registry.add_execution(
            ExecutionRecord(
                id=ids.execution_id(cid, 1),
                attempt_id=cid,
                config={"placeholder": True},
                seed=ctx.engine.protocol.seed,
                code_hash="none",
                data_hash=ctx.data_hash,
                protocol_hash=cycle.protocol_hash,
                metrics=(),
                duration_seconds=0.0,
                outcome="inconclusive",
                failure_kind="hypothesis_falsified",
                reason=text,
                resources={"placeholder": True},
            )
        )
        self.registry.close_attempt(cid, f"Inconclusive: {text}")

    # -- execute, fix, close ---------------------------------------------------

    def _bundles_of(self, detail: AttemptDetail):
        base = detail.attempt.bundle_id
        return [
            b
            for b in self.registry.list_bundles(self.ctx.cycle_id)
            if b.id == base or b.id.startswith(base + "-f")
        ]

    def _execute(self, round_id: str, cid: str, outputs: RoleOutputs | None) -> None:
        ctx = self.ctx
        while True:
            detail = self.view.detail(cid)
            executions = detail.executions
            bundles = self._bundles_of(detail)
            if executions:
                last = executions[-1]
                if last.outcome == "succeeded" or not self._may_fix(last, len(executions)):
                    self._close(cid, detail)
                    return
                if len(bundles) == len(executions):  # the fix for the last failure is not written
                    outputs = outputs or self._recover_outputs(round_id, cid, detail)
                    if not self._write_fix(cid, detail, last, bundles, outputs):
                        continue  # a terminal execution was recorded: the next pass closes
                    bundles = self._bundles_of(detail)
            record = bundles[-1]
            loaded = load_bundle(Path(record.reference))
            score = score_bundle(
                ctx.engine,
                ctx.runner_factory(loaded),
                ctx.frame,
                attempt_id=cid,
                execution_id=ids.execution_id(cid, len(executions) + 1),
                config={"bundle_id": record.id, "round_id": round_id},
                data_hash=ctx.data_hash,
            )
            self.registry.add_execution(score.execution)

    def _may_fix(self, last: ExecutionRecord, count: int) -> bool:
        """A crashed run with room left may be fixed; integrity failures and timeouts never."""
        if last.failure_kind != "code_error":
            return False
        if (last.resources or {}).get("retryable") is False:
            return False
        return count < 1 + self.limits.fix_retries

    def _write_fix(self, cid, detail, last, bundles, outputs) -> bool:
        """Ask the writer for a new bundle of the same attempt; False if none could be made."""
        ctx = self.ctx
        previous = bundles[-1]
        retry = len(bundles)
        new_id = ids.fix_bundle_id(detail.attempt.bundle_id, retry)
        reason = clean(last.reason or "the run failed")
        for _ in range(self.limits.max_rejections):
            request = BundleRequest(
                attempt=detail.attempt,
                outputs=outputs,
                bundle_id=new_id,
                folder=ctx.workspace / "bundles" / new_id,
                fix=FixRequest(retry, reason, previous.id),
            )
            try:
                loaded = self._load(request)
            except (BundleError, ValidationError) as exc:
                reason = clean(exc)
                continue
            register_bundle(self.registry, loaded, ctx.cycle_id, reference=str(loaded.folder))
            return True
        self.registry.add_execution(
            ExecutionRecord(
                id=ids.execution_id(cid, len(detail.executions) + 1),
                attempt_id=cid,
                config={"bundle_id": previous.id},
                seed=ctx.engine.protocol.seed,
                code_hash=last.code_hash,
                data_hash=ctx.data_hash,
                protocol_hash=last.protocol_hash,
                metrics=(),
                duration_seconds=0.0,
                outcome="failed",
                failure_kind="code_error",
                reason=f"no valid fix bundle: {reason}",
                resources={"sandbox_status": "invalid_fix_bundle", "retryable": False},
            )
        )
        return False

    def _recover_outputs(self, round_id: str, cid: str, detail: AttemptDetail) -> RoleOutputs:
        """The role outputs of a candidate after a resume, from its traces (best effort)."""
        traces = self.registry.list_role_traces(self.ctx.cycle_id, round_id=round_id)
        findings = DataFindings("(findings not recovered after a resume)")
        for trace in traces:
            if trace.role == "data" and trace.output is not None:
                findings = DataFindings.from_dict(trace.output)
        feature_engineering = detail.attempt
        for trace in traces:
            if trace.role == "feature_engineering" and trace.candidate_id == cid and trace.output:
                feature_engineering = Attempt.from_dict(trace.output)
        return RoleOutputs(findings, feature_engineering, detail.attempt)

    def _close(self, cid: str, detail: AttemptDetail) -> None:
        last = detail.executions[-1]
        if last.outcome == "succeeded":
            metric = next(m for m in last.metrics if m.partition == "validation")
            conclusion = (
                f"Validation {self.ctx.engine.protocol.metric}: {metric.mean:.4f} "
                f"(dispersion {metric.dispersion:.4f}) after {len(detail.executions)} execution(s)."
            )
        else:
            conclusion = (
                f"Failed ({last.failure_kind}) after {len(detail.executions)} execution(s): "
                f"{clean(last.reason)}"
            )
        self.registry.close_attempt(cid, conclusion)
