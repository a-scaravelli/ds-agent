"""Build the real ``CycleContext`` of a competition from a configuration (design section 15, step 12).

``build_context`` is the only place that wires the pieces together: the Kaggle download, the frozen
evaluation engine over a SQLite registry, the five roles over one Claude client, the Claude bundle
writer, the bundle runner factory, the file lead input and the pipeline hooks. Every external piece
comes from ``Dependencies``, so the whole flow runs with fakes in tests; ``real_dependencies`` is
the only function that constructs real adapters, and it imports them lazily.

Recording: the Claude client is built WITHOUT any ``on_call`` hook, because ``agents.call`` and
``ClaudeBundleWriter`` already store the ``llm_calls`` rows themselves; ``build_context`` checks
that, so a call is recorded exactly once.

Resume: a project that already has a ``project.json`` makes NO Kaggle call; the data in the project
folder is checked against the hashes recorded at start.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kaggle_agent.agents import DataSummary
from kaggle_agent.bundle import BundleRunner
from kaggle_agent.evaluation import EvaluationEngine, Protocol, get_metric
from kaggle_agent.kaggle import KaggleClient, KaggleError, SubmissionSpec
from kaggle_agent.llm import LlmClient, ToolPermissions
from kaggle_agent.loop import CycleContext, Hooks, RoleSet, client_roles
from kaggle_agent.pipeline import PipelineContext, runner_model_store_factory, start_cycle
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.sandbox import Sandbox, SandboxPolicy
from kaggle_agent.writer import ClaudeBundleWriter, WriterTask

from .config import FROM_INFO, FROM_METRIC, FROM_SPEC, Config
from .errors import ConfigError, ProjectError
from .lead_file import FileLeadInput, Ledger, make_plan_hook
from .metric import infer_metric
from .project import (
    LEDGER_FILE,
    ProjectState,
    check_config,
    cycle_id_for,
    file_sha256,
    read_project,
    verify_data,
    write_project,
)
from .summary import build_summary

ROLES = ("coordinator", "data", "feature_engineering", "modeling", "reviewer")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Dependencies:
    """Everything external: the Kaggle client, a Claude client factory, the sandbox and the clocks."""

    kaggle: KaggleClient
    llm: Callable[[], LlmClient]
    sandbox: Sandbox
    clock: Callable[[], datetime] = _utc_now
    sleep: Callable[[float], None] = time.sleep


def real_dependencies() -> Dependencies:
    """The real Kaggle client, Claude client and sandbox (credentials from their usual places)."""
    from kaggle_agent.kaggle.real import RealKaggleClient
    from kaggle_agent.llm.real import RealClaudeClient
    from kaggle_agent.sandbox import SubprocessSandbox

    return Dependencies(kaggle=RealKaggleClient(), llm=lambda: RealClaudeClient(),
                        sandbox=SubprocessSandbox())


@dataclass
class Built:
    """The context of a run and the objects the commands need besides it."""

    ctx: CycleContext
    pctx: PipelineContext
    lead: FileLeadInput
    state: ProjectState
    registry: SqliteRegistry
    client: LlmClient
    config: Config

    def close(self) -> None:
        self.registry.close()


def _resolve_target(config: Config, spec: SubmissionSpec) -> str:
    if config.competition.target != FROM_SPEC:
        return config.competition.target
    if len(spec.columns) != 2:
        raise ConfigError(
            f"competition.target: the sample submission has {len(spec.columns)} columns "
            f"({', '.join(spec.columns)}); set competition.target explicitly"
        )
    return spec.columns[1]


def _resolve_metric(config: Config, evaluation_text: str) -> tuple[str, str]:
    comp = config.competition
    metric = comp.metric
    if metric == FROM_INFO:
        metric, natural = infer_metric(evaluation_text)
    else:
        natural = get_metric(metric).direction
    direction = natural if comp.direction == FROM_METRIC else comp.direction
    if direction != natural:
        raise ConfigError(f"competition.direction: metric {metric!r} is naturally {natural}")
    return metric, direction


def create_project(config: Config, deps: Dependencies) -> ProjectState:
    """Download the data, read info and spec, freeze the choices in ``project.json``."""
    paths, comp = config.paths, config.competition
    for folder in (paths.folder, paths.data, paths.workspace, paths.out):
        folder.mkdir(parents=True, exist_ok=True)
    download = deps.kaggle.download(comp.slug, paths.data)
    info = deps.kaggle.get_competition_info(comp.slug)
    spec = deps.kaggle.get_submission_spec(comp.slug)
    names = {file.name for file in download.files}
    for needed in (comp.train_file, comp.test_file):
        if needed not in names:
            raise ProjectError(f"{needed} is not among the downloaded files ({', '.join(sorted(names))}); "
                               "set competition.train_file / competition.test_file")
    target = _resolve_target(config, spec)
    metric, direction = _resolve_metric(config, info.evaluation)
    try:
        frame = pd.read_csv(paths.data / comp.train_file)
    except (OSError, ValueError) as error:
        raise ProjectError(f"cannot read {comp.train_file}: {error}") from None
    if target not in frame.columns:
        raise ConfigError(f"competition.target: column {target!r} is not in {comp.train_file} "
                          f"(columns: {', '.join(map(str, frame.columns))})")
    protocol = config.protocol.build(metric, direction, target)
    build_summary(frame, protocol)  # fails now, not in the middle of the cycle, on unusable data
    state = ProjectState(
        cycle_id=cycle_id_for(comp.slug), competition=comp.slug, target=target, metric=metric,
        direction=direction, input_kind=get_metric(metric).input_kind, protocol=protocol.to_dict(),
        protocol_config=config.protocol.to_dict(), train_file=comp.train_file, test_file=comp.test_file,
        files={file.name: file.sha256 for file in download.files},
    )
    write_project(paths, state)
    return state


def build_context(config: Config, deps: Dependencies) -> Built:
    """Create the project on the first call, then build the context; creates the cycle if missing."""
    paths = config.paths
    state = read_project(paths)
    if state is None:
        try:
            state = create_project(config, deps)
        except KaggleError as error:
            raise ProjectError(f"Kaggle: {error}") from None
    else:
        check_config(state, config)
        verify_data(paths, state)
    for folder in (paths.workspace, paths.out):
        folder.mkdir(parents=True, exist_ok=True)

    protocol = Protocol.from_dict(state.protocol)
    frame = pd.read_csv(paths.data / state.train_file)
    summary: DataSummary = build_summary(frame, protocol)
    registry = SqliteRegistry(str(paths.registry))
    try:
        if state.cycle_id not in {c.id for c in registry.list_cycles()}:
            start_cycle(registry, state.cycle_id, config.seed_plan(state.metric, state.target),
                        config.limits, project_id=state.competition)
        engine = EvaluationEngine(registry, state.cycle_id, protocol)
        client = deps.llm()
        if client.on_call is not None:
            raise ConfigError("the Claude client must be built without an on_call hook: the roles and "
                              "the writer store the llm_calls rows themselves")
        permissions = ToolPermissions(paths.workspace, ())
        sets = {role: client_roles(client, registry, permissions, config.model_for(role), deps.clock)
                for role in ROLES}
        roles = RoleSet(**{role: getattr(sets[role], role) for role in ROLES})
        task = WriterTask(state.target, state.metric, state.direction, state.input_kind, summary)
        writer = ClaudeBundleWriter(
            client, registry, state.cycle_id, paths.workspace / "staging", task,
            max_repairs=config.runtime.writer_max_repairs,
            allowed_dependencies=config.runtime.allowed_dependencies, model=config.model_for("writer"),
        )
        policy = SandboxPolicy(timeout_seconds=config.runtime.bundle_timeout,
                               hidden_paths=(paths.registry,))
        kind = state.input_kind

        def runner_factory(loaded):
            return BundleRunner(deps.sandbox, policy, loaded, target_column=state.target, input_kind=kind)

        lead = FileLeadInput(config.lead_path, paths.folder, registry, state.cycle_id, deps.clock)
        ledger = Ledger(paths.folder / LEDGER_FILE)
        ctx = CycleContext(
            registry=registry, cycle_id=state.cycle_id, engine=engine, frame=frame,
            data_hash=state.data_hash, data_summary=summary, roles=roles, writer=writer,
            runner_factory=runner_factory, lead=lead, workspace=paths.workspace, now=deps.clock,
            limits=config.limits, hooks=Hooks(plan=make_plan_hook(ledger)),
        )
        pctx = PipelineContext(
            paths.out, runner_model_store_factory(deps.sandbox, policy, target_column=state.target,
                                                  input_kind=kind),
            clock=deps.clock, db_path=str(paths.registry), data_path=str(paths.data / state.train_file),
        )
    except BaseException:
        registry.close()
        raise
    return Built(ctx, pctx, lead, state, registry, client, config)
