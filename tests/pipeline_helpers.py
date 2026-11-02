"""Builders for the pipeline tests: the loop fakes plus a fake model store, all offline."""

import dataclasses

from loop_helpers import CYCLE, FRAME, Boom, by_round, make_setup  # noqa: F401
from registry_helpers import PLAN  # noqa: F401

from kaggle_agent.loop import CycleView
from kaggle_agent.pipeline import PipelineContext, run_pipeline
from kaggle_agent.pipeline.fake import FakeModelStore
from kaggle_agent.review import LeadResponse, SimulatedLead

GUIDANCE_TEXT = "Try gradient boosting with shallow trees"


@dataclasses.dataclass
class PipelineSetup:
    """A loop ``Setup`` plus the pipeline context and the fake model store."""

    setup: object
    pctx: PipelineContext
    store: FakeModelStore

    @property
    def registry(self):
        return self.setup.registry

    @property
    def ctx(self):
        return self.setup.ctx

    @property
    def engine(self):
        return self.setup.engine

    def run(self, **changes):
        ctx = dataclasses.replace(self.ctx, **changes) if changes else self.ctx
        return run_pipeline(ctx, self.pctx)

    def view(self):
        return CycleView(self.registry, CYCLE, self.engine)


def make_pipeline(tmp_path, *, store=None, guidance=True, **kwargs) -> PipelineSetup:
    """A cycle that improves in round 1 and then stagnates (3 rounds, 2 candidates, K=2)."""
    kwargs.setdefault("rounds", 5)
    kwargs.setdefault("rule", by_round([0.1, 0.1, 0.1], baseline=0.3))
    if guidance and "lead" not in kwargs:
        kwargs["lead"] = SimulatedLead(guidance=[LeadResponse("guidance", GUIDANCE_TEXT)])
    setup = make_setup(tmp_path, **kwargs)
    store = store or FakeModelStore()
    out = tmp_path / "out"
    pctx = PipelineContext(out, lambda loaded: store, db_path="registry.db", data_path="data.csv")
    return PipelineSetup(setup, pctx, store)


MAJORITY_MODEL = '''\
import json
import os


def fit(train):
    label = int(train["y"].mode().iloc[0])
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "majority.json"), "w") as handle:
        json.dump({"label": label}, handle)


def predict(bundle_id, X):
    with open(os.path.join(os.environ["BUNDLE_MODEL_DIR"], "majority.json")) as handle:
        label = json.load(handle)["label"]
    return [label] * len(X)
'''


class MajorityWriter:
    """A bundle writer for the tests with a real process: fit saves the majority label, predict reads it."""

    def __init__(self) -> None:
        self.requests = []

    def write(self, request):
        import json

        self.requests.append(request)
        folder = request.folder
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"bundle_id": request.bundle_id, "dependencies": [],
                    "entry_points": {"fit": "model:fit", "predict": "model:predict"}}
        (folder / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
        # a comment that differs per candidate keeps the fingerprints apart
        (folder / "model.py").write_text(f"# {request.attempt.hypothesis} {request.bundle_id}\n{MAJORITY_MODEL}",
                                         encoding="utf-8")
        return folder


def make_real_pipeline(tmp_path, sandbox=None):
    """A pipeline whose bundles run in real child processes (few partitions, one candidate per round).

    The registry is a file, so the commands can open it read-only. Returns (pipe, db path).
    """
    from kaggle_agent.bundle import BundleRunner
    from kaggle_agent.pipeline import runner_model_store_factory
    from kaggle_agent.registry import SqliteRegistry
    from kaggle_agent.sandbox import SandboxPolicy, SubprocessSandbox

    sandbox = sandbox or SubprocessSandbox()
    sandbox_policy = SandboxPolicy(timeout_seconds=120)
    db = tmp_path / "registry.db"
    setup = make_setup(
        tmp_path, rounds=2, candidates=1, stagnation_k=1, registry=SqliteRegistry(str(db)),
        writer=MajorityWriter(), lead=SimulatedLead(),
    )

    def factory(loaded):
        return BundleRunner(sandbox, sandbox_policy, loaded, target_column="y", input_kind="labels")

    setup.ctx = dataclasses.replace(setup.ctx, runner_factory=factory)
    pctx = PipelineContext(
        tmp_path / "out", runner_model_store_factory(sandbox, sandbox_policy, target_column="y",
                                                    input_kind="labels"),
    )
    return PipelineSetup(setup, pctx, None), db
