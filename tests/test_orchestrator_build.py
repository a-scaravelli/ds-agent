"""build_context (cases 1 and 2): a real CycleContext from a configuration, everything injected."""

import json

from orchestrator_helpers import SLUG, World, drive_to_stopped

from kaggle_agent.evaluation import EvaluationEngine
from kaggle_agent.loop import RoleSet
from kaggle_agent.orchestrator import build_context, load_config
from kaggle_agent.pipeline import PipelineContext
from kaggle_agent.writer import ClaudeBundleWriter


# 1
def test_a_context_is_built_from_the_config_with_every_external_piece_injected(tmp_path):
    world = World(tmp_path)
    config = load_config(world.config)
    built = build_context(config, world.deps)
    try:
        assert ("download", SLUG) in world.kaggle.calls
        names = {call[0] for call in world.kaggle.calls}
        assert {"download", "get_competition_info", "get_submission_spec"} <= names
        assert not any(call[0] in ("send", "get_submission") for call in world.kaggle.calls)
        assert (config.paths.data / "train.csv").is_file() and (config.paths.data / "test.csv").is_file()
        ctx = built.ctx
        assert isinstance(ctx.roles, RoleSet) and isinstance(ctx.writer, ClaudeBundleWriter)
        assert isinstance(ctx.engine, EvaluationEngine) and ctx.engine.protocol.metric == "accuracy"
        assert ctx.engine.protocol.target == "y" and ctx.engine.protocol.direction == "maximize"
        assert ctx.registry.get_cycle("toy-1").protocol_hash is None  # frozen only at the lead's approval
        assert isinstance(built.pctx, PipelineContext) and callable(built.pctx.model_store_factory)
        assert config.paths.registry.is_file() and config.paths.workspace.is_dir() and config.paths.out.is_dir()
        assert ctx.hooks.plan is not None and callable(ctx.runner_factory)
        summary = ctx.data_summary
        assert [c.name for c in summary.columns] == ["id", "x1", "x2", "y"] and summary.target == "y"
        assert summary.rows == 48  # the development part only: 60 rows minus the 20 percent holdout
        assert "60" not in json.dumps(summary.to_dict()["notes"])  # sizes and statistics only, no rows
        assert all(set(c.to_dict()) == {"name", "dtype", "missing_fraction", "distinct_values"}
                   for c in summary.columns)
        assert world.llm_factory_calls == 1 and world.claude.prompts == []  # nothing asked of Claude yet
    finally:
        built.close()


def test_building_again_makes_no_kaggle_call_and_creates_no_second_cycle(tmp_path):
    world = World(tmp_path)
    config = load_config(world.config)
    build_context(config, world.deps).close()
    calls = list(world.kaggle.calls)
    built = build_context(config, world.deps)
    try:
        assert world.kaggle.calls == calls and len(built.registry.list_cycles()) == 1
    finally:
        built.close()


# 2
def test_the_claude_calls_are_recorded_once(tmp_path):
    other = World(tmp_path / "other")
    built = build_context(load_config(other.config), other.deps)
    assert built.client.on_call is None
    built.close()
    world = World(tmp_path / "main")
    drive_to_stopped(world)
    assert world.claude.on_call is None
    registry = world.registry()
    try:
        rows = registry.list_llm_calls("toy-1")
    finally:
        registry.close()
    assert len(rows) == len(world.claude.prompts) == len(world.claude.call_log())
    assert {"writer", "coordinator", "data", "feature_engineering", "modeling", "reviewer"} <= set(world.claude.kinds)
    writer_rows = [r for r in rows if r.purpose.startswith("bundle writer:")]
    assert len(writer_rows) == world.claude.kinds.count("writer") > 0
    roles = sorted({r.role for r in rows if r.role})
    assert roles == ["coordinator", "data", "feature_engineering", "modeling", "reviewer"]


def test_a_client_with_an_on_call_hook_is_refused(tmp_path):
    import pytest

    from kaggle_agent.orchestrator import ConfigError

    world = World(tmp_path)
    world.claude._on_call = lambda record: None
    with pytest.raises(ConfigError, match="on_call"):
        build_context(load_config(world.config), world.deps)


def test_models_per_role_reach_the_client(tmp_path):
    world = World(tmp_path, extra='\n[models]\ncoordinator = "model-c"\nwriter = "model-w"\n')
    world.run("start")
    assert world.claude.kinds[0] == "coordinator" and world.claude.models[0] == "model-c"
    world.lead('[plan]\ndecision = "approve"\n')
    world.run("resume")
    by_kind = {}
    for kind, model in zip(world.claude.kinds, world.claude.models):
        by_kind.setdefault(kind, set()).add(model)
    assert by_kind["writer"] == {"model-w"} and by_kind["data"] == {None} and by_kind["coordinator"] == {"model-c"}
