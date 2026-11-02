"""A bundle written by the scripted Claude runs in a real subprocess and scores under a frozen protocol."""

from bundle_helpers import make_runner
from evaluation_helpers import classification_frame, frozen_engine
from writer_helpers import Env, valid_step

from kaggle_agent.bundle import score_bundle
from kaggle_agent.sandbox import hash_paths
from kaggle_agent.writer import default_protected

FRAME = classification_frame(n=60)


def test_a_written_bundle_scores_through_the_real_runner_under_a_frozen_protocol(tmp_path):
    env = Env(tmp_path, [valid_step("B1")], protected=default_protected())
    sources = hash_paths(default_protected())
    folder = env.writer.write(env.request("B1"))

    engine, _ = frozen_engine()
    loaded, runner = make_runner(folder)
    outcome = score_bundle(
        engine, runner, FRAME, attempt_id="A1", execution_id="A1-E1",
        config={"bundle": "B1"}, data_hash="data",
    )
    assert outcome.error is None and outcome.execution.outcome == "succeeded"
    partitions = {m.partition for m in outcome.execution.metrics}
    assert "validation" in partitions
    assert all(m.mean is not None for m in outcome.execution.metrics)
    assert outcome.execution.code_hash == loaded.folder_hash
    engine.verify()  # the frozen protocol and the evaluation sources still match
    assert hash_paths(default_protected()) == sources
    assert (folder / "bundle.json").is_file() and not (folder / "model").exists()
