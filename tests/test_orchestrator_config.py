"""Configuration (cases 3, 4, 5, 35): strict TOML, documented defaults, metric and target from the competition."""

import pytest
from orchestrator_helpers import CONFIG, World, competition, sample_csv

from kaggle_agent.budget import CycleLimits
from kaggle_agent.orchestrator import ConfigError, load_config

FULL = """\
[competition]
slug = "titanic"
target = "Survived"
metric = "accuracy"
direction = "maximize"
train_file = "t.csv"
test_file = "u.csv"

[project]
folder = "runs/titanic"
registry = "db/registry.sqlite"

[limits]
rounds = 4
candidates = 2
stagnation = 3
pause_every = 2
fix_retries = 1

[protocol]
validation_scheme = "random"
validation_params = { n_folds = 4 }
holdout_share = 0.25
seed = 7
repetitions = 2
threshold_k = 1.5

[models]
coordinator = "model-a"
writer = "model-b"

[runtime]
bundle_timeout = 90
wait_seconds = 120
poll_interval = 10

[lead]
file = "decisions.toml"
"""


def write(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    return path


# 3
def test_a_valid_config_is_accepted_with_documented_defaults(tmp_path):
    config = load_config(write(tmp_path, FULL))
    assert config.competition.slug == "titanic" and config.competition.target == "Survived"
    assert config.limits == CycleLimits(rounds=4, candidates=2, stagnation_rounds=3, pause_every=2, fix_retries=1)
    assert config.protocol.seed == 7 and config.protocol.holdout_share == 0.25
    assert config.models == {"coordinator": "model-a", "writer": "model-b"}
    assert config.paths.folder == (tmp_path / "runs/titanic").resolve()
    assert config.paths.registry == (tmp_path / "db/registry.sqlite").resolve()
    assert config.paths.workspace == config.paths.folder / "workspace"
    assert config.lead_path == (tmp_path / "decisions.toml").resolve()
    minimal = load_config(write(tmp_path, '[competition]\nslug = "x"\n[project]\nfolder = "p"\n'))
    assert minimal.competition.target == "from_submission_spec" and minimal.competition.metric == "from_competition_info"
    assert minimal.competition.train_file == "train.csv" and minimal.limits == CycleLimits()
    assert minimal.protocol.validation_params == {"n_folds": 5} and minimal.protocol.holdout_share == 0.2
    assert minimal.lead_path == minimal.paths.folder / "lead.toml" and minimal.model_for("data") is None


BASE = '[competition]\nslug = "x"\n[project]\nfolder = "p"\n'


@pytest.mark.parametrize("text, key", [
    (BASE + "[limits]\nrounds = 2\nbogus = 1\n", "limits.bogus"),
    (BASE + "[extra]\na = 1\n", "extra"),
    (BASE.replace('slug = "x"', "slug = 5"), "competition.slug"),
    ('[competition]\ntarget = "y"\n[project]\nfolder = "p"\n', "competition.slug"),
    (BASE + "[limits]\nrounds = 0\n", "limits.rounds"),
    (BASE + '[limits]\ncandidates = "3"\n', "limits.candidates"),
    (BASE + "[limits]\nrounds = true\n", "limits.rounds"),
    (BASE.replace('slug = "x"', 'slug = "x"\nmetric = "rmse"\ndirection = "maximize"'), "competition.direction"),
    (BASE.replace('slug = "x"', 'slug = "x"\nmetric = "nope"'), "competition.metric"),
    (BASE + "[protocol]\nholdout_share = 1.5\n", "protocol.holdout_share"),
    (BASE + "[protocol]\nvalidation_params = 3\n", "protocol.validation_params"),
    (BASE + "[models]\ncoordinator = 3\n", "models.coordinator"),
    (BASE + "[runtime]\nbundle_timeout = 0\n", "runtime.bundle_timeout"),
    ('[competition]\nslug = "x"\n', "[project]"),
    (BASE + "this is not toml", "not valid TOML"),
])
# 4
def test_strict_validation_names_the_key_path(tmp_path, text, key):
    with pytest.raises(ConfigError) as error:
        load_config(write(tmp_path, text))
    assert key in str(error.value)


@pytest.mark.parametrize("text, key", [
    (BASE + '[models]\ntoken = "abc"\n', "models.token"),
    (BASE + "[runtime]\nKAGGLE_KEY = 1\n", "runtime.KAGGLE_KEY"),
    (BASE + 'kaggle_username = "me"\n', "kaggle_username"),
    (BASE.replace('slug = "x"', 'slug = "x"\npassword = "p"'), "competition.password"),
    (BASE + 'ANTHROPIC_API_KEY = "k"\n', "ANTHROPIC_API_KEY"),
])
def test_a_secret_like_key_is_refused_and_points_to_the_credential_mechanisms(tmp_path, text, key):
    with pytest.raises(ConfigError) as error:
        load_config(write(tmp_path, text))
    message = str(error.value)
    assert key in message and "secret" in message
    assert "kaggle.json" in message and "subscription" in message


def test_a_missing_config_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


# 5
def test_metric_direction_and_target_come_from_competition_info_and_the_spec(tmp_path):
    world = World(tmp_path / "a")
    code, text = world.run("start")
    assert code == 10 and "target y, metric accuracy (maximize)" in text


@pytest.mark.parametrize("evaluation, word", [
    ("Submissions are scored with a custom formula.", "no supported metric"),
    ("We use accuracy and also log loss.", "ambiguous"),
])
def test_an_unknown_or_ambiguous_metric_asks_for_an_explicit_one(tmp_path, evaluation, word):
    world = World(tmp_path, comp=competition(evaluation=evaluation))
    code, text = world.run("start")
    assert code == 2 and word in text and "competition.metric" in text
    assert not (world.project / "project.json").exists()


def test_a_sample_submission_with_several_columns_needs_an_explicit_target(tmp_path):
    world = World(tmp_path, comp=competition(sample=sample_csv(columns="id,a,b")))
    code, text = world.run("start")
    assert code == 2 and "competition.target" in text


def test_the_explicit_target_must_be_a_column_of_the_train_file(tmp_path):
    world = World(tmp_path)
    world.config.write_text(CONFIG.format(extra="").replace('target = "from_submission_spec"', 'target = "zzz"'))
    code, text = world.run("start")
    assert code == 2 and "zzz" in text
