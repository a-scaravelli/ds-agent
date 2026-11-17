"""Setup form, strict validation, frozen fields, protocol preview, data profile and failing jobs."""

import json
import time

from cockpit_live_helpers import FORM, live, live_factory  # noqa: F401  (fixtures)


def run_job(project, kind, timeout=60):
    started = project.ok(f"/api/jobs/{kind}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = project.api(f"/api/jobs/{started['id']}")
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def config_body(**sections):
    return {"config": {**{k: dict(v) for k, v in FORM.items()}, **sections}}


# 15
def test_the_form_shows_every_ticket_field(live):
    html = live.get("/setup")[1]
    for name in ("competition.slug", "competition.target", "competition.metric", "competition.direction",
                 "limits.rounds", "limits.candidates", "limits.stagnation", "limits.pause_every",
                 "limits.fix_retries", "protocol.validation_scheme", "protocol.repetitions", "protocol.seed",
                 "protocol.threshold_k", "models.coordinator", "models.modeling", "models.writer"):
        assert f'name="{name}"' in html, name
    assert "from_competition_info" in html and "ambiguous" in html
    assert "times the baseline noise" in html and 'id="bar-value"' in html  # threshold_k in plain words
    assert "Allowed libraries" in html and "<code>numpy</code>" in html  # display only
    assert 'id="baseline-text"' in html and "Seed plan baseline" in html
    assert 'type="password"' not in html


# 15
def test_a_valid_form_is_saved_atomically_and_keeps_the_old_copy(live):
    before = live.dir.joinpath("config.toml").read_text(encoding="utf-8")
    body = config_body(limits={"rounds": 3, "candidates": 2, "stagnation": 2, "pause_every": 1, "fix_retries": 1})
    result = live.ok("/api/config/save", body)
    assert result["ok"] is True and result["saved"] is True
    saved = live.dir.joinpath("config.toml").read_text(encoding="utf-8")
    assert "rounds = 3" in saved and "candidates = 2" in saved
    history = sorted((live.dir / "cockpit" / "config-history").iterdir())
    assert len(history) == 1 and history[0].read_text(encoding="utf-8") == before
    assert not (live.dir / "config.candidate.toml").exists()
    assert live.api("/api/config")["config"]["limits"]["rounds"] == 3
    assert "save" in live.api("/api/audit")["entries"][-1]["what"]


# 15
def test_invalid_input_shows_the_key_path_and_leaves_the_file_unchanged(live):
    before = live.dir.joinpath("config.toml").read_text(encoding="utf-8")
    bad = config_body(protocol={"validation_params": {"n_folds": 3}, "threshold_k": -2})
    result = live.ok("/api/config/save", bad)
    assert result["ok"] is False and result["saved"] is False
    assert any("threshold_k" in e["key"] or "threshold_k" in e["message"] for e in result["errors"]), result
    assert live.dir.joinpath("config.toml").read_text(encoding="utf-8") == before
    assert not (live.dir / "cockpit" / "config-history").exists()
    check = live.ok("/api/config/validate", config_body(limits={"rounds": 0}))
    assert check["ok"] is False and any("rounds" in e["key"] + e["message"] for e in check["errors"])


# 15
def test_secret_like_values_are_refused(live):
    before = live.dir.joinpath("config.toml").read_text(encoding="utf-8")
    result = live.ok("/api/config/save", config_body(competition={
        "slug": "toy", "target": "from_submission_spec", "metric": "from_competition_info",
        "api_token": "abc123"}))
    assert result["ok"] is False and "secret" in json.dumps(result["errors"]).lower()
    assert live.dir.joinpath("config.toml").read_text(encoding="utf-8") == before
    assert "abc123" not in (live.dir / "config.toml").read_text(encoding="utf-8")
    assert live.post("/api/config/save", {"config": {"nonsense": {"a": 1}}})[0] == 400


# 15
def test_the_ambiguous_metric_refusal_is_shown(live_factory):
    project = live_factory(scenario={"evaluation": "Scored with accuracy and also with logloss."})
    job = run_job(project, "preview")
    assert job["status"] == "failed" and "ambiguous" in job["result"]["error"]
    assert "competition.metric" in job["result"]["error"]


# 16
def test_frozen_fields_are_read_only_and_refused_after_the_project_started(live):
    live.ok("/api/run/start")
    live.wait_exit(10)
    html = live.get("/setup")[1]
    assert "already started" in html and "a new plan version or a new run" in html
    for name in ("competition.slug", "protocol.seed", "protocol.threshold_k", "protocol.holdout_share"):
        field = html[html.index(f'name="{name}"'):].split(">", 1)[0]
        assert "readonly" in field, name
    assert "disabled" in html[html.index('name="protocol.validation_scheme"'):].split(">", 1)[0]
    config = live.api("/api/config")
    assert config["started"] is True and "protocol" in config["frozen"]
    for change in ({"competition": {"slug": "toy", "target": "from_submission_spec", "metric": "rmse"}},
                   {"protocol": {"validation_params": {"n_folds": 4}}},
                   {"protocol": {"validation_params": {"n_folds": 3}, "seed": 99}},
                   {"protocol": {"validation_params": {"n_folds": 3}, "threshold_k": 3.0}},
                   {"protocol": {"validation_params": {"n_folds": 3}, "holdout_share": 0.3}}):
        result = live.ok("/api/config/save", config_body(**change))
        assert result["ok"] is False and result.get("frozen") is True, change
        assert "new plan version or a new run" in json.dumps(result["errors"])
    # limits are not frozen: they can still be saved
    assert live.ok("/api/config/save", config_body(limits={"rounds": 4}))["saved"] is True


# 17
def test_the_protocol_preview_job_matches_the_hash_the_start_freezes(live):
    job = run_job(live, "preview")
    assert job["status"] == "done"
    result = job["result"]
    assert result["ok"] and result["partitions"] == 3 and len(result["folds"]) == 3
    assert 0 < result["holdout_share"] < 1 and result["development_rows"] + result["holdout_rows"] == result["rows"]
    assert "times" in result["bar_text"] or "x the baseline noise" in result["bar_text"]
    assert result["already_started"] is False
    assert not (live.dir / "registry.db").exists()  # a preview freezes nothing
    live.ok("/api/run/start")
    live.wait_exit(10)
    live.decide_plan()  # the plan approval freezes the protocol
    live.ok("/api/run/resume")
    live.wait_exit(10)
    assert [c.protocol_hash for c in _cycles(live)] == [result["protocol_hash"]]
    page = live.get("/setup")[1]
    assert 'id="run-preview"' in page and 'id="preview-result"' in page


def _cycles(project):
    from kaggle_agent.registry import SqliteRegistry

    registry = SqliteRegistry(project.registry_path(), read_only=True)
    try:
        return registry.list_cycles()
    finally:
        registry.close()


# 18
def test_the_profile_uses_the_development_rows_only(live_factory):
    rows = [f"{i},{i % 7},{'' if i % 5 == 0 else i % 3},7,{1 if i % 3 == 0 else 0}" for i in range(1, 61)]
    rows += [rows[1], rows[2]]  # two exact duplicates
    text = "id,x1,x2,c,y\n" + "\n".join(rows) + "\n"
    project = live_factory(scenario={"train": text})
    preview = run_job(project, "preview")["result"]
    profile = run_job(project, "profile")
    assert profile["status"] == "done", profile
    result = profile["result"]
    assert result["part"] == "development" and result["rows"] == preview["development_rows"] < preview["rows"]
    names = [c["name"] for c in result["columns"]]
    assert "x1" in names and "id" in names
    x2 = next(c for c in result["columns"] if c["name"] == "x2")
    assert x2["missing"] > 0 and result["target"] == "y"
    assert "c" in result["constant_columns"]
    assert sum(result["target_balance"].values()) == result["rows"]
    assert result["duplicate_rows"] >= 0 and "x1" in result["distributions"]
    page = project.get("/setup")[1]
    assert "development rows only" in page


# 18
def test_the_profile_finds_id_like_columns(tmp_path):
    from cockpit_live_helpers import LiveProject

    rows = [f"{i},{i % 7},{1 if i % 3 == 0 else 0}" for i in range(1, 61)]
    project = LiveProject(tmp_path / "ids", scenario={"train": "id,x1,y\n" + "\n".join(rows) + "\n"})
    try:
        result = run_job(project, "profile")["result"]
        assert "id" in result["id_like_columns"] and "x1" not in result["id_like_columns"]
        assert result["duplicate_rows"] == 0
    finally:
        project.cleanup()


# 18, 53
def test_a_failing_job_shows_failed_and_the_page_still_renders(live_factory):
    project = live_factory(scenario={"slug": "other-competition"})  # the fake Kaggle does not know the slug
    job = run_job(project, "profile")
    assert job["status"] == "failed" and job["error"]
    assert project.get("/setup")[0] == 200
    assert project.api(f"/api/jobs/{job['id']}")["status"] == "failed"


# 53
def test_a_job_that_times_out_degrades_without_breaking_the_page(live_factory):
    project = live_factory(controller_kwargs={"job_timeout": 0.01})
    job = run_job(project, "preview")
    assert job["status"] == "timeout" and "did not finish" in job["error"]
    assert project.get("/setup")[0] == 200 and project.get("/api/jobs/nope")[0] == 404
    check = project.ok("/api/config/validate", config_body())  # the validation degrades to 'unknown'
    assert check["ok"] is False and check.get("unknown") is True
