"""Environment health cards: fake claude scripts, credentials by presence only, libraries and disk."""

import builtins
import time

from cockpit_live_helpers import fake_claude, live, live_factory  # noqa: F401  (fixtures)

from kaggle_agent.web.environment import KAGGLE_STEPS, kaggle_card


def card_of(project, name, wait=10):
    """The environment cards after the background claude check finished (the page itself never waits)."""
    project.api("/api/environment")  # starts the check
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        cards = project.api("/api/environment")
        if cards["claude"]["state"] != "checking":
            return cards[name]
        time.sleep(0.05)
    raise AssertionError("the claude card stayed at checking")


# 12
def test_environment_cards_degrade_to_unknown_and_never_block_rendering(live_factory):
    expected = {"in": "logged_in", "out": "logged_out", "garbage": "unknown", "hang": "unknown"}
    for mode, state in expected.items():
        project = live_factory(claude=mode, controller_kwargs={"claude_timeout": 1.0})
        # the page must answer at once, whatever the script does
        started = time.monotonic()
        status, html = project.get("/environment")
        assert status == 200 and time.monotonic() - started < 2
        assert "checking" in html and 'id="card-claude"' in html
        card = card_of(project, "claude")
        assert card["state"] == state, (mode, card)
        if state != "logged_in":
            assert card["fix"] and "claude auth login" in card["fix"] and "normal Terminal" in card["fix"]
        else:
            assert card["fix"] is None
        project.cleanup()


# 12
def test_a_missing_claude_executable_is_unknown(live_factory, tmp_path):
    project = live_factory()
    project.controller.claude_executable = str(tmp_path / "no-such-claude")
    assert card_of(project, "claude")["state"] == "unknown"


# 13
def test_kaggle_card_reports_presence_only_in_lookup_order_and_never_opens_a_secret(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    secret = "s3cret-value-that-must-not-appear"
    # missing: exact steps in words, no token anywhere
    card = kaggle_card({}, home)
    assert card["state"] == "missing" and card["steps"] == list(KAGGLE_STEPS) and card["source"] is None
    # legacy json file only
    legacy = home / ".kaggle" / "kaggle.json"
    legacy.write_text(secret, encoding="utf-8")
    legacy.chmod(0o644)
    card = kaggle_card({}, home)
    assert card["state"] == "present" and "kaggle.json" in card["source"]
    assert card["warnings"] and "chmod 600" in card["warnings"][0]
    # user name and key beat the legacy file
    card = kaggle_card({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": secret}, home)
    assert "KAGGLE_USERNAME" in card["source"]
    # the token file beats both
    token = home / ".kaggle" / "access_token"
    token.write_text(secret, encoding="utf-8")
    token.chmod(0o600)
    card = kaggle_card({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": secret}, home)
    assert "access_token" in card["source"] and card["warnings"] == []
    # the environment variable beats everything
    card = kaggle_card({"KAGGLE_API_TOKEN": secret}, home)
    assert "KAGGLE_API_TOKEN" in card["source"]
    assert secret not in repr(card)
    assert not [path for path in opened if ".kaggle" in path]  # stat only: no secret file was ever opened


# 13
def test_the_environment_page_has_no_token_input_and_never_returns_a_secret(live, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "tok-do-not-leak-1234567890")
    for url in ("/environment", "/setup", "/api/environment"):
        status, body = live.get(url)
        text = body if isinstance(body, str) else repr(body)
        assert status == 200 and "tok-do-not-leak" not in text, url
    html = live.get("/environment")[1] + live.get("/setup")[1]
    assert 'type="password"' not in html and 'name="token"' not in html and "KAGGLE_KEY</" not in html
    assert "never asks for a token" in html


# 14
def test_library_openmp_python_and_disk_cards(live):
    cards = live.api("/api/environment")
    versions = cards["libraries"]["versions"]
    assert set(versions) == {"numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "flask"}
    assert all(isinstance(v, str) and v for v in versions.values())
    assert versions["numpy"] != "unknown" and versions["flask"] != "unknown"
    assert cards["openmp"]["state"] in ("present", "missing", "unknown")
    assert cards["python"]["version"].count(".") == 2
    assert cards["disk"]["state"] == "ok" and cards["disk"]["free_gb"] >= 0


# 14
def test_a_failing_card_is_unknown(monkeypatch):
    import ctypes.util

    from kaggle_agent.web import environment

    def boom(name):
        raise OSError("no lookup")

    monkeypatch.setattr(ctypes.util, "find_library", boom)
    assert environment.openmp_card()["state"] == "unknown"
    monkeypatch.setattr(environment.shutil, "disk_usage", boom)
    assert environment.disk_card("anywhere")["state"] == "unknown"
    monkeypatch.setattr(environment.metadata, "version", boom)
    assert set(environment.libraries_card()["versions"].values()) == {"unknown"}
