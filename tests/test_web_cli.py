"""Start command of the web interface, without opening a socket (no network)."""

import hashlib

import pytest
from web_helpers import MAIN, db_path, no_network, page  # noqa: F401  (fixtures)

from kaggle_agent.web.__main__ import main


class RunSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, app, host, port, debug):
        self.calls.append((app, host, port, debug))
        self.client = app.test_client()


# 37
def test_main_builds_the_app_and_starts_it_with_the_defaults(db_path):
    spy = RunSpy()
    assert main(["--db", str(db_path)], run=spy) == 0
    [(app, host, port, debug)] = spy.calls
    assert (host, port, debug) == ("127.0.0.1", 8000, False)
    assert app.url_map.iter_rules()


def test_main_passes_the_changed_values(db_path):
    spy = RunSpy()
    main(["--db", str(db_path), "--port", "9000", "--host", "127.0.0.1"], run=spy)
    assert spy.calls[0][1:] == ("127.0.0.1", 9000, False)


def test_main_opens_the_registry_read_only(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    class Probe(RunSpy):
        def __call__(self, app, host, port, debug):
            super().__call__(app, host, port, debug)
            assert MAIN in page(self.client, "/")
            assert self.client.post("/").status_code == 405

    main(["--db", str(db_path)], run=Probe())
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


# 38
def test_missing_file_gives_exit_code_2_and_a_message(tmp_path, capsys):
    missing = tmp_path / "nope.db"
    spy = RunSpy()
    assert main(["--db", str(missing)], run=spy) == 2
    err = capsys.readouterr().err
    assert "does not exist" in err and "Traceback" not in err
    assert not missing.exists() and not spy.calls


def test_invalid_file_gives_exit_code_2(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a database", encoding="utf-8")
    assert main(["--db", str(bad)], run=RunSpy()) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_db_option_is_required():
    with pytest.raises(SystemExit) as exc:
        main([], run=RunSpy())
    assert exc.value.code == 2
