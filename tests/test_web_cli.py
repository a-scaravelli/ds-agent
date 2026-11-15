"""Start command of the cockpit, without opening a socket (no network)."""

import ast
import hashlib
from pathlib import Path

import pytest
from web_helpers import MAIN, db_path, no_network, page  # noqa: F401  (fixtures)

import kaggle_agent.web.__main__ as web_main
from kaggle_agent.web.__main__ import main


class RunSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, app, host, port, debug):
        self.calls.append((app, host, port, debug))
        self.client = app.test_client()


def test_main_builds_the_app_and_starts_it_with_the_defaults(db_path):
    spy = RunSpy()
    assert main(["--db", str(db_path)], run=spy) == 0
    [(app, host, port, debug)] = spy.calls
    assert (host, port, debug) == ("127.0.0.1", 8000, False)
    assert app.url_map.iter_rules()
    assert app.config["PROJECT_DIR"] == db_path.resolve().parent  # --project defaults to the db folder


def test_main_passes_the_changed_values_and_the_project_folder(db_path, tmp_path):
    spy = RunSpy()
    main(["--db", str(db_path), "--port", "9000", "--host", "127.0.0.1", "--project", str(tmp_path)],
         run=spy)
    assert spy.calls[0][1:] == ("127.0.0.1", 9000, False)
    assert spy.calls[0][0].config["PROJECT_DIR"] == tmp_path


def test_main_opens_the_registry_read_only_for_every_request(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    class Probe(RunSpy):
        def __call__(self, app, host, port, debug):
            super().__call__(app, host, port, debug)
            assert MAIN in page(self.client, "/")
            assert self.client.post("/").status_code == 405
            assert self.client.get("/api/meta").get_json()["data"]["registry"]["mode"] == "read-only"

    main(["--db", str(db_path)], run=Probe())
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


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


# 25
def test_the_default_host_is_loopback_and_a_public_host_prints_a_warning(db_path, monkeypatch, capsys):
    seen = []

    class Spy:
        server_port = 1

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def fake_make(app, host, port, threaded=True):
        seen.append((host, port, threaded))
        return Spy()

    monkeypatch.setattr(web_main, "make_web_server", fake_make)
    assert main(["--db", str(db_path)]) == 0
    assert seen == [("127.0.0.1", 8000, True)]
    assert "warning" not in capsys.readouterr().err
    assert main(["--db", str(db_path), "--host", "0.0.0.0"]) == 0
    err = capsys.readouterr().err
    assert "warning" in err and "0.0.0.0" in err and "loopback" in err
    assert seen[-1][0] == "0.0.0.0"


def test_the_start_module_is_documented_as_the_reversal_of_ticket_24():
    doc = web_main.__doc__
    assert "Decision reversal of ticket 24" in doc and "threaded" in doc and "WAL" in doc
    assert "--workspace" not in Path(web_main.__file__).read_text(encoding="utf-8").split('"""')[2]
    tree = ast.parse(Path(web_main.__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Constant) and n.value == "--workspace" for n in ast.walk(tree))
