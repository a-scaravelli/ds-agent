"""The real web server over a file registry, requested from other threads (ticket 24, reversed by 26)."""

import hashlib
import socket
import urllib.error
import urllib.request

from cockpit_helpers import Served, loopback_only  # noqa: F401  (fixture)
from web_helpers import MAIN, db_path  # noqa: F401  (fixture)

from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web.__main__ import main


def test_the_real_server_answers_with_200_and_is_read_only(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    with Served(db_path) as server:
        status, headers, body = server.get("/")
        assert status == 200 and MAIN in body
        assert server.get(f"/cycles/{MAIN}")[0] == 200
        assert server.get("/", method="POST")[0] == 405
        assert server.get(f"/api/runs/{MAIN}", method="POST")[0] == 405
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_a_threaded_server_works_with_the_per_request_registry(db_path):
    # in ticket 24 a threaded server failed with 500 because one connection was shared;
    # now every request opens its own read-only connection, so it works
    with Served(db_path) as server:
        results = [server.get(p)[0] for p in ("/", f"/cycles/{MAIN}", "/api/runs", "/suggestions")]
    assert results == [200] * 4


def test_the_server_can_also_be_given_one_registry_instance_for_single_requests(db_path):
    with Served(db_path, factory=False) as server:
        assert server.get("/")[0] == 200


def test_a_busy_port_gives_a_clear_message_naming_port_option(db_path, capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert main(["--db", str(db_path), "--port", str(port)]) == 2
    err = capsys.readouterr().err
    assert "already in use" in err and "--port" in err and "Traceback" not in err
    # nothing is left open: the file can be opened and written again
    SqliteRegistry(db_path).close()
