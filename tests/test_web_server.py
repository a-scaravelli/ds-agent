"""The real web server over a file registry, requested from other threads (ticket 24)."""

import hashlib
import socket
import threading
import urllib.error
import urllib.request

import pytest
from web_helpers import MAIN, db_path  # noqa: F401  (fixture)

import kaggle_agent.web.__main__ as web_main
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.web import create_app
from kaggle_agent.web.__main__ import main, make_web_server

TIMEOUT = 10


class ServedRegistry:
    """Does what production does in one thread: open the registry, build the app, serve."""

    def __init__(self, path, threaded):
        self.path, self.threaded = path, threaded
        self.ready = threading.Event()
        self.port = None
        self.error = None
        self.server = None
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        try:
            with SqliteRegistry(self.path, read_only=True) as registry:
                app = create_app(registry)
                self.server = make_web_server(app, "127.0.0.1", 0, threaded=self.threaded)
                self.port = self.server.server_port
                self.ready.set()
                self.server.serve_forever()
                self.server.server_close()
        except BaseException as exc:  # reported by __enter__
            self.error = exc
            self.ready.set()

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(TIMEOUT), "the server did not start"
        assert self.error is None, self.error
        return self

    def __exit__(self, *exc):
        if self.server is not None:
            self.server.shutdown()
        self.thread.join(TIMEOUT)
        assert not self.thread.is_alive()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"


def fetch(url, data=None):
    """(status, body) of one request; HTTP errors are returned, not raised."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=TIMEOUT) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def fetch_from_threads(urls):
    """Request every url from its own thread; results in order."""
    results = [None] * len(urls)

    def work(index, url):
        results[index] = fetch(url)

    threads = [threading.Thread(target=work, args=(i, u)) for i, u in enumerate(urls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT * 2)
    assert all(r is not None for r in results)
    return results


def _urls(server):
    return [server.url(p) for p in ("/", f"/cycles/{MAIN}", "/", f"/cycles/{MAIN}")]


# 1
def test_the_real_server_answers_requests_from_other_threads_with_200(db_path):
    with ServedRegistry(db_path, threaded=False) as server:
        results = fetch_from_threads(_urls(server))
    assert [status for status, _ in results] == [200] * 4
    assert all(MAIN in body for _, body in results)


# 2
def test_a_threaded_server_fails_with_500_which_is_the_bug_the_test_detects(db_path):
    with ServedRegistry(db_path, threaded=True) as server:
        results = fetch_from_threads(_urls(server))
    assert {status for status, _ in results} == {500}


# 3
def test_the_real_server_is_read_only_and_leaves_the_file_untouched(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    with ServedRegistry(db_path, threaded=False) as server:
        assert fetch(server.url("/"))[0] == 200
        assert fetch(server.url("/"), data=b"x=1")[0] == 405
        assert fetch(server.url(f"/cycles/{MAIN}"), data=b"x=1")[0] == 405
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


# 4
def test_main_serves_single_threaded_and_the_choice_is_documented(db_path, monkeypatch):
    seen = {}

    class Spy:
        server_port = 1

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen["closed"] = True

    def fake_make(app, host, port, threaded=True):
        seen.update(threaded=threaded, host=host, port=port)
        return Spy()

    monkeypatch.setattr(web_main, "make_web_server", fake_make)
    assert main(["--db", str(db_path)]) == 0
    assert seen == {"threaded": False, "host": "127.0.0.1", "port": 8000, "closed": True}
    assert "threaded=False" in web_main.__doc__ and "single-threaded" in web_main.__doc__


# 5
def test_a_busy_port_gives_a_clear_message_naming_port_option(db_path, capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert main(["--db", str(db_path), "--port", str(port)]) == 2
    err = capsys.readouterr().err
    assert "already in use" in err and "--port" in err and "Traceback" not in err
    # the registry was closed: the file can be opened and written again
    SqliteRegistry(db_path).close()
