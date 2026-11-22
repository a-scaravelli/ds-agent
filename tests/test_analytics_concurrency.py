"""Readers stay safe while the orchestrator writes (case 11)."""

import json
import threading
from pathlib import Path

import pytest
from analytics_helpers import oof_bytes, oof_frame

from kaggle_agent.orchestrator.analytics_record import AnalyticsWriter
from kaggle_agent.web import store

BUNDLES = 40
READERS = 16


# 11
def test_sixteen_readers_never_see_a_partial_file_or_a_tmp_file_while_a_writer_records(tmp_path, monkeypatch):
    project = tmp_path.resolve() / "project"
    project.mkdir()
    writer = AnalyticsWriter(project)
    opened: list[str] = []
    real_read = store._read_capped

    def spy(path, cap):
        opened.append(Path(path).name)
        return real_read(path, cap)

    monkeypatch.setattr(store, "_read_capped", spy)
    done = threading.Event()
    problems: list[str] = []
    seen_ok = [0] * READERS

    def write():
        try:
            for i in range(BUNDLES):
                blob = oof_bytes(oof_frame([0, 1] * (20 + i), prediction=[0, 1] * (20 + i)))
                writer.put("oof", f"B{i}", ".csv.gz", blob, bundle_id=f"B{i}", rows=1, partitions=1,
                           fold_scores={}, classes=[0, 1], input_kind="labels")
        finally:
            done.set()

    def read(slot):
        try:
            while not done.is_set() or seen_ok[slot] == 0:
                latest, _ = store.analytics_index(project)
                for key, line in list(latest.items()):
                    if key[0] != "oof":
                        continue
                    blob, reason = store.read_analytics_file(project, line)
                    if blob is None:
                        if "bad" not in str(key[1]):
                            problems.append(f"{key}: {reason}")
                    else:
                        seen_ok[slot] += 1
        except BaseException as error:  # a reader must never raise
            problems.append(repr(error))

    # a torn last line and a file whose bytes do not match the hash of its index line
    (project / "analytics").mkdir(exist_ok=True)
    bad_name = "oof/Bbad-aaaaaaaaaaaa.csv.gz"
    (project / "analytics" / "oof").mkdir(exist_ok=True)
    (project / "analytics" / bad_name).write_bytes(b"different bytes")
    writer.line(kind="oof", bundle_id="Bbad", file=bad_name, sha256="a" * 64, bytes=15, status="ok")
    threads = [threading.Thread(target=read, args=(i,)) for i in range(READERS)] + [threading.Thread(target=write)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    with open(project / "analytics" / "index.jsonl", "ab") as handle:
        handle.write(b'{"v":1,"kind":"oof","bundle_id":"Bx","sha2')
    assert not problems, problems[:3]
    assert all(n > 0 for n in seen_ok)
    latest, skipped = store.analytics_index(project)
    assert skipped == 1 and len([k for k in latest if k[0] == "oof"]) == BUNDLES + 1
    blob, reason = store.read_analytics_file(project, latest[("oof", "Bbad")])
    assert blob is None and "does not match" in reason
    assert opened and not [n for n in opened if ".tmp" in n or n.startswith(".")]


def test_a_reader_only_opens_a_file_that_an_index_line_names_and_never_creates_the_folder(tmp_path):
    project = tmp_path.resolve() / "p"
    project.mkdir()
    assert store.analytics_index(project) == ({}, 0)
    assert not (project / "analytics").exists()
    stray = project / "analytics" / "oof"
    stray.mkdir(parents=True)
    (stray / "B1-abcdefabcdef.csv.gz").write_bytes(b"x")
    line = {"v": 1, "kind": "oof", "bundle_id": "B1", "file": "oof/../../etc/passwd", "sha256": "a" * 64}
    assert store.read_analytics_file(project, line)[0] is None
    for bad in ("../x", "B 1", "", "a/b"):
        with pytest.raises(ValueError):
            store.analytics_path(project, "oof", bad, "abcdefabcdef")
    with pytest.raises(ValueError):
        store.analytics_path(project, "oof", "B1", "../../etc")
