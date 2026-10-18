"""Builders shared by the Kaggle client tests: fake competition, files, clock, stub library."""

import io
import os
import zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import SimpleNamespace

from kaggle_agent.kaggle import FakeCompetition, FakeKaggleClient

SECRET = "tok-SECRET-0123456789abcdef"
IDS = ("a", "b", "c")
SAMPLE_CSV = "id,target\na,0\nb,0\nc,0\n"
TRAIN = b"id,x,target\na,1,0\nb,2,1\n"
TEST = b"id,x\na,3\nb,4\nc,5\n"
DAY = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)


class Clock:
    """Mutable UTC clock; the sleep advances it, so waiting never takes real time."""

    def __init__(self, now=DAY):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


def make_competition(**kwargs):
    data = {
        "files": {"train.csv": TRAIN, "test.csv": TEST, "sample_submission.csv": SAMPLE_CSV.encode()},
        "description": "Predict the target.",
        "rules": "One account per person.",
        "evaluation": "Metric: AUC.",
        "sample_submission": SAMPLE_CSV,
        "daily_limit": 2,
        **kwargs,
    }
    return FakeCompetition(**data)


def make_fake(clock=None, name="comp", earlier=None, **kwargs):
    clock = clock or Clock()
    client = FakeKaggleClient(
        {name: make_competition(**kwargs)},
        clock=clock,
        sleep=clock.sleep,
        earlier_submissions={name: earlier} if earlier else None,
    )
    return client, clock


def write_submission(path, rows=None, header="id,target"):
    rows = rows if rows is not None else [f"{i},0.5" for i in IDS]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


# -- stub in place of the official library -----------------------------------


class HttpError(Exception):
    def __init__(self, status, text="boom"):
        super().__init__(text)
        self.response = SimpleNamespace(status_code=status)


class Status(Enum):
    PENDING = 0
    COMPLETE = 1
    ERROR = 2


def zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class StubApi:
    """Mimics the used surface of ``KaggleApi`` with canned responses; no network."""

    CONFIG_NAME_TOKEN = "token"
    CONFIG_NAME_USER = "username"
    CONFIG_NAME_KEY = "key"
    events = None  # optional shared list, set by tests to record ordering

    def __init__(self):
        if self.events is not None:
            self.events.append("api_created")
        self.config_values = {}
        self.calls = []
        self.env_seen = []
        self.authenticate_calls = 0
        self.errors = {}
        self.num_today = 0
        self.num_allowed_now = 2
        self.submissions = [
            SimpleNamespace(
                ref=77, date=datetime(2026, 3, 10, 12, 1), description="msg",
                status=Status.COMPLETE, public_score="0.81",
            ),
            SimpleNamespace(
                ref=78, date=datetime(2026, 3, 10, 12, 2), description="bad",
                status=Status.ERROR, public_score=None,
            ),
        ]
        self.submit_ref = 77

    def authenticate(self):
        self.authenticate_calls += 1

    def _enter(self, name, *args, **kwargs):
        self.calls.append(name)
        self.env_seen.append(dict(os.environ))
        if name in self.errors:
            raise self.errors[name]

    def competition_download_files(self, competition, path=None, force=False, quiet=True):
        self._enter("competition_download_files")
        archive = zip_bytes({"train.csv": TRAIN, "test.csv": TEST})
        (Path(path) / f"{competition}.zip").write_bytes(archive)

    def competition_list_pages(self, competition, page_name=None):
        self._enter("competition_list_pages")
        texts = {"description": "desc", "rules": "rules", "evaluation": "eval"}
        return [SimpleNamespace(name=page_name, content=texts[page_name])]

    def competition_list_files(self, competition, page_token=None, page_size=20):
        self._enter("competition_list_files")
        return SimpleNamespace(files=[SimpleNamespace(name="train.csv"), SimpleNamespace(name="sample_submission.csv")])

    def competition_download_file(self, competition, file_name, path=None, force=False, quiet=False):
        self._enter("competition_download_file")
        (Path(path) / file_name).write_text(SAMPLE_CSV)

    def competition_get_submission_limits(self, competition_name):
        self._enter("competition_get_submission_limits")
        return SimpleNamespace(num_today=self.num_today, num_allowed_now=self.num_allowed_now)

    def competition_submissions(self, competition, **kwargs):
        self._enter("competition_submissions")
        return self.submissions

    def competition_submit(self, file_name, message, competition, quiet=False, sandbox=False):
        self._enter("competition_submit")
        return SimpleNamespace(message="ok", ref=self.submit_ref)
