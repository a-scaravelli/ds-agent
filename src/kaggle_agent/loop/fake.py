"""A fake bundle writer for tests: a valid minimal bundle, no model code of any value."""

import hashlib
import json
from pathlib import Path

from .writer import BundleRequest, BundleWriter

_MODEL = '''\
def fit(train):
    pass


def predict(bundle_id, X):
    return [0] * len(X)
'''


class FakeBundleWriter(BundleWriter):
    """Writes ``bundle.json`` and ``model.py``; every call is recorded in ``requests``.

    The folder content depends on the hypothesis and the bundle id, so different candidates give
    different folders. With ``duplicate=True`` the content depends on neither, so two bundles are
    identical. With ``invalid_for`` (a set of bundle ids or a predicate on the request) the
    folder breaks the contract (no ``predict`` entry point).
    """

    def __init__(self, duplicate: bool = False, invalid_for=None) -> None:
        self.duplicate = duplicate
        self.invalid_for = invalid_for
        self.requests: list[BundleRequest] = []

    def _invalid(self, request: BundleRequest) -> bool:
        rule = self.invalid_for
        if rule is None:
            return False
        if callable(rule):
            return bool(rule(request))
        return request.bundle_id in rule

    def write(self, request: BundleRequest) -> Path:
        self.requests.append(request)
        folder = Path(request.folder)
        folder.mkdir(parents=True, exist_ok=True)
        entry_points = {"fit": "model:fit", "predict": "model:predict"}
        if self._invalid(request):
            entry_points = {"fit": "model:fit", "predict": "missing:predict"}
        manifest = {
            "bundle_id": request.bundle_id,
            "dependencies": [],
            "entry_points": entry_points,
        }
        (folder / "bundle.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        seed = "same" if self.duplicate else f"{request.attempt.hypothesis}|{request.bundle_id}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        (folder / "model.py").write_text(f"# fake bundle {digest}\n{_MODEL}", encoding="utf-8")
        return folder
