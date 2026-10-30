"""The interface the engine needs from a candidate, and nothing more.

The engine calls ``fit_predict`` and assumes nothing about the class behind
it. In the bundle contract of design section 8 a candidate is a bundle with
``fit(train)`` and ``predict(bundle_id, X)``; the bundle runner (a later
ticket, fixed code) will implement this interface by calling ``fit`` on the
training rows and then ``predict`` on the rows to score, so the engine, which
owns the labels and the metric, stays unchanged. Tests use small fakes.
"""

from typing import Protocol as Structural

import pandas as pd


class Candidate(Structural):
    """What the engine needs from a model: fit on train, predict on validation."""

    def fit_predict(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, seed: int
    ):
        """Return predictions for ``x_valid`` (one per row)."""
