"""The schema-only summary the roles may see, built from the development part of the data."""

import pandas as pd

from kaggle_agent.agents import ColumnSummary, DataSummary
from kaggle_agent.evaluation import FrozenSplit, Protocol


def build_summary(frame: pd.DataFrame, protocol: Protocol) -> DataSummary:
    """Names, dtypes, missing fractions and distinct counts; never a row.

    Only the development rows are read (``FrozenSplit.development_data``), so the holdout is
    not even summarized.
    """
    features, target = FrozenSplit.create(frame, protocol).development_data()
    development = features.assign(**{protocol.target: target})
    rows = len(development)
    columns = tuple(
        ColumnSummary(
            name=str(name), dtype=str(development[name].dtype),
            missing_fraction=float(development[name].isna().mean()) if rows else 0.0,
            distinct_values=int(development[name].nunique(dropna=True)),
        )
        for name in development.columns
    )
    return DataSummary(rows, protocol.target, columns,
                       ("statistics of the development part only; no rows are shown",))
