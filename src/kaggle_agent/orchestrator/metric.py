"""Infer the registered metric from the free text of the competition's evaluation page.

A small keyword table, deliberately conservative: when no registered metric is named, or more
than one is, the answer is a ``ConfigError`` asking for an explicit ``competition.metric``.
"""

import re

from kaggle_agent.evaluation import get_metric

from .errors import ConfigError

_PATTERNS = {
    "rmse": r"\brmse\b|root[ -]mean[ -](?:squared|square)[ -]error",
    "mae": r"\bmae\b|mean absolute error",
    "r2": r"\br2\b|r-squared|r squared|coefficient of determination",
    "balanced_accuracy": r"balanced accuracy",
    "accuracy": r"\baccuracy\b",
    "f1": r"\bf1\b|f-?score",
    "log_loss": r"log[ -]?loss|logarithmic loss|cross[ -]entropy",
    "roc_auc": r"\bauc\b|roc[ -]auc|area under the (?:roc|receiver)",
    "average_precision": r"average precision|\bmap@",
}


def infer_metric(evaluation_text: str) -> tuple[str, str]:
    """Return (metric, natural direction) named by the evaluation text, or raise ``ConfigError``."""
    text = evaluation_text.lower()
    found = []
    for name, pattern in _PATTERNS.items():
        if re.search(pattern, text):
            found.append(name)
            if name == "balanced_accuracy":
                text = re.sub(_PATTERNS["balanced_accuracy"], " ", text)  # not also plain accuracy
    ask = "set competition.metric (and competition.direction) explicitly"
    if not found:
        raise ConfigError(f"competition.metric: the evaluation page names no supported metric; {ask}")
    if len(found) > 1:
        raise ConfigError(f"competition.metric: the evaluation page is ambiguous "
                          f"({', '.join(found)}); {ask}")
    return found[0], get_metric(found[0]).direction
