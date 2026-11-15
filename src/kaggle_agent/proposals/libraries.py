"""The third-party libraries a bundle may use, and the rules for the boosting ones.

One source of truth: the writer checks, the writer prompt and the role prompts all read it.
Stdlib only, so every package (agents, writer, loop, orchestrator) can import it without a cycle.
"""

ALLOWED_LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")

# the scikit-learn-compatible classes of the boosting libraries
LIBRARY_APIS = {
    "xgboost": "XGBClassifier",
    "lightgbm": "LGBMClassifier",
    "catboost": "CatBoostClassifier",
}

# short bundle rules per library (explicit threads and seeds keep a run reproducible)
LIBRARY_RULES = {
    "xgboost": "set n_jobs and random_state",
    "lightgbm": "set n_jobs, random_state and verbosity=-1",
    "catboost": "pass allow_writing_files=False and set thread_count and random_seed",
}
