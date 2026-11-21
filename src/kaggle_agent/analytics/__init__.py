"""Performance analytics over development data only (ticket 29): pure functions, no file or registry access.

What goes in: the out-of-fold (OOF) table of an attempt (``oof_format``), the development features, the
per-partition scores and plain dicts of attempts. What comes out: JSON-ready dicts with plain-words texts and an
explicit "not available because ..." state. Nothing here reads a file, the registry, the network or the holdout:
the orchestrator records the OOF files (``orchestrator.analytics_record``) and the web reads them
(``web.analytics_data``). Modules: ``tasks`` (task and panel support), ``oof_format`` (file format),
``describe`` (describe.json validation), ``families``, ``folds`` (F1, F10), ``compare`` (F2), ``correlation``
(F12), ``history`` (G1-G3), ``classification`` and ``regression`` (F4), ``segments`` (F5), ``hardest`` (F6),
``flags`` (F11), ``importance`` (F7, F8), ``errors`` and ``words``.
"""
