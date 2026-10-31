"""Read-only web interface over the registry (design section 15, step 10, level 1)."""

from .app import create_app

__all__ = ["create_app"]
