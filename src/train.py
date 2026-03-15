"""Compatibility shim for legacy imports.

CLI lives in src.cli.train; keep this module so existing imports keep working.
"""

from src.cli.train import main  # noqa: F401


if __name__ == "__main__":
    main()
