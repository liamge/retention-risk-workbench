"""Compatibility shim for legacy imports.

CLI lives in src.cli.evaluate; keep this module so existing imports keep working.
"""

from src.cli.evaluate import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    main()
