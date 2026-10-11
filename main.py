"""Convenience entry point: ``python main.py`` == ``python -m mltest``."""
import sys

from mltest.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
