"""Allow ``python -m mtmbot``."""

from __future__ import annotations

import sys

from mtmbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
