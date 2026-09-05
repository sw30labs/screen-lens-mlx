"""Allow ``python -m src.web`` to start the command deck."""
from __future__ import annotations

from src.web.server import main

if __name__ == "__main__":
    main()
