"""Direct-script launcher for the RescueBench teleport capture probe."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


def _load_cli_main() -> Callable[[Sequence[str] | None], int]:
    repository_root = Path(__file__).resolve().parent.parent
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    module = importlib.import_module("benchmark.teleport_probe_cli")
    return module.main


def main(argv: Sequence[str] | None = None) -> int:
    """Load the package implementation and run its command-line interface."""

    return _load_cli_main()(argv)


if __name__ == "__main__":
    raise SystemExit(main())
