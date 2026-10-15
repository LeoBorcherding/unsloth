"""Apple Silicon leg of mps_budget_probe (staging runs mac_* jobs only on macOS arm64)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mps_budget_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
