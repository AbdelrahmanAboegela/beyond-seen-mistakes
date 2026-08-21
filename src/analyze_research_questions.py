"""Compatibility entry point for the final matched-v2 research questions."""
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from .compose_final_results import main
except ImportError:
    from compose_final_results import main


if __name__ == "__main__":
    main()
