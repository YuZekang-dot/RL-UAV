"""Optional KaiwuDRL compatibility package.

The platform-independent implementation is installed from ``src/rl_uav``.
For competition zip execution without an editable install, this package adds
the repository's ``src`` directory to the import path only when imported.
"""

from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

