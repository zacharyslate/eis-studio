"""Entry point for hosted deployments (e.g. Streamlit Community Cloud: main file = app.py).

Local users run the ``eis-studio`` command instead; both execute eisstudio/app.py.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "eisstudio" / "app.py"), run_name="__main__")
