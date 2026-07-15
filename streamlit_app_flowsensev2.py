"""Streamlit entrypoint for the Flowsense V2 plotter."""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).parent / "Flowsense V2" / "plotter.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
