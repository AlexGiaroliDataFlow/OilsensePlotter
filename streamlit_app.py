"""Streamlit Cloud entrypoint for the OilSense plotter."""

from pathlib import Path
import runpy


APP_PATH = (
    Path(__file__).parent
    / "test"
    / "Da 0% a 50% Sonda con PT100 e Pressostato"
    / "plotter.py"
)

runpy.run_path(str(APP_PATH), run_name="__main__")
