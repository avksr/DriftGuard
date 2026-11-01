"""
test_dashboard.py - Headless test for DriftGuard Streamlit Dashboard.
Verifies all tabs and experiment views in app.py render without exceptions.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_dashboard_renders_without_exceptions():
    """Verify app.py loads all tabs without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Exceptions encountered rendering app.py: {at.exception}"

    # Verify each experiment view in Tab 4
    exp_selectboxes = [sb for sb in at.selectbox if sb.label == "Select experiment"]
    if exp_selectboxes:
        sb = exp_selectboxes[0]
        for exp_label in ["A", "B", "C", "D"]:
            if exp_label in sb.options:
                sb.select(exp_label)
                at.run()
                assert not at.exception, f"Exception rendering Experiment {exp_label} in Tab 4: {at.exception}"

