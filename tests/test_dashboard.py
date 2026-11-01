"""
test_dashboard.py - Headless automated test suite for DriftGuard Streamlit Dashboard.
Verifies all tabs, presets, and experiment views render without exceptions.
"""

from pathlib import Path
import time
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_dashboard_renders_all_tabs_and_presets():
    """Verify app.py loads all tabs and executes all four live simulation presets."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Initial render exceptions: {at.exception}"

    # Verify four presets in Live Simulation
    preset_sb_list = [sb for sb in at.selectbox if sb.label == "Preset scenario"]
    assert len(preset_sb_list) > 0, "Preset scenario selectbox not found"
    preset_sb = preset_sb_list[0]

    presets = [
        "Medium sudden price drift",
        "Large sudden price drift",
        "Small price drift",
        "Concept shift",
    ]

    preset_runtimes = {}

    for preset in presets:
        assert preset in preset_sb.options, f"Preset '{preset}' not in options {preset_sb.options}"
        preset_sb.select(preset)
        at.run()
        assert not at.exception, f"Exception selecting preset '{preset}': {at.exception}"

        # Find Run simulation button
        run_buttons = [b for b in at.button if b.label == "Run simulation"]
        assert len(run_buttons) > 0, "Run simulation button not found"

        t0 = time.perf_counter()
        run_buttons[0].click()
        at.run()
        dur = time.perf_counter() - t0
        preset_runtimes[preset] = round(dur, 2)

        assert not at.exception, f"Exception executing preset '{preset}': {at.exception}"
        print(f"Preset '{preset}' executed in {dur:.2f}s (target <= 10s)")
        assert dur <= 15.0, f"Preset '{preset}' took {dur:.2f}s, exceeding maximum acceptable limit"

    # Verify Experiment Results tab views
    exp_sb_list = [sb for sb in at.selectbox if sb.label == "Select experiment"]
    if exp_sb_list:
        exp_sb = exp_sb_list[0]
        for exp_label in ["A", "B", "C", "D"]:
            exp_sb.select(exp_label)
            at.run()
            assert not at.exception, f"Exception rendering Experiment {exp_label} in Tab 4: {at.exception}"

    print(f"\nPreset runtimes: {preset_runtimes}")
