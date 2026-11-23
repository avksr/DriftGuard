import os
import sys
import time
import subprocess
from datetime import datetime, timezone
import pandas as pd
from playwright.sync_api import sync_playwright
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "report_assets", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

PORT = 8530
URL = f"http://localhost:{PORT}"

def run_recapture():
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["PYTHONIOENCODING"] = "utf-8"

    python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

    print(f"Starting Streamlit server on port {PORT}...")
    server_proc = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--server.address", "127.0.0.1",
         "--browser.gatherUsageStats", "false"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    records = []

    try:
        import urllib.request
        server_ready = False
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1) as resp:
                    if resp.status == 200:
                        server_ready = True
                        break
            except Exception:
                time.sleep(1)

        if not server_ready:
            raise RuntimeError("Streamlit server failed to start within 30 seconds.")
        print("Streamlit server is ready!")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Use tall viewport (1600 x 4000) with device_scale_factor=2
            context = browser.new_context(
                viewport={"width": 1600, "height": 4000},
                device_scale_factor=2
            )
            page = context.new_page()

            def inject_overflow_style():
                page.evaluate("""() => {
                    const style = document.createElement('style');
                    style.innerHTML = `
                        section.main, .stMain, [data-testid="stAppViewContainer"], .block-container {
                            overflow: visible !important;
                            height: auto !important;
                            max-height: none !important;
                        }
                    `;
                    document.head.appendChild(style);
                }""")

            def wait_for_idle(extra_sec=4):
                page.wait_for_timeout(1500)
                try:
                    page.wait_for_selector(".stProgress", state="detached", timeout=25000)
                except Exception:
                    pass
                try:
                    page.wait_for_selector("div[data-testid='stStatusWidget']", state="detached", timeout=25000)
                except Exception:
                    pass
                page.wait_for_timeout(extra_sec * 1000)

            # ------------------------------------------------------------------
            # 1. Medium sudden price drift (Live simulation)
            # ------------------------------------------------------------------
            print("Capturing 1/5: Medium sudden price drift...")
            page.goto(URL)
            page.wait_for_selector("[data-testid='stTab']", timeout=20000)
            page.locator("[data-testid='stTab']", has_text="Live simulation").click()
            page.wait_for_timeout(1000)
            inject_overflow_style()

            # Click Run simulation
            page.locator("button:has-text('Run simulation')").click()
            wait_for_idle(extra_sec=4)
            inject_overflow_style()

            shot1 = os.path.join(SCREENSHOTS_DIR, "fig_3_6_live_medium_sudden.png")
            page.screenshot(path=shot1, full_page=True)
            im1 = Image.open(shot1)
            dt1 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Captured {shot1} - Size: {im1.size}")
            records.append({
                "figure_number": "3.6",
                "file": "screenshots/fig_3_6_live_medium_sudden.png",
                "preset_or_tab": "Live simulation / Medium sudden price drift",
                "seed": "42",
                "date_time": dt1,
                "url": URL,
                "size_px": f"{im1.size[0]} x {im1.size[1]}",
                "lowest_visible": "Interactive stream step-through slider (Batch 18 state across Fixed_1, Fixed_4, Adaptive)"
            })

            # ------------------------------------------------------------------
            # 2. Small price drift (Live simulation)
            # ------------------------------------------------------------------
            print("Capturing 2/5: Small price drift...")
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)

            # Select "Small price drift" preset
            sb = page.locator("[data-testid='stSelectbox']").filter(has_text="Preset scenario")
            sb.locator("div[role='combobox'], div[data-baseweb='select'], input").first.click()
            page.wait_for_timeout(800)
            page.locator("[role='option']", has_text="Small price drift").click()
            page.wait_for_timeout(1000)

            # Click Run simulation
            page.locator("button:has-text('Run simulation')").click()
            wait_for_idle(extra_sec=4)
            inject_overflow_style()

            shot2 = os.path.join(SCREENSHOTS_DIR, "fig_3_7_live_small_drift.png")
            page.screenshot(path=shot2, full_page=True)
            im2 = Image.open(shot2)
            dt2 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Captured {shot2} - Size: {im2.size}")
            records.append({
                "figure_number": "3.7",
                "file": "screenshots/fig_3_7_live_small_drift.png",
                "preset_or_tab": "Live simulation / Small price drift",
                "seed": "42",
                "date_time": dt2,
                "url": URL,
                "size_px": f"{im2.size[0]} x {im2.size[1]}",
                "lowest_visible": "Interactive stream step-through slider (Batch 18 state across Fixed_1, Fixed_4, Adaptive)"
            })

            # ------------------------------------------------------------------
            # 3. Investigation tab (Batch 16 medium price drift)
            # ------------------------------------------------------------------
            print("Capturing 3/5: Investigation tab...")
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("[data-testid='stTab']", has_text="Investigation").click()
            page.wait_for_timeout(1500)

            # Adjust slider from default 18 to 16
            slider = page.locator("div[role='slider']")
            if slider.count() > 0:
                slider.first.focus()
                page.keyboard.press("ArrowLeft")
                page.wait_for_timeout(500)
                page.keyboard.press("ArrowLeft")
                page.wait_for_timeout(1000)

            wait_for_idle(extra_sec=4)
            inject_overflow_style()

            shot3 = os.path.join(SCREENSHOTS_DIR, "fig_3_8_investigation.png")
            page.screenshot(path=shot3, full_page=True)
            im3 = Image.open(shot3)
            dt3 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Captured {shot3} - Size: {im3.size}")
            records.append({
                "figure_number": "3.8",
                "file": "screenshots/fig_3_8_investigation.png",
                "preset_or_tab": "Investigation / Batch 16 Medium price drift (retail_price)",
                "seed": "42",
                "date_time": dt3,
                "url": URL,
                "size_px": f"{im3.size[0]} x {im3.size[1]}",
                "lowest_visible": "Reference versus incoming batch distribution histogram (retail_price)"
            })

            # ------------------------------------------------------------------
            # 4. Decisions and incidents tab
            # ------------------------------------------------------------------
            print("Capturing 4/5: Decisions and incidents tab...")
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("[data-testid='stTab']", has_text="Decisions and incidents").click()
            wait_for_idle(extra_sec=4)
            inject_overflow_style()

            shot4 = os.path.join(SCREENSHOTS_DIR, "fig_3_9_decisions.png")
            page.screenshot(path=shot4, full_page=True)
            im4 = Image.open(shot4)
            dt4 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Captured {shot4} - Size: {im4.size}")
            records.append({
                "figure_number": "3.9",
                "file": "screenshots/fig_3_9_decisions.png",
                "preset_or_tab": "Decisions and incidents / Incident log & summaries",
                "seed": "42",
                "date_time": dt4,
                "url": URL,
                "size_px": f"{im4.size[0]} x {im4.size[1]}",
                "lowest_visible": "Export buttons (Download report Markdown, Download log CSV, Download log JSON)"
            })

            # ------------------------------------------------------------------
            # 5. Experiment results tab (Experiment B)
            # ------------------------------------------------------------------
            print("Capturing 5/5: Experiment results tab (Experiment B)...")
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("[data-testid='stTab']", has_text="Experiment results").click()
            wait_for_idle(extra_sec=4)
            inject_overflow_style()

            shot5 = os.path.join(SCREENSHOTS_DIR, "fig_3_10_experiment_results.png")
            page.screenshot(path=shot5, full_page=True)
            im5 = Image.open(shot5)
            dt5 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Captured {shot5} - Size: {im5.size}")
            records.append({
                "figure_number": "3.10",
                "file": "screenshots/fig_3_10_experiment_results.png",
                "preset_or_tab": "Experiment results / Experiment B & Manifest card",
                "seed": "N/A (30 seeds aggregate)",
                "date_time": dt5,
                "url": URL,
                "size_px": f"{im5.size[0]} x {im5.size[1]}",
                "lowest_visible": "Experiment B result plot (Detection delay & Number of checks per strategy across 30 seeds)"
            })

            browser.close()

    finally:
        print("Stopping Streamlit server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server stopped.")

    out_csv = os.path.join(PROJECT_ROOT, "report_assets", "scripts", "screenshot_records.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nAll 5 tall screenshots captured! Metadata saved to {out_csv}")
    for r in records:
        print(f"  {r['file']:46s} : {r['size_px']:15s} | Lowest visible: {r['lowest_visible']}")

if __name__ == "__main__":
    run_recapture()
