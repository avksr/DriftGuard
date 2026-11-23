import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_audit():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PROJECT_ROOT
    
    python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    script_path = os.path.join(PROJECT_ROOT, "src", "data_audit.py")
    out_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "data_audit_output.txt")
    
    res = subprocess.run(
        [python_exe, script_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env
    )
    
    full_output = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    
    print(f"Data audit completed with exit code {res.returncode}. Saved {len(full_output)} chars to {out_path}")

if __name__ == "__main__":
    run_audit()
