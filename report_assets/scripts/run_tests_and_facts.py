import os
import sys
import re
import subprocess
import platform
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def run_tests_and_facts():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PROJECT_ROOT

    python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    pytest_out_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "pytest_output.txt")

    print("Running pytest tests/ -v...")
    res = subprocess.run(
        [python_exe, "-m", "pytest", "tests/", "-v"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env
    )
    
    full_output = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
    with open(pytest_out_path, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"Pytest exited with code {res.returncode}. Output saved to {pytest_out_path}")

    # Parse tests by module
    # Lines look like: tests/test_adaptive_monitor.py::test_... PASSED [  1%]
    # Or in summary:
    module_stats = {}
    pattern = re.compile(r"^(tests[/\\][^\s:]+\.py)::([^\s]+)\s+(PASSED|FAILED|SKIPPED|XFAIL)")

    for line in full_output.splitlines():
        m = pattern.match(line.strip())
        if m:
            test_file, test_name, status = m.groups()
            test_file = test_file.replace("\\", "/")
            if test_file not in module_stats:
                module_stats[test_file] = {"tests": 0, "passed": 0, "failed": 0}
            module_stats[test_file]["tests"] += 1
            if status == "PASSED":
                module_stats[test_file]["passed"] += 1
            elif status == "FAILED":
                module_stats[test_file]["failed"] += 1

    # Extract total run time from summary line, e.g. "=== 47 passed in 3.45s ==="
    time_match = re.search(r"in\s+([\d\.]+s)", full_output)
    total_time = time_match.group(1) if time_match else "unknown"

    module_records = []
    for test_file, stat in sorted(module_stats.items()):
        module_records.append({
            "test_file": test_file,
            "number_of_tests": stat["tests"],
            "passed": stat["passed"],
            "failed": stat["failed"],
            "total_run_time": total_time
        })
    
    tests_df = pd.DataFrame(module_records)
    out_tests_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "tests_by_module.csv")
    tests_df.to_csv(out_tests_path, index=False, encoding="utf-8")
    print(f"Saved {out_tests_path}")
    print(tests_df.to_string(index=False))

    # Project Facts
    def get_pkg_version(pkg_name):
        try:
            mod = __import__(pkg_name)
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return "not installed"

    import numpy
    import scipy
    import sklearn
    import streamlit
    import plotly
    import pytest

    # Git commit count
    git_count_res = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    git_commits = git_count_res.stdout.strip() if git_count_res.returncode == 0 else "unknown"

    # LOC & python files count
    def count_loc(dir_rel):
        dir_path = os.path.join(PROJECT_ROOT, dir_rel)
        if not os.path.exists(dir_path):
            return 0, 0
        py_files = 0
        total_lines = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    py_files += 1
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        total_lines += sum(1 for _ in f)
        return py_files, total_lines

    src_files, src_loc = count_loc("src")
    exp_files, exp_loc = count_loc("experiments")
    tests_files, tests_loc = count_loc("tests")

    # CPU and RAM
    cpu_model = platform.processor() or "unknown"
    # Try wmic or powershell for friendlier cpu name if available
    try:
        cpu_res = subprocess.run(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, encoding="utf-8")
        if cpu_res.returncode == 0 and cpu_res.stdout.strip():
            cpu_model = cpu_res.stdout.strip()
    except Exception:
        pass

    ram_info = "unknown"
    try:
        ram_res = subprocess.run(["powershell", "-Command", "[math]::round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)"], capture_output=True, text=True, encoding="utf-8")
        if ram_res.returncode == 0 and ram_res.stdout.strip():
            ram_info = f"{ram_res.stdout.strip()} GB"
    except Exception:
        pass

    facts = [
        {"fact": "python_version", "value": platform.python_version()},
        {"fact": "pandas_version", "value": pd.__version__},
        {"fact": "numpy_version", "value": numpy.__version__},
        {"fact": "scipy_version", "value": scipy.__version__},
        {"fact": "scikit_learn_version", "value": sklearn.__version__},
        {"fact": "streamlit_version", "value": streamlit.__version__},
        {"fact": "plotly_version", "value": plotly.__version__},
        {"fact": "pytest_version", "value": pytest.__version__},
        {"fact": "git_commits_count", "value": git_commits},
        {"fact": "src_python_files", "value": src_files},
        {"fact": "src_lines_of_code", "value": src_loc},
        {"fact": "experiments_python_files", "value": exp_files},
        {"fact": "experiments_lines_of_code", "value": exp_loc},
        {"fact": "tests_python_files", "value": tests_files},
        {"fact": "tests_lines_of_code", "value": tests_loc},
        {"fact": "cpu_model", "value": cpu_model},
        {"fact": "total_ram", "value": ram_info}
    ]

    facts_df = pd.DataFrame(facts)
    out_facts_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "project_facts.csv")
    facts_df.to_csv(out_facts_path, index=False, encoding="utf-8")
    print(f"Saved {out_facts_path}")
    print(facts_df.to_string(index=False))

if __name__ == "__main__":
    run_tests_and_facts()

