import os
import shutil
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def copy_and_check():
    src_dir = os.path.join(PROJECT_ROOT, "results")
    dst_dir = os.path.join(PROJECT_ROOT, "report_assets", "figures")
    
    mapping = {
        "experiment_a.png": "fig_3_1_experiment_a.png",
        "experiment_b.png": "fig_3_2_experiment_b.png",
        "experiment_c.png": "fig_3_4_experiment_c.png",
        "experiment_d.png": "fig_3_5_experiment_d.png",
    }
    
    for src_name, dst_name in mapping.items():
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(dst_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        img = Image.open(dst_path)
        dpi = img.info.get("dpi", (72, 72))
        width_in = img.width / dpi[0] if dpi[0] else img.width / 100
        height_in = img.height / dpi[1] if dpi[1] else img.height / 100
        print(f"Copied {src_name} -> {dst_name}: size={img.size}, dpi={dpi}, calc_inches=({width_in:.2f} x {height_in:.2f})")

if __name__ == "__main__":
    copy_and_check()

