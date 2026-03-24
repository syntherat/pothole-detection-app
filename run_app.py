#!/usr/bin/env python
"""
Launch the filtered pothole detection app
"""
import sys
import traceback
from pathlib import Path

try:
    root_dir = Path(__file__).resolve().parent
    app_dir = root_dir / "app"
    sys.path.insert(0, str(app_dir))
    sys.path.insert(0, str(root_dir))
    
    # Verify models exist
    model_dir = root_dir / "model"
    best_pt = model_dir / "best.pt"
    road_seg_pt = model_dir / "road_seg.pt"
    
    print(f"[INFO] Root dir: {root_dir}")
    print(f"[INFO] Looking for models in: {model_dir}")
    print(f"[INFO] best.pt exists: {best_pt.exists()}")
    print(f"[INFO] road_seg.pt exists: {road_seg_pt.exists()}")
    
    import tkinter as tk
    print("[INFO] Tkinter imported successfully")
    
    from pothole_app_filtered import PotholeAppFiltered
    print("[INFO] PotholeAppFiltered imported successfully")
    
    if __name__ == "__main__":
        root = tk.Tk()
        app = PotholeAppFiltered(root)
        root.mainloop()

except Exception as e:
    print(f"\n[ERROR] Failed to launch app: {e}\n")
    traceback.print_exc()
    sys.exit(1)
