# app/main.py
import os
import time
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import ImageTk

from utils import run_detection, pil_resize, ensure_dirs, set_conf_threshold, load_model, DEFAULT_MODEL_PATH, OUTPUT_DIR

APP_TITLE = "Pothole Detection App"
CANVAS_W, CANVAS_H = 640, 480

# Resolve project root (folder above /app)
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent


class PotholeApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        ensure_dirs()

        self.img_path = None
        self.tk_img = None

        # Try to load model up-front to surface missing-model errors early
        try:
            load_model()  # uses model/best.pt by default
        except FileNotFoundError as e:
            messagebox.showwarning(
                "Model not found",
                f"{e}\n\nPick your trained weight (.pt) file now."
            )
            self.browse_model()

        header = tk.Label(root, text="Upload an image and detect potholes", font=("Segoe UI", 16))
        header.pack(pady=12)

        ctrl = tk.Frame(root)
        ctrl.pack(pady=8)

        self.upload_btn = tk.Button(ctrl, text="Upload Image", width=16, command=self.upload_image)
        self.upload_btn.grid(row=0, column=0, padx=6)

        self.detect_btn = tk.Button(ctrl, text="Detect Potholes", width=16, state=tk.DISABLED, command=self.detect_potholes)
        self.detect_btn.grid(row=0, column=1, padx=6)

        self.open_out_btn = tk.Button(ctrl, text="Open Output Folder", width=18, command=self.open_output_folder)
        self.open_out_btn.grid(row=0, column=2, padx=6)

        self.pick_model_btn = tk.Button(ctrl, text="Select Model .pt", width=16, command=self.browse_model)
        self.pick_model_btn.grid(row=0, column=3, padx=6)

        conf_frame = tk.Frame(root)
        conf_frame.pack(pady=6)
        tk.Label(conf_frame, text="Confidence threshold:").grid(row=0, column=0, padx=6, sticky="e")
        self.conf_var = tk.DoubleVar(value=0.35)
        self.conf_scale = tk.Scale(conf_frame, from_=0.10, to=0.90, resolution=0.01,
                                   orient=tk.HORIZONTAL, length=300, variable=self.conf_var,
                                   command=self._on_conf_change)
        self.conf_scale.grid(row=0, column=1, padx=6, sticky="w")

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="#222")
        self.canvas.pack(pady=10)

        self.status_var = tk.StringVar(value="Ready.")
        status = tk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=12, pady=6)

    def _on_conf_change(self, _val):
        set_conf_threshold(float(self.conf_var.get()))

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO .pt model",
            filetypes=[("PyTorch Weights", "*.pt")]
        )
        if path:
            # copy to model/best.pt so future runs just work
            dest = DEFAULT_MODEL_PATH
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                # If same path, skip copy
                if Path(path).resolve() != dest.resolve():
                    import shutil
                    shutil.copy2(path, dest)
                from utils import _model  # ensure we reset cached model
                if _model is not None:
                    # reload model on next detection
                    from importlib import reload
                    import utils as utils_mod
                    reload(utils_mod)
                self.status_var.set(f"Model set: {dest}")
            except Exception as e:
                messagebox.showerror("Model load error", str(e))

    def upload_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if not path:
            return
        self.img_path = path
        self._show_on_canvas(self.img_path)
        self.detect_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Loaded: {Path(self.img_path).name}")

    def _show_on_canvas(self, image_path):
        img = pil_resize(image_path, max_size=(CANVAS_W, CANVAS_H))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.tk_img)

    def detect_potholes(self):
        if not self.img_path:
            messagebox.showerror("Error", "No image selected.")
            return
        try:
            self.detect_btn.config(state=tk.DISABLED)
            self.status_var.set("Running detection...")
            self.root.update_idletasks()

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"pred_{ts}_{Path(self.img_path).name}"
            out_path = OUTPUT_DIR / out_name

            out_path = run_detection(self.img_path, out_path)  # utils returns saved path
            self._show_on_canvas(out_path)
            self.status_var.set(f"Done. Saved to: {out_path}")
            messagebox.showinfo("Detection complete", f"Saved to:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Detection failed", str(e))
            self.status_var.set("Error. See message.")
        finally:
            self.detect_btn.config(state=tk.NORMAL)

    def open_output_folder(self):
        abs_path = (OUTPUT_DIR).resolve()
        if os.name == "nt":
            os.startfile(str(abs_path))  # Windows
        else:
            webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeApp(root)
    root.mainloop()
