# app/main_enhanced.py
"""
Enhanced Pothole Detection Application
Features: Batch processing, Export, Preprocessing, Monitoring, Keyboard shortcuts
"""
import os
import time
import webbrowser
import tkinter as tk
import threading
import queue
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import ImageTk, Image
import logging
import cv2
import numpy as np

from utils import run_detection, pil_resize, ensure_dirs, set_conf_threshold, load_model, DEFAULT_MODEL_PATH, OUTPUT_DIR, _conf
from enhanced_utils import (
    batch_process_images, preprocess_image, PerformanceMonitor,
    export_detections_csv, export_detections_json
)
from two_stage_detection import create_two_stage_detector

logger = logging.getLogger(__name__)

APP_TITLE = "PAVE"
CANVAS_W, CANVAS_H = 720, 480
ROAD_MASK_STRIDE = 10
ROAD_MASK_WIDTH = 800

# Color scheme
BG_COLOR = "#f0f0f0"
PRIMARY_COLOR = "#0078d4"
SUCCESS_COLOR = "#107c10"
ERROR_COLOR = "#d13438"
TEXT_COLOR = "#323130"
LABEL_COLOR = "#605e5c"

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent


class PotholeAppEnhanced:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x950")
        self.root.resizable(True, True)
        self.root.configure(bg=BG_COLOR)
        ensure_dirs()

        self.img_path = None
        self.vid_path = None
        self.tk_img = None
        self.detection_history = []
        self.performance_monitor = PerformanceMonitor()
        self.mode = tk.StringVar(value="image")  # "image" or "video"
        
        # Preprocessing options
        self.preprocess_var = tk.BooleanVar(value=False)
        self.enhance_var = tk.BooleanVar(value=True)
        self.denoise_var = tk.BooleanVar(value=True)

        # Two-stage detection (road segmentation + pothole detection)
        self.use_road_seg_var = tk.BooleanVar(value=True)
        self.show_road_mask_var = tk.BooleanVar(value=True)
        self.two_stage_detector = None
        self.video_queue = queue.Queue()
        self.video_thread = None

        # Try to load model and initialize two-stage detector
        try:
            load_model()
            # Initialize two-stage detector with road segmentation
            road_model_path = ROOT / "model" / "road_seg.pt"
            pothole_model_path = ROOT / "model" / "best.pt"
            if not pothole_model_path.exists():
                pothole_model_path = DEFAULT_MODEL_PATH
            
            self.two_stage_detector = create_two_stage_detector(
                str(pothole_model_path),
                str(road_model_path) if road_model_path.exists() else None
            )
            
            if self.two_stage_detector.use_road_seg:
                logger.info("Two-stage detection enabled with road segmentation")
            else:
                logger.info("Road segmentation model not found, using single-stage detection")
                
        except FileNotFoundError as e:
            messagebox.showwarning(
                "Model not found",
                f"{e}\n\nPick your trained weight (.pt) file now."
            )
            self.browse_model()

        self._create_ui()
        self._bind_keyboard_shortcuts()

    def _create_ui(self):
        """Create the entire UI"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=PRIMARY_COLOR, height=70)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        header_title = tk.Label(
            header_frame, 
            text="Road Pothole Detection System - Enhanced",
            font=("Segoe UI", 18, "bold"),
            bg=PRIMARY_COLOR,
            fg="white"
        )
        header_title.pack(pady=18)
        
        # Main container
        main_container = tk.Frame(self.root, bg=BG_COLOR)
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg=BG_COLOR, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Mode selector
        mode_frame = tk.LabelFrame(
            left_panel,
            text="Mode",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        mode_frame.pack(fill="x", pady=(0, 10))
        
        tk.Radiobutton(
            mode_frame,
            text="Image",
            variable=self.mode,
            value="image",
            bg=BG_COLOR,
            font=("Segoe UI", 9),
            command=self._update_upload_button
        ).pack(anchor="w")
        
        tk.Radiobutton(
            mode_frame,
            text="Video",
            variable=self.mode,
            value="video",
            bg=BG_COLOR,
            font=("Segoe UI", 9),
            command=self._update_upload_button
        ).pack(anchor="w")
        
        # Action buttons
        action_frame = tk.LabelFrame(
            left_panel,
            text="Detection",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        action_frame.pack(fill="x", pady=(0, 10))
        
        self.upload_btn = tk.Button(
            action_frame,
            text="Choose Image  (Ctrl+O)",
            command=self.upload_file,
            font=("Segoe UI", 9, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            activebackground="#005a9e",
            relief="flat",
            padx=10,
            pady=8,
            cursor="hand2"
        )
        self.upload_btn.pack(fill="x", pady=3)
        
        self.detect_btn = tk.Button(
            action_frame,
            text="Run Detection  (Ctrl+D)",
            command=self.detect_potholes,
            font=("Segoe UI", 9, "bold"),
            bg=SUCCESS_COLOR,
            fg="white",
            activebackground="#0a6f0a",
            state=tk.DISABLED,
            relief="flat",
            padx=10,
            pady=8,
            cursor="hand2"
        )
        self.detect_btn.pack(fill="x", pady=3)
        
        self.batch_btn = tk.Button(
            action_frame,
            text="Batch Process  (Ctrl+B)",
            command=self.batch_process,
            font=("Segoe UI", 9, "bold"),
            bg="#6264a7",
            fg="white",
            activebackground="#464775",
            relief="flat",
            padx=10,
            pady=8,
            cursor="hand2"
        )
        self.batch_btn.pack(fill="x", pady=3)
        
        # Preprocessing frame
        preproc_frame = tk.LabelFrame(
            left_panel,
            text="Image Preprocessing",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        preproc_frame.pack(fill="x", pady=(0, 10))
        
        tk.Checkbutton(
            preproc_frame,
            text="Enable Preprocessing",
            variable=self.preprocess_var,
            bg=BG_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=2)
        
        tk.Checkbutton(
            preproc_frame,
            text="Enhance (brightness/contrast)",
            variable=self.enhance_var,
            bg=BG_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=2)
        
        tk.Checkbutton(
            preproc_frame,
            text="Denoise",
            variable=self.denoise_var,
            bg=BG_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=2)
        
        # Settings frame
        settings_frame = tk.LabelFrame(
            left_panel,
            text="Detection Settings",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        settings_frame.pack(fill="x", pady=(0, 10))
        
        conf_label = tk.Label(
            settings_frame,
            text="Confidence Threshold",
            font=("Segoe UI", 9, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        conf_label.pack(anchor="w", pady=(0, 5))
        
        self.conf_var = tk.DoubleVar(value=0.35)
        self.conf_value_label = tk.Label(
            settings_frame,
            text="0.35",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=PRIMARY_COLOR
        )
        self.conf_value_label.pack(anchor="e", pady=(0, 2))
        
        self.conf_scale = tk.Scale(
            settings_frame,
            from_=0.10,
            to=0.90,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            command=self._on_conf_change,
            bg=BG_COLOR,
            fg=PRIMARY_COLOR,
            highlightthickness=0,
            troughcolor="#e0e0e0",
            showvalue=0
        )
        self.conf_scale.pack(fill="x", pady=5)
        
        conf_info = tk.Label(
            settings_frame,
            text="Lower = more detections\nHigher = fewer but confident",
            font=("Segoe UI", 8),
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            justify="left"
        )
        conf_info.pack(anchor="w", pady=5)

        # Two-stage detection section
        road_seg_label = tk.Label(
            settings_frame,
            text="Road Segmentation",
            font=("Segoe UI", 9, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        road_seg_label.pack(anchor="w", pady=(8, 0))

        tk.Checkbutton(
            settings_frame,
            text="Enable road segmentation (filters false positives)",
            variable=self.use_road_seg_var,
            command=self._toggle_road_seg,
            bg=BG_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(2, 2))

        tk.Checkbutton(
            settings_frame,
            text="Show road mask overlay",
            variable=self.show_road_mask_var,
            command=self._update_preview,
            bg=BG_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(0, 6))

        # Status indicator
        if self.two_stage_detector and self.two_stage_detector.use_road_seg:
            status_text = "✓ Road segmentation model loaded"
            status_color = SUCCESS_COLOR
        else:
            status_text = "⚠ Road segmentation model not found (using single-stage)"
            status_color = "#d13438"
        
        status_label = tk.Label(
            settings_frame,
            text=status_text,
            bg=BG_COLOR,
            fg=status_color,
            font=("Segoe UI", 8, "italic")
        )
        status_label.pack(anchor="w", pady=(0, 6))
        
        # Export frame
        export_frame = tk.LabelFrame(
            left_panel,
            text="Export Results",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        export_frame.pack(fill="x", pady=(0, 10))
        
        tk.Button(
            export_frame,
            text="Export to CSV  (Ctrl+E)",
            command=self.export_csv,
            font=("Segoe UI", 9),
            bg="#217346",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2"
        ).pack(fill="x", pady=3)
        
        tk.Button(
            export_frame,
            text="Export to JSON",
            command=self.export_json,
            font=("Segoe UI", 9),
            bg="#217346",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2"
        ).pack(fill="x", pady=3)
        
        # Model & Output frame
        model_frame = tk.LabelFrame(
            left_panel,
            text="Model & Output",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        model_frame.pack(fill="x")
        
        tk.Button(
            model_frame,
            text="Load Model",
            command=self.browse_model,
            font=("Segoe UI", 9),
            bg="#555555",
            fg="white",
            relief="flat",
            padx=10,
            pady=7,
            cursor="hand2"
        ).pack(fill="x", pady=3)
        
        tk.Button(
            model_frame,
            text="Open Output Folder",
            command=self.open_output_folder,
            font=("Segoe UI", 9),
            bg="#6c757d",
            fg="white",
            relief="flat",
            padx=10,
            pady=7,
            cursor="hand2"
        ).pack(fill="x", pady=3)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg=BG_COLOR)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Canvas frame
        canvas_frame = tk.LabelFrame(
            right_panel,
            text="Image Preview",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=5,
            pady=5
        )
        canvas_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#1e1e1e",
            highlightthickness=1,
            highlightbackground="#cccccc"
        )
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(
            right_panel,
            text="Detection Results",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        stats_frame.pack(fill="x", pady=(0, 10))
        
        # Stats grid
        stats_grid = tk.Frame(stats_frame, bg=BG_COLOR)
        stats_grid.pack(fill="x")
        
        self.potholes_label = tk.Label(
            stats_grid,
            text="Potholes: --",
            font=("Segoe UI", 10, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        self.potholes_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        self.confidence_label = tk.Label(
            stats_grid,
            text="Confidence: --",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        self.confidence_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        self.time_label = tk.Label(
            stats_grid,
            text="Time: --",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        self.time_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        self.fps_label = tk.Label(
            stats_grid,
            text="Avg FPS: --",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        self.fps_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Performance monitor frame
        perf_frame = tk.LabelFrame(
            right_panel,
            text="Performance Monitor",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            padx=10,
            pady=10
        )
        perf_frame.pack(fill="x")
        
        self.perf_label = tk.Label(
            perf_frame,
            text="No detections yet",
            font=("Segoe UI", 9),
            bg=BG_COLOR,
            fg=LABEL_COLOR,
            justify="left"
        )
        self.perf_label.pack(anchor="w")
        
        tk.Button(
            perf_frame,
            text="Reset Monitor",
            command=self.reset_monitor,
            font=("Segoe UI", 8),
            bg="#999999",
            fg="white",
            relief="flat",
            padx=8,
            pady=4,
            cursor="hand2"
        ).pack(anchor="e", pady=(5, 0))
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#f0f0f0", height=40)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        separator = tk.Frame(status_frame, bg="#d0d0d0", height=1)
        separator.pack(fill="x", side="top")
        
        self.status_var = tk.StringVar(value="Ready | Use Ctrl+O to open, Ctrl+D to detect, Ctrl+B for batch")
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg="#f0f0f0",
            fg=LABEL_COLOR,
            anchor="w",
            padx=15
        ).pack(fill="both", expand=True)

    def _bind_keyboard_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.upload_image())
        self.root.bind('<Control-O>', lambda e: self.upload_image())
        self.root.bind('<Control-d>', lambda e: self.detect_potholes())
        self.root.bind('<Control-D>', lambda e: self.detect_potholes())
        self.root.bind('<Control-b>', lambda e: self.batch_process())
        self.root.bind('<Control-B>', lambda e: self.batch_process())
        self.root.bind('<Control-e>', lambda e: self.export_csv())
        self.root.bind('<Control-E>', lambda e: self.export_csv())
        self.root.bind('<Control-s>', lambda e: self.save_current())
        self.root.bind('<Control-S>', lambda e: self.save_current())

    def _on_conf_change(self, val):
        """Update confidence threshold"""
        conf_val = float(self.conf_var.get())
        self.conf_value_label.config(text=f"{conf_val:.2f}")
        set_conf_threshold(conf_val)

    def _update_upload_button(self):
        """Update button text and state based on current mode"""
        if self.mode.get() == "image":
            self.upload_btn.config(text="Choose Image  (Ctrl+O)")
        else:
            self.upload_btn.config(text="Choose Video  (Ctrl+O)")
        self.detect_btn.config(state=tk.DISABLED)
        self.img_path = None
        self.vid_path = None

    def _toggle_road_seg(self):
        """Toggle road segmentation on/off"""
        if self.two_stage_detector:
            # This doesn't actually change the detector, just a flag
            # Real toggling happens during detection
            self._update_preview()
    
    def _update_preview(self):
        """Update preview when settings change"""
        if self.mode.get() == "image" and self.img_path:
            try:
                img = pil_resize(self.img_path, max_size=(CANVAS_W, CANVAS_H))
                self._display_image_with_detection_preview(img)
            except Exception as e:
                logger.error(f"Error updating preview: {e}")
        elif self.mode.get() == "video" and self.vid_path:
            try:
                cap = cv2.VideoCapture(str(self.vid_path))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    pil_img = pil_resize(str(self.vid_path), max_size=(CANVAS_W, CANVAS_H))
                    self._display_image_with_detection_preview(pil_img)
            except Exception as e:
                logger.error(f"Error updating video preview: {e}")

    def browse_model(self):
        """Load a different model"""
        path = filedialog.askopenfilename(
            title="Select YOLO .pt model",
            filetypes=[("PyTorch Weights", "*.pt")]
        )
        if path:
            dest = DEFAULT_MODEL_PATH
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if Path(path).resolve() != dest.resolve():
                    import shutil
                    shutil.copy2(path, dest)
                    logger.info(f"Model copied to {dest}")
                
                import utils as utils_mod
                utils_mod._model = None
                
                load_model()
                self.status_var.set(f"Model loaded: {dest}")
                logger.info(f"Model loaded: {dest}")
            except Exception as e:
                messagebox.showerror("Error", f"Model load error: {e}")
                self.status_var.set("Error loading model")

    def upload_file(self):
        if self.mode.get() == "image":
            self.upload_image()
        else:
            self.upload_video()

    def upload_image(self):
        """Upload a single image"""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if not path:
            return
        
        self.img_path = path
        self._show_on_canvas(self.img_path)
        self.detect_btn.config(state=tk.NORMAL)
        
        # Reset stats
        self.potholes_label.config(text="Potholes: --", fg=TEXT_COLOR)
        self.confidence_label.config(text="Confidence: --")
        self.time_label.config(text="Time: --")
        
        self.status_var.set(f"Loaded: {Path(self.img_path).name}")

    def upload_video(self):
        """Upload a single video"""
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
        path = filedialog.askopenfilename(title="Select a video", filetypes=filetypes)
        if not path:
            return
        
        self.vid_path = path
        self.detect_btn.config(state=tk.NORMAL)
        
        # Reset stats
        self.potholes_label.config(text="Potholes: --", fg=TEXT_COLOR)
        self.confidence_label.config(text="Confidence: --")
        self.time_label.config(text="Time: --")
        
        # Show first frame
        frame = self._get_first_video_frame(path)
        if frame is not None:
            pil_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
            self._display_image_with_detection_preview(pil_frame)

        # Show video info
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.status_var.set(f"Loaded: {Path(self.vid_path).name} ({frame_count} frames @ {fps:.0f} fps)")
        else:
            messagebox.showerror("Error", "Failed to open video file")
            self.vid_path = None

    def _show_on_canvas(self, image_path):
        """Display image on canvas"""
        try:
            img = pil_resize(image_path, max_size=(CANVAS_W, CANVAS_H))
            self._display_image_with_detection_preview(img)
        except Exception as e:
            logger.error(f"Error displaying image: {e}")

    def _get_first_video_frame(self, video_path):
        """Get the first frame from a video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        max_size = max(h, w)
        if max_size > CANVAS_W:
            scale = CANVAS_W / max_size
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def _display_image_with_detection_preview(self, pil_img):
        """Display PIL image (simple display, no detection overlay for preview)"""
        display_img = pil_img.copy()
        self.tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.tk_img)

    def detect_potholes(self):
        """Run detection on current file (image or video)"""
        if self.mode.get() == "image":
            self._detect_image()
        else:
            self._detect_video()

    def _detect_image(self):
        """Run detection on current image using two-stage approach"""
        if not self.img_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.detect_btn.config(state=tk.DISABLED)
            self.status_var.set("Processing...")
            self.root.update_idletasks()

            # Preprocess if enabled
            if self.preprocess_var.get():
                self.status_var.set("Preprocessing image...")
                self.root.update_idletasks()
                
                import cv2
                preprocessed = preprocess_image(
                    self.img_path,
                    enhance=self.enhance_var.get(),
                    denoise=self.denoise_var.get()
                )
                temp_path = OUTPUT_DIR / "temp_preprocessed.jpg"
                cv2.imwrite(str(temp_path), preprocessed)
                detection_path = temp_path
            else:
                detection_path = self.img_path

            # Load image
            frame = cv2.imread(str(detection_path))
            if frame is None:
                raise ValueError("Failed to load image")
            
            # Use two-stage detection if enabled
            start_time = time.time()
            
            if self.use_road_seg_var.get() and self.two_stage_detector and self.two_stage_detector.use_road_seg:
                self.status_var.set("Segmenting road...")
                self.root.update_idletasks()
                results, road_mask = self.two_stage_detector.detect_potholes(
                    frame, 
                    conf=_conf,
                    return_mask=True,
                    lowres_width=ROAD_MASK_WIDTH
                )
                annotated = self.two_stage_detector.visualize(
                    frame, results, road_mask, 
                    show_mask=self.show_road_mask_var.get()
                )
            else:
                # Single-stage detection
                results = self.two_stage_detector.pothole_model(frame, conf=_conf, verbose=False)[0]
                annotated = results.plot()
            
            inf_time = (time.time() - start_time) * 1000
            
            # Save result
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"pred_{ts}_{Path(self.img_path).name}"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
            
            # Calculate statistics
            num_potholes = len(results.boxes)
            if num_potholes > 0:
                avg_conf = float(results.boxes.conf.mean())
            else:
                avg_conf = 0.0
            
            self._show_on_canvas(out_path)
            
            # Update statistics
            self.potholes_label.config(
                text=f"Potholes: {num_potholes}",
                fg=ERROR_COLOR if num_potholes > 0 else SUCCESS_COLOR
            )
            self.confidence_label.config(text=f"Confidence: {avg_conf:.1%}")
            self.time_label.config(text=f"Time: {inf_time:.1f}ms")
            
            # Update performance monitor
            self.performance_monitor.add_detection(num_potholes, inf_time, avg_conf)
            self._update_performance_display()
            
            # Save to history
            self.detection_history.append({
                'timestamp': ts,
                'image': Path(self.img_path).name,
                'num_potholes': num_potholes,
                'avg_confidence': avg_conf,
                'inference_time': inf_time
            })
            
            status_msg = f"Complete - {num_potholes} pothole(s) detected"
            self.status_var.set(status_msg)
            
            result_msg = (
                f"Detection Results:\n\n"
                f"Potholes: {num_potholes}\n"
                f"Confidence: {avg_conf:.1%}\n"
                f"Time: {inf_time:.1f}ms\n\n"
                f"Saved to output folder"
            )
            messagebox.showinfo("Complete", result_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection error: {str(e)}")
            self.status_var.set("Error during detection")
            logger.error(f"Detection error: {e}")
        finally:
            self.detect_btn.config(state=tk.NORMAL)

    def _detect_video(self):
        """Run detection on current video using two-stage approach"""
        if not self.vid_path:
            messagebox.showerror("Error", "Please select a video first")
            return

        if self.video_thread and self.video_thread.is_alive():
            messagebox.showinfo("Info", "Video processing is already running")
            return

        self.detect_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing video...")
        self.root.update_idletasks()

        use_road_seg = self.use_road_seg_var.get()
        show_road_mask = self.show_road_mask_var.get()
        
        self.video_queue = queue.Queue()
        self.video_thread = threading.Thread(
            target=self._detect_video_worker,
            args=(use_road_seg, show_road_mask),
            daemon=True
        )
        self.video_thread.start()
        self._poll_video_queue()

    def _detect_video_worker(self, use_road_seg: bool, show_road_mask: bool):
        """Background worker for video detection"""
        try:
            cap = cv2.VideoCapture(self.vid_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"pred_{ts}_{Path(self.vid_path).stem}.mp4"
            out_path = OUTPUT_DIR / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

            frame_idx = 0
            total_potholes = 0
            total_time = 0.0
            use_road_seg = use_road_seg and self.two_stage_detector and self.two_stage_detector.use_road_seg
            last_road_mask = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                progress = int(100 * frame_idx / max(1, frame_count))

                t0 = time.time()
                if use_road_seg:
                    if frame_idx == 1 or frame_idx % ROAD_MASK_STRIDE == 0 or last_road_mask is None:
                        last_road_mask = self.two_stage_detector.get_road_mask(frame, lowres_width=ROAD_MASK_WIDTH)
                    results = self.two_stage_detector.detect_potholes(
                        frame,
                        conf=_conf,
                        return_mask=False,
                        road_mask=last_road_mask,
                        lowres_width=ROAD_MASK_WIDTH
                    )
                    annotated = self.two_stage_detector.visualize(frame, results, last_road_mask, show_mask=show_road_mask)
                else:
                    results = self.two_stage_detector.pothole_model(frame, conf=_conf, verbose=False)[0]
                    annotated = results.plot()
                
                total_time += (time.time() - t0) * 1000
                total_potholes += len(results.boxes) if results.boxes else 0

                writer.write(annotated)

                if frame_idx % max(1, frame_count // 10) == 0 or frame_idx == 1:
                    display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    h, w = display_frame.shape[:2]
                    max_size = max(h, w)
                    if max_size > CANVAS_W:
                        scale = CANVAS_W / max_size
                        display_frame = cv2.resize(display_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    self.video_queue.put(("frame", display_frame, progress, total_potholes))

            cap.release()
            writer.release()

            avg_time = total_time / max(1, frame_idx)
            self.video_queue.put(("done", out_path, total_potholes, frame_idx, avg_time))

        except Exception as e:
            self.video_queue.put(("error", str(e)))

    def _poll_video_queue(self):
        """Poll for video processing updates"""
        try:
            while True:
                msg = self.video_queue.get_nowait()
                msg_type = msg[0]

                if msg_type == "frame":
                    _, display_frame, progress, total_potholes = msg
                    pil_frame = Image.fromarray(display_frame.astype('uint8'), 'RGB')
                    self.tk_img = ImageTk.PhotoImage(pil_frame)
                    self.canvas.delete("all")
                    self.canvas.create_image(CANVAS_W // 2, CANVAS_H // 2, image=self.tk_img)
                    self.status_var.set(f"Processing... {progress}% | Potholes detected: {total_potholes}")
                elif msg_type == "done":
                    _, out_path, total_potholes, frame_count, avg_time = msg
                    self.potholes_label.config(
                        text=f"Potholes: {total_potholes}",
                        fg=ERROR_COLOR if total_potholes > 0 else SUCCESS_COLOR
                    )
                    self.confidence_label.config(text=f"Avg Time: {avg_time:.1f}ms/frame")
                    self.time_label.config(text=f"Frames: {frame_count}")

                    self.status_var.set(f"Video saved: {out_path.name}")
                    messagebox.showinfo(
                        "Complete",
                        f"Video processing complete!\n\n"
                        f"Potholes: {total_potholes}\n"
                        f"Frames: {frame_count}\n"
                        f"Avg Time: {avg_time:.1f}ms/frame\n\n"
                        f"Saved to output folder"
                    )
                    self.detect_btn.config(state=tk.NORMAL)
                    return
                elif msg_type == "error":
                    _, err = msg
                    messagebox.showerror("Error", f"Video processing error: {err}")
                    self.status_var.set("Error during video processing")
                    self.detect_btn.config(state=tk.NORMAL)
                    return

        except queue.Empty:
            pass

        if self.video_thread and self.video_thread.is_alive():
            self.root.after(50, self._poll_video_queue)
        else:
            self.detect_btn.config(state=tk.NORMAL)

    def batch_process(self):
        """Batch process multiple images"""
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        
        self.status_var.set("Processing batch...")
        self.root.update_idletasks()
        
        try:
            result = batch_process_images(folder, roi=None)
            if result:
                msg = (
                    f"Batch Processing Complete!\n\n"
                    f"Images: {result['total_images']}\n"
                    f"Potholes: {result['total_potholes']}\n"
                    f"Avg Time: {result['avg_inference_time']:.1f}ms\n\n"
                    f"Results saved to:\n{result['output_folder']}"
                )
                messagebox.showinfo("Batch Complete", msg)
                self.status_var.set(f"Batch complete: {result['total_images']} images processed")
            else:
                messagebox.showwarning("Warning", "No images found in folder")
        except Exception as e:
            messagebox.showerror("Error", f"Batch processing error: {str(e)}")
            logger.error(f"Batch error: {e}")

    def export_csv(self):
        """Export detection history to CSV"""
        if not self.detection_history:
            messagebox.showinfo("Info", "No detections to export")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"detections_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if path:
            try:
                export_detections_csv(self.detection_history, path)
                messagebox.showinfo("Success", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export error: {e}")

    def export_json(self):
        """Export detection history to JSON"""
        if not self.detection_history:
            messagebox.showinfo("Info", "No detections to export")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"detections_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        if path:
            try:
                export_detections_json(self.detection_history, path)
                messagebox.showinfo("Success", f"Exported to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export error: {e}")

    def save_current(self):
        """Save current result (Ctrl+S)"""
        # Placeholder for future implementation
        pass

    def reset_monitor(self):
        """Reset performance monitor"""
        self.performance_monitor.reset()
        self._update_performance_display()
        self.fps_label.config(text="Avg FPS: --")

    def _update_performance_display(self):
        """Update performance monitor display"""
        stats = self.performance_monitor.get_stats()
        if stats:
            fps = stats['fps']
            self.fps_label.config(text=f"Avg FPS: {fps:.1f}")
            
            perf_text = (
                f"Total: {stats['total_detections']} detections\n"
                f"Potholes: {stats['total_potholes']}\n"
                f"Avg Time: {stats['avg_inference_time']:.1f}ms\n"
                f"Range: {stats['min_inference_time']:.1f}-{stats['max_inference_time']:.1f}ms"
            )
            self.perf_label.config(text=perf_text)

    def open_output_folder(self):
        """Open output folder in file explorer"""
        abs_path = OUTPUT_DIR.resolve()
        if os.name == "nt":
            os.startfile(str(abs_path))
        else:
            webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeAppEnhanced(root)
    root.mainloop()
