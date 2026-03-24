# app/pothole_app_filtered.py
"""
Enhanced Pothole Detection App with:
1. Road-area only detection (segmentation masking)
2. Class filtering checkboxes
3. Configurable visualization
"""
import json
import random
import time
import tkinter as tk
import threading
import queue
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from datetime import datetime, timezone
from PIL import ImageTk, Image
import logging
import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors and typography
BG_COLOR = "#f4f7fb"
PANEL_COLOR = "#ffffff"
PRIMARY_COLOR = "#1769e0"
SUCCESS_COLOR = "#0b8f4d"
ERROR_COLOR = "#cc2f3d"
TEXT_COLOR = "#1e2a3a"
MUTED_TEXT = "#5a6b7f"
ACCENT_COLOR = "#f8b400"
FONT_UI = "Segoe UI"

class PotholeAppFiltered:
    def __init__(self, root):
        self.root = root
        self.root.title("PAVE - Smart Pothole Event Review")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 760)
        self.root.configure(bg=BG_COLOR)
        
        self.pothole_model = None
        self.road_model = None
        self.current_image = None
        self.current_result = None
        
        # Default models
        pothole_path = MODEL_DIR / "best.pt"  # Your original trained model
        road_path = MODEL_DIR / "road_seg.pt"  # Road segmentation model
        
        try:
            logger.info("Loading pothole detection model...")
            self.pothole_model = YOLO(str(pothole_path))
            
            if road_path.exists():
                logger.info("Loading road segmentation model...")
                self.road_model = YOLO(str(road_path))
            else:
                logger.warning(f"Road model not found at {road_path}")
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load models: {e}")
            return
        
        # Detection settings
        self.conf_threshold = tk.DoubleVar(value=0.35)
        self.use_road_mask = tk.BooleanVar(value=True)
        self.show_road_overlay = tk.BooleanVar(value=True)

        # Video runtime state
        self.video_queue = queue.Queue(maxsize=2)
        self.video_thread = None
        self.video_running = False
        self.video_capture = None
        self.video_output_dir = None
        self.video_frame_idx = 0
        self.saved_detection_frames = 0
        self.runtime_settings = {}
        self.events_jsonl_path = None
        self.events_records = []
        self.current_video_name = "-"
        self.demo_base_lat = 37.7749
        self.demo_base_lon = -122.4194
        
        # Class selection checkboxes
        self.class_visibility = {}  # {class_id: BooleanVar}
        self.road_class_filter = {}  # {class_id: BooleanVar} for road segmentation classes
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the UI layout"""
        # Header
        header = tk.Frame(self.root, bg=PRIMARY_COLOR, height=82)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="PAVE · Pothole Detection + Event Tagging",
            font=(FONT_UI, 19, "bold"),
            bg=PRIMARY_COLOR,
            fg="white"
        ).pack(pady=(12, 2))
        tk.Label(
            header,
            text="Realtime video detection, flagged pothole frames, and per-frame JSON metadata",
            font=(FONT_UI, 10),
            bg=PRIMARY_COLOR,
            fg="#d9e8ff",
        ).pack()
        
        # Main container
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel
        left = tk.Frame(main, bg=BG_COLOR, width=360)
        left.pack(side="left", fill="both", padx=(0, 10))
        left.pack_propagate(False)
        
        # File selector
        file_frame = tk.LabelFrame(
            left,
            text="Input",
            font=(FONT_UI, 11, "bold"),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
        )
        file_frame.pack(fill="x", pady=(0, 10))
        
        tk.Button(
            file_frame,
            text="Choose Image / Video",
            command=self.browse_file,
            bg=PRIMARY_COLOR,
            fg="white",
            font=(FONT_UI, 10, "bold"),
            padx=10,
            pady=7,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=5, pady=5)
        
        self.file_label = tk.Label(
            file_frame,
            text="No file selected",
            wraplength=300,
            justify="left",
            bg=PANEL_COLOR,
            fg=MUTED_TEXT,
            font=(FONT_UI, 9),
        )
        self.file_label.pack(padx=5, pady=5)
        
        # Confidence threshold
        conf_frame = tk.LabelFrame(
            left,
            text="Detector Confidence",
            font=(FONT_UI, 11, "bold"),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
        )
        conf_frame.pack(fill="x", pady=(0, 10))
        
        tk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.conf_threshold,
            bg=PANEL_COLOR,
            length=250
        ).pack(fill="x", padx=5, pady=5)

        self.conf_value_label = tk.Label(
            conf_frame,
            text=f"Value: {self.conf_threshold.get():.2f}",
            bg=PANEL_COLOR,
            fg=MUTED_TEXT,
            font=(FONT_UI, 9),
        )
        self.conf_value_label.pack(anchor="w", padx=6)
        self.conf_threshold.trace_add("write", lambda *_: self.conf_value_label.config(text=f"Value: {self.conf_threshold.get():.2f}"))
        
        # Road masking options
        mask_frame = tk.LabelFrame(
            left,
            text="Road Filter Options",
            font=(FONT_UI, 11, "bold"),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
        )
        mask_frame.pack(fill="x", pady=(0, 10))
        
        tk.Checkbutton(
            mask_frame,
            text="Use Road Mask (Seg Model)",
            variable=self.use_road_mask,
            bg=PANEL_COLOR,
            font=(FONT_UI, 9)
        ).pack(anchor="w", padx=5, pady=3)
        
        tk.Checkbutton(
            mask_frame,
            text="Show Road Overlay",
            variable=self.show_road_overlay,
            bg=PANEL_COLOR,
            font=(FONT_UI, 9)
        ).pack(anchor="w", padx=5, pady=3)
        
        # Detection class selection
        class_frame = tk.LabelFrame(
            left,
            text="Class Filters",
            font=(FONT_UI, 11, "bold"),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
        )
        class_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Scrollable class list
        canvas = tk.Canvas(class_frame, bg=PANEL_COLOR, highlightthickness=0, height=220)
        scrollbar = tk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
        class_list = tk.Frame(canvas, bg=PANEL_COLOR)
        
        class_list.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=class_list, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_class_mousewheel(event):
            # Windows uses 120-step wheel deltas; scale to Tk scroll units.
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_class_mousewheel(_event):
            canvas.bind_all("<MouseWheel>", _on_class_mousewheel)

        def _unbind_class_mousewheel(_event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_class_mousewheel)
        canvas.bind("<Leave>", _unbind_class_mousewheel)
        
        # Pothole model classes
        if self.pothole_model:
            tk.Label(class_list, text="Detection Classes:", font=(FONT_UI, 9, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=5, pady=(5, 3))
            classes = self.pothole_model.names
            for class_id, class_name in classes.items():
                var = tk.BooleanVar(value=True)
                self.class_visibility[class_id] = var
                tk.Checkbutton(
                    class_list,
                    text=f"{class_name}",
                    variable=var,
                    bg=PANEL_COLOR,
                    font=(FONT_UI, 9)
                ).pack(anchor="w", padx=5, pady=2)
        
        # Road segmentation filter classes
        if self.road_model:
            tk.Label(class_list, text="\nRoad Type Filter:", font=(FONT_UI, 9, "bold"), bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=5, pady=(8, 3))
            self.road_class_filter = {}  # {class_id: BooleanVar}
            road_classes = self.road_model.names
            for class_id, class_name in road_classes.items():
                var = tk.BooleanVar(value=True)
                self.road_class_filter[class_id] = var
                tk.Checkbutton(
                    class_list,
                    text=f"{class_name}",
                    variable=var,
                    bg=PANEL_COLOR,
                    font=(FONT_UI, 8)
                ).pack(anchor="w", padx=5, pady=2)

        # Live event feed
        event_frame = tk.LabelFrame(
            left,
            text="Flagged Pothole Events",
            font=(FONT_UI, 11, "bold"),
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            padx=8,
            pady=8,
        )
        event_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.event_listbox = tk.Listbox(
            event_frame,
            height=7,
            bg="#f8fbff",
            fg=TEXT_COLOR,
            font=("Consolas", 9),
            selectbackground="#dbe9ff",
            relief="flat",
        )
        self.event_listbox.pack(fill="both", expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Action buttons
        btn_frame = tk.Frame(left, bg=BG_COLOR)
        btn_frame.pack(fill="x", pady=(0, 10))
        
        tk.Button(
            btn_frame,
            text="Start Detection",
            command=self.run_detection,
            bg=SUCCESS_COLOR,
            fg="white",
            font=(FONT_UI, 10, "bold"),
            padx=10,
            pady=9,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", pady=(0, 5))

        tk.Button(
            btn_frame,
            text="Stop Video",
            command=self.stop_video_detection,
            bg=ERROR_COLOR,
            fg="white",
            font=(FONT_UI, 10, "bold"),
            padx=10,
            pady=9,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", pady=(0, 5))
        
        tk.Button(
            btn_frame,
            text="Save Current View",
            command=self.save_result,
            bg=PRIMARY_COLOR,
            fg="white",
            font=(FONT_UI, 10, "bold"),
            padx=10,
            pady=9,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x")
        
        # Right panel - Canvas for display
        right = tk.Frame(main, bg=PANEL_COLOR)
        right.pack(side="right", fill="both", expand=True)

        # KPI strip
        kpi_row = tk.Frame(right, bg=PANEL_COLOR)
        kpi_row.pack(fill="x", padx=12, pady=(8, 6))

        self.kpi_video_var = tk.StringVar(value="Video: -")
        self.kpi_frame_var = tk.StringVar(value="Frame: 0")
        self.kpi_det_var = tk.StringVar(value="Detections: 0")
        self.kpi_saved_var = tk.StringVar(value="Flagged Saved: 0")

        for var in [self.kpi_video_var, self.kpi_frame_var, self.kpi_det_var, self.kpi_saved_var]:
            tk.Label(
                kpi_row,
                textvariable=var,
                bg="#edf4ff",
                fg=TEXT_COLOR,
                font=(FONT_UI, 9, "bold"),
                padx=10,
                pady=6,
                relief="flat",
            ).pack(side="left", padx=(0, 8))
        
        self.canvas = tk.Canvas(right, bg="#0b1624", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#e9eff8", height=34)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready. Select an image/video to get started.",
            font=(FONT_UI, 9),
            bg="#e9eff8",
            fg=MUTED_TEXT,
            justify="left"
        )
        self.status_label.pack(anchor="w", padx=10, pady=5)
    
    def update_status(self, msg):
        """Update status bar"""
        self.status_label.config(text=msg)
        self.root.update()
    
    def browse_file(self):
        """Browse and select image/video"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("Video files", "*.mp4 *.avi *.mov"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes, initialdir=ROOT)
        if path:
            self.file_path = path
            self.file_label.config(text=f"File:\n{Path(path).name}")
            self.current_video_name = Path(path).name
            self.kpi_video_var.set(f"Video: {self.current_video_name[:28]}")
            self.update_status(f"Loaded: {Path(path).name}")

    def _random_dummy_gps(self, frame_idx):
        """Generate stable-ish dummy GPS values near a base coordinate."""
        lat = self.demo_base_lat + random.uniform(-0.0008, 0.0008) + (frame_idx * 1e-7)
        lon = self.demo_base_lon + random.uniform(-0.0008, 0.0008) - (frame_idx * 1e-7)
        return round(lat, 7), round(lon, 7)

    def _build_event_id(self, frame_idx):
        """Generate unique event/frame identifier for flagged pothole frames."""
        return f"PH-{time.strftime('%Y%m%d%H%M%S')}-{frame_idx:06d}-{self.saved_detection_frames + 1:04d}"

    def _append_event_row(self, event_id, potholes_detected):
        """Push latest event to the UI event feed list."""
        row = f"{event_id} | potholes={potholes_detected}"
        self.event_listbox.insert(tk.END, row)
        self.event_listbox.yview_moveto(1.0)

    def _get_drivable_class_ids(self):
        """Infer road-like classes from segmentation names for robust pothole gating."""
        if not self.road_model:
            return set()

        names = self.road_model.names
        road_tokens = ("road", "lane", "drivable", "street", "asphalt", "pavement")
        blocked_tokens = ("object", "obstacle", "vehicle", "pedestrian", "shadow", "vegetation")
        drivable = {
            int(class_id)
            for class_id, class_name in names.items()
            if any(tok in str(class_name).lower() for tok in road_tokens)
            and not any(tok in str(class_name).lower() for tok in blocked_tokens)
        }

        # Fallback for unexpected label names.
        if not drivable and names:
            drivable = {int(next(iter(names.keys())))}
        return drivable

    def _find_class_id_by_token(self, token):
        """Return first segmentation class id containing the token in its name."""
        if not self.road_model:
            return None
        for class_id, class_name in self.road_model.names.items():
            if token in str(class_name).lower():
                return int(class_id)
        return None

    def _refine_road_mask(self, mask):
        """Remove tiny/side artifacts from road mask to avoid false road regions."""
        if mask is None:
            return None

        h, w = mask.shape[:2]
        binary = (mask > 0).astype(np.uint8)

        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        min_area = max(300, int(0.001 * h * w))

        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area < min_area:
                continue

            cx, cy = centroids[idx]
            touches_bottom = (y + bh) >= int(0.78 * h)
            near_center = (0.12 * w) <= cx <= (0.88 * w)
            if touches_bottom and near_center:
                keep[labels == idx] = 1

        if np.count_nonzero(keep) == 0:
            # Fallback: keep only lower-half regions to avoid masking vehicles/signs/sky.
            lower = binary.copy()
            lower[: int(0.5 * h), :] = 0
            keep = lower

        return (keep * 255).astype(np.uint8)
    
    def get_road_mask(self, frame, use_road_mask=True, enabled_road_classes=None):
        """Extract road mask from frame"""
        if not use_road_mask or not self.road_model:
            return None
        
        try:
            h, w = frame.shape[:2]
            results = self.road_model.predict(frame, conf=0.5, verbose=False)
            
            if results and results[0].masks:
                # Combine filtered road classes
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                class_masks = {}
                
                for i, (mask, class_id) in enumerate(zip(results[0].masks.data, results[0].boxes.cls)):
                    class_id = int(class_id)

                    mask_resized = cv2.resize(
                        mask.cpu().numpy().astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    class_masks[class_id] = np.maximum(
                        class_masks.get(class_id, np.zeros((h, w), dtype=np.uint8)),
                        mask_resized * 255,
                    )
                    
                    # Keep only enabled road classes from a thread-safe snapshot.
                    if enabled_road_classes is not None and class_id not in enabled_road_classes:
                        continue

                    combined_mask = np.maximum(combined_mask, mask_resized * 255)

                # Fallback for mislabeled models: if visible_road selected but empty mask,
                # borrow roadside_object mask as surrogate road area.
                if enabled_road_classes is not None and np.count_nonzero(combined_mask) < int(0.005 * h * w):
                    visible_id = self._find_class_id_by_token("visible_road")
                    roadside_id = self._find_class_id_by_token("roadside_object")
                    if (
                        visible_id is not None
                        and roadside_id is not None
                        and visible_id in enabled_road_classes
                        and roadside_id in class_masks
                    ):
                        combined_mask = np.maximum(combined_mask, class_masks[roadside_id])
                
                return self._refine_road_mask(combined_mask)
        except Exception as e:
            logger.error(f"Road mask extraction failed: {e}")
        
        return None
    
    def run_detection(self):
        """Run pothole detection with filtering"""
        if not hasattr(self, 'file_path'):
            messagebox.showwarning("No File", "Please select an image or video first")
            return
        
        if not self.pothole_model:
            messagebox.showerror("Model Error", "Pothole model not loaded")
            return
        
        self.update_status("Running detection...")
        
        # Read image or start realtime video loop
        path = Path(self.file_path)
        if path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            self.start_video_detection(path)
            return
        else:
            frame = cv2.imread(str(path))
        
        if frame is None:
            messagebox.showerror("Error", "Could not read file")
            return
        
        self.current_image = frame
        settings = self._snapshot_runtime_settings()
        display, num_det, _ = self._detect_frame(frame, settings=settings)
        
        # Display result
        self._display_on_canvas(display)
        self.current_result = display
        self.update_status(f"Detection complete. Potholes: {num_det}. Result ready to save.")

    def _snapshot_runtime_settings(self):
        """Capture Tk UI values into plain Python data for safe use in worker threads."""
        enabled_det_classes = {
            int(class_id)
            for class_id, var in self.class_visibility.items()
            if var.get()
        }
        enabled_road_classes = {
            int(class_id)
            for class_id, var in self.road_class_filter.items()
            if var.get()
        }
        drivable_classes = self._get_drivable_class_ids()
        if enabled_road_classes:
            effective_road_classes = enabled_road_classes & drivable_classes
            if not effective_road_classes:
                effective_road_classes = drivable_classes
        else:
            effective_road_classes = drivable_classes

        settings = {
            "conf": float(self.conf_threshold.get()),
            "use_road_mask": bool(self.use_road_mask.get()),
            "show_road_overlay": bool(self.show_road_overlay.get()),
            "enabled_det_classes": enabled_det_classes,
            "enabled_road_classes": effective_road_classes,
        }
        self.runtime_settings = settings
        return settings

    def _detect_frame(self, frame, settings=None):
        """Run full frame detection with optional road mask filtering and overlay."""
        if settings is None:
            settings = self._snapshot_runtime_settings()

        road_mask = self.get_road_mask(
            frame,
            use_road_mask=settings["use_road_mask"],
            enabled_road_classes=settings["enabled_road_classes"],
        )
        conf = settings["conf"]
        results = self.pothole_model.predict(frame, conf=conf, verbose=False)

        display = frame.copy()
        num_det = 0

        if results and results[0].boxes:
            for box, class_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                class_id = int(class_id)

                if class_id not in settings["enabled_det_classes"]:
                    continue

                x1, y1, x2, y2 = map(int, box)

                if road_mask is not None:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    center_x = int(np.clip(center_x, 0, road_mask.shape[1] - 1))
                    center_y = int(np.clip(center_y, 0, road_mask.shape[0] - 1))
                    if road_mask[center_y, center_x] == 0:
                        continue

                num_det += 1
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    self.pothole_model.names[class_id],
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        if road_mask is not None and settings["show_road_overlay"]:
            mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0
            mask_colored[:, :, 2] = 0
            display = cv2.addWeighted(display, 0.8, mask_colored, 0.2, 0)

        cv2.putText(
            display,
            f"Potholes: {num_det}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        return display, num_det, road_mask

    def start_video_detection(self, path: Path):
        """Start realtime detection on video file."""
        self.stop_video_detection(silent=True)
        self._snapshot_runtime_settings()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video: {path.name}")
            return

        self.video_capture = cap
        self.video_running = True
        self.video_frame_idx = 0
        self.saved_detection_frames = 0
        self.events_records = []
        self.event_listbox.delete(0, tk.END)
        self.demo_base_lat = 37.7749 + random.uniform(-0.05, 0.05)
        self.demo_base_lon = -122.4194 + random.uniform(-0.05, 0.05)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_output_dir = OUTPUT_DIR / f"video_detect_{timestamp}"
        (self.video_output_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (self.video_output_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.video_output_dir / "road_mask").mkdir(parents=True, exist_ok=True)
        (self.video_output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        self.events_jsonl_path = self.video_output_dir / "metadata" / "events.jsonl"
        self.kpi_frame_var.set("Frame: 0")
        self.kpi_det_var.set("Detections: 0")
        self.kpi_saved_var.set("Flagged Saved: 0")

        self.update_status("Realtime video detection started...")
        self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.video_thread.start()
        self._consume_video_queue()

    def _video_worker(self):
        """Background worker for video decode + inference."""
        cap = self.video_capture
        settings = dict(self.runtime_settings)
        if cap is None:
            self.video_running = False
            return

        while self.video_running:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            self.video_frame_idx += 1
            display, num_det, road_mask = self._detect_frame(frame, settings=settings)

            if num_det > 0 and self.video_output_dir is not None:
                event_id = self._build_event_id(self.video_frame_idx)
                timestamp_iso = datetime.now(timezone.utc).isoformat()
                lat, lon = self._random_dummy_gps(self.video_frame_idx)

                cv2.putText(
                    display,
                    f"FLAGGED FRAME | ID={event_id}",
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 70, 255),
                    2,
                )

                stem = event_id
                cv2.imwrite(str(self.video_output_dir / "frames" / f"{stem}.jpg"), frame)
                cv2.imwrite(str(self.video_output_dir / "annotated" / f"{stem}.jpg"), display)
                if road_mask is not None:
                    cv2.imwrite(str(self.video_output_dir / "road_mask" / f"{stem}.png"), road_mask)

                event_record = {
                    "id": event_id,
                    "potholes_detected": int(num_det),
                    "latitude": lat,
                    "longitude": lon,
                    "timestamp": timestamp_iso,
                    "frame_index": int(self.video_frame_idx),
                    "frame_path": str((self.video_output_dir / "frames" / f"{stem}.jpg").name),
                    "annotated_path": str((self.video_output_dir / "annotated" / f"{stem}.jpg").name),
                    "road_mask_path": str((self.video_output_dir / "road_mask" / f"{stem}.png").name) if road_mask is not None else None,
                }
                self.events_records.append(event_record)

                if self.events_jsonl_path is not None:
                    with self.events_jsonl_path.open("a", encoding="utf-8") as jf:
                        jf.write(json.dumps(event_record) + "\n")

                with (self.video_output_dir / "metadata" / f"{stem}.json").open("w", encoding="utf-8") as sf:
                    json.dump(event_record, sf, indent=2)

                self.saved_detection_frames += 1
            else:
                event_id = None

            payload = (display, num_det, self.video_frame_idx, self.saved_detection_frames, event_id)
            try:
                self.video_queue.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.video_queue.get_nowait()
                except queue.Empty:
                    pass
                self.video_queue.put_nowait(payload)

        self.video_running = False
        if cap is not None:
            cap.release()
        self.video_capture = None

        if self.video_output_dir is not None:
            summary_path = self.video_output_dir / "metadata" / "events_summary.json"
            with summary_path.open("w", encoding="utf-8") as outj:
                json.dump(self.events_records, outj, indent=2)

    def _consume_video_queue(self):
        """Transfer processed frames from worker thread to Tk canvas."""
        latest = None
        while not self.video_queue.empty():
            try:
                latest = self.video_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            display, num_det, frame_idx, saved_count, event_id = latest
            self.current_result = display
            self._display_on_canvas(display)
            self.kpi_frame_var.set(f"Frame: {frame_idx}")
            self.kpi_det_var.set(f"Detections: {num_det}")
            self.kpi_saved_var.set(f"Flagged Saved: {saved_count}")
            if event_id is not None:
                self._append_event_row(event_id, num_det)
            self.update_status(
                f"Video realtime | Frame {frame_idx} | Detections {num_det} | Saved {saved_count} pothole frames"
            )

        if self.video_running:
            self.root.after(15, self._consume_video_queue)
        else:
            end_msg = f"Video complete. Saved {self.saved_detection_frames} pothole frames"
            if self.video_output_dir is not None:
                end_msg += f" to {self.video_output_dir}"
            self.update_status(end_msg)

    def stop_video_detection(self, silent: bool = False):
        """Stop realtime video detection loop."""
        was_running = self.video_running
        self.video_running = False
        cap = self.video_capture
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.video_capture = None
        if was_running and not silent:
            self.update_status("Video detection stopped")

    def _on_close(self):
        """Clean shutdown for background processing threads."""
        self.stop_video_detection(silent=True)
        self.root.destroy()
    
    def _display_on_canvas(self, img):
        """Display image on canvas"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100:
            canvas_w = 800
        if canvas_h < 100:
            canvas_h = 600
        
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        pil_img = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_img)
    
    def save_result(self):
        """Save detection result"""
        if self.current_result is None:
            messagebox.showwarning("No Result", "Run detection first")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"detection_{timestamp}.jpg"
        
        cv2.imwrite(str(output_path), self.current_result)
        messagebox.showinfo("Saved", f"Saved to:\n{output_path}")
        self.update_status(f"Result saved: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeAppFiltered(root)
    root.mainloop()
