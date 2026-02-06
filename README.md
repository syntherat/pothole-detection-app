# Pothole Detection Application

An intelligent real-time pothole detection system powered by YOLOv11/YOLOv8 that identifies road potholes in images and videos with high accuracy. Features a user-friendly GUI, batch processing capabilities, and optional two-stage detection to minimize false positives.

---

## 🎯 About The Project

This project provides a complete end-to-end solution for pothole detection on road surfaces. It includes:
- **Pre-trained YOLO models** fine-tuned for pothole detection
- **Interactive desktop GUI** for single image analysis
- **Batch processing scripts** for handling multiple images/videos
- **Training pipeline** to create custom models on your own datasets
- **Two-stage detection** combining road segmentation with pothole detection for improved accuracy
- **Comprehensive evaluation tools** to measure model performance

Built for transportation departments, road maintenance crews, researchers, and developers working on smart city infrastructure.

---

## ✨ Features

### Detection Capabilities
- **State-of-the-Art Models**: YOLOv11 (nano, small, medium) and YOLOv8 architectures
- **Real-time Performance**: 2-15ms inference time depending on model size
- **Confidence Scoring**: Adjustable detection threshold (10-90%) for sensitivity control
- **Batch Processing**: Process entire folders of images or videos at once
- **Two-Stage Detection**: Optional road surface segmentation to reduce false positives

### User Interface
- **Interactive Tkinter GUI**: Easy-to-use desktop application
- **Visual Feedback**: Live bounding boxes, confidence scores, and statistics
- **Model Switching**: Load different trained models on-the-fly
- **Detection Metrics**: Real-time display of detection count, confidence, and inference time

### Training & Evaluation
- **Complete Training Pipeline**: Automated dataset organization and model training
- **Multiple Model Sizes**: Train nano (fastest), small (balanced), or medium (most accurate)
- **Hyperparameter Presets**: Baseline, aggressive, and conservative training configurations
- **Performance Evaluation**: Comprehensive mAP, precision, recall metrics
- **Result Tracking**: Timestamped training runs with full metric logging

---

## 📁 Project Structure

```
pothole_detect_app/
├── app/
│   ├── main_enhanced.py         # GUI application with advanced features
│   ├── two_stage_detection.py   # Two-stage detector implementation
│   ├── utils.py                 # Core detection utilities
│   └── enhanced_utils.py        # Extended utility functions
├── scripts/
│   ├── organize_dataset.py      # Dataset preparation (train/val/test split)
│   ├── train_model.py           # Model training with hyperparameter options
│   ├── evaluate_model.py        # Model performance evaluation
│   ├── predict_script.py        # Batch image prediction
│   ├── predict_videos.py        # Video processing
│   ├── download_road_model.py   # Download road segmentation model
│   └── test_road_segmentation.py # Test road segmentation
├── data/
│   ├── data.yaml                # Dataset configuration
│   ├── raw/                     # Raw images and annotations (VOC format)
│   └── dataset_v*/              # Organized YOLO format datasets
├── model/                       # Trained model weights (.pt files)
├── quick_start.bat              # Windows quick start script
├── train_gpu.bat                # Training automation script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.10 recommended)
- **pip** package manager
- **CUDA-capable GPU** (optional but recommended for training)
- **4GB+ RAM** for inference, 8GB+ for training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pothole_detect_app.git
   cd pothole_detect_app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or train a model**
   - **Option A**: Place your pre-trained model at `model/best.pt`
   - **Option B**: Use the GUI's "Select Model" button to load any `.pt` file
   - **Option C**: Train your own (see Training section below)

---

## 💻 Usage

### GUI Application (Recommended)

```bash
python app/main_enhanced.py
```

**How to use:**
1. Click **"Upload Image"** to select a road image
2. Adjust **confidence threshold** slider (default: 35%)
3. Click **"Detect Potholes"** to run detection
4. View annotated results with bounding boxes and statistics
5. Results are automatically saved to `app/output/`

### Batch Image Processing

Process multiple images at once:

```bash
python scripts/predict_script.py --input ./input --output ./output --conf 0.35
```

Options:
- `--input`: Input directory containing images
- `--output`: Output directory for annotated images
- `--conf`: Confidence threshold (0.0-1.0)

### Video Processing

Process video files:

```bash
python scripts/predict_videos.py
```

The script will process all videos in the `input/` directory and save results to `output/videos/`

---

## 🏋️ Training Your Own Model

### Using Quick Start (Windows)

```bash
quick_start.bat
```

This automated script will:
1. Organize your dataset
2. Train a YOLOv11-small model with baseline hyperparameters
3. Evaluate performance
4. Generate prediction examples

### Manual Training

**Step 1: Prepare Your Dataset**

Place your annotated images in `data/raw/`:
- `data/raw/images/` - Road images
- `data/raw/annotations/` - VOC XML format annotations

Then organize the dataset:

```bash
python scripts/organize_dataset.py
```

This creates a proper train/val/test split (70/15/15) in YOLO format.

**Step 2: Train a Model**

```bash
# Train with default settings (small model, baseline hyperparameters)
python scripts/train_model.py --model small --hyperparams baseline

# Train all model sizes for comparison
python scripts/train_model.py --all

# Custom training
python scripts/train_model.py --model medium --hyperparams aggressive --epochs 150
```

**Model Options:**
- `nano`: Fastest inference (~2ms), good for embedded systems
- `small`: Balanced speed/accuracy (~5ms), recommended
- `medium`: Highest accuracy (~15ms), best for server deployment

**Hyperparameter Presets:**
- `baseline`: Standard training, good for most cases
- `aggressive`: Heavy augmentation, prevents overfitting
- `conservative`: Light augmentation, good for small datasets

Results are saved to `training_results/pothole_<model>_<preset>_<timestamp>/`

**Step 3: Evaluate Performance**

```bash
python scripts/evaluate_model.py
```

Generates comprehensive metrics:
- mAP@0.5 and mAP@0.5:0.95
- Precision and Recall curves
- Confusion matrix
- Per-class performance

---

## 🎨 Two-Stage Detection (Advanced)

Reduce false positives by first detecting road surfaces, then finding potholes only on roads.

**Step 1: Download Road Segmentation Model**

```bash
python scripts/download_road_model.py
```

**Step 2: Test on Your Data**

```bash
python scripts/test_road_segmentation.py
```

Check the output in `output/road_seg_test/` to verify the road segmentation works well.

**Step 3: Use Two-Stage Detector**

```python
from app.two_stage_detection import create_two_stage_detector

detector = create_two_stage_detector(
    pothole_model_path="model/best.pt",
    road_model_path="model/road_seg.pt"
)

results = detector.detect_potholes(image, conf=0.35)
annotated = detector.visualize(image, results, show_mask=True)
```

---

## 🛠️ Technology Stack

**Core Frameworks:**
- **[Ultralytics YOLOv11/v8](https://github.com/ultralytics/ultralytics)** - Object detection models
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision operations

**GUI & Utilities:**
- **Tkinter** - Desktop GUI
- **Pillow (PIL)** - Image processing
- **NumPy** - Numerical operations

**Data Processing:**
- **lxml** - XML annotation parsing
- **tqdm** - Progress bars

---

## 📊 Model Performance

Typical performance metrics on pothole detection datasets:

| Model | Size | Speed (ms) | mAP@0.5 | Precision | Recall |
|-------|------|------------|---------|-----------|--------|
| YOLOv11n | 2.6MB | 2-3 | ~85% | ~82% | ~78% |
| YOLOv11s | 9.4MB | 4-6 | ~90% | ~88% | ~85% |
| YOLOv11m | 20MB | 12-15 | ~93% | ~91% | ~89% |

*Benchmarked on NVIDIA GTX 1660 Ti. Actual performance varies by dataset quality.*

---

## 📝 Dataset Format

The application supports two annotation formats:

**VOC XML Format** (for raw data):
```xml
<annotation>
  <object>
    <name>pothole</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

**YOLO Format** (auto-converted):
```
0 0.425 0.512 0.156 0.178
```
Format: `class_id center_x center_y width height` (all normalized 0-1)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Mobile/edge deployment optimization
- Multi-class detection (cracks, road damage types)
- Web-based interface
- API for integration with existing systems

---

## 📄 License

**Copyright © 2026. All Rights Reserved.**

This software is proprietary and confidential. Unauthorized copying, transfer, modification, distribution, or use of this software, via any medium, is strictly prohibited without prior written permission from the copyright holder.

For licensing inquiries, please contact the project maintainer.

See the [LICENSE](LICENSE) file for complete terms and conditions.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com/)** for the excellent YOLO implementation
- Pothole datasets from kaggle and roboflow.
- Open-source computer vision community

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation files
- Review closed issues for solutions

---

## 🗺️ Roadmap

- [ ] Real-time webcam/stream detection
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment with REST API
- [ ] Multi-language support
- [ ] Integration with GIS systems for geolocation
- [ ] Automated severity assessment
- [ ] Dashboard for maintenance tracking

---

**Made with ❤️ for safer roads**

- **Model**: YOLOv11 (ultralytics)
- **Input**: Images (any standard format: JPG, PNG, BMP, etc.)
- **Output**: Bounding boxes with confidence scores
- **Classes**: Single class - "pothole"
- **Confidence Range**: 0.10 - 0.90 (adjustable in GUI)

## Configuration

### Confidence Threshold
- Controls detection sensitivity
- **Lower values** (0.10-0.35): More detections, higher false positives
- **Higher values** (0.60-0.90): Fewer detections, higher confidence
- **Recommended**: Start at 0.35-0.50

### Road ROI Mask
- Use the ROI mask to ignore roadside areas (trees, bushes) in both GUI and video runs
- Ratios are 0..1 and define a rectangle: left/right/top/bottom
- If you still see false positives, add more negative examples (roadsides with no potholes) to training data

### Custom Model

To use a different trained model:

1. Click **"Select Model .pt"** in the GUI
2. Or modify code to load from a specific path:
   ```python
   load_model("/path/to/your/model.pt")
   ```

## Output Files

Detection results are saved in `app/output/` with naming format:
```
pred_YYYYMMDD_HHMMSS_imagename.jpg
```

Each output includes:
- Original image with bounding boxes
- Detection statistics logged to `pothole_detection.log`

## Logging

Logs are saved to `pothole_detection.log` with:
- Timestamp
- Log level (INFO, ERROR, DEBUG)
- Detailed operation information

Example log output:
```
2025-02-03 10:30:45,123 - utils - INFO - Running detection on: test_image.jpg
2025-02-03 10:30:46,456 - utils - INFO - Detection complete: 3 potholes found (avg confidence: 0.87)
```

## Performance Metrics

Typical performance on standard hardware:
- **Inference Time**: 50-200ms per image (depends on image size)
- **GPU Acceleration**: Automatically used if available (CUDA/Metal)
- **CPU Mode**: Falls back automatically if GPU unavailable

## Troubleshooting

### "Model not found" Error
- Ensure `model/best.pt` exists
- Use "Select Model .pt" button to load a trained model
- Check file permissions

### "Failed to read image" Error
- Ensure image is in valid format (JPG, PNG, BMP)
- Try with a different image
- Check file is not corrupted

### Slow Detection
- Try GPU mode (requires CUDA/Metal)
- Reduce image resolution before uploading
- Check system resources

### Import Errors
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`
- Consider using a virtual environment

## Future Enhancements (for car integration)

- [ ] Real-time video stream processing from car camera
- [ ] GPS coordinate logging of detected potholes
- [ ] Driver alert system (audio/visual warnings)
- [ ] Pothole severity classification
- [ ] Network transmission to cloud database
- [ ] Map integration for pothole location tracking

## Requirements

- Python 3.8+
- 4GB RAM (8GB+ recommended)
- GPU optional but recommended (NVIDIA CUDA 11.8+ for acceleration)
- 500MB+ disk space for model

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For issues or improvements, please refer to the project documentation or logs for debugging information.
