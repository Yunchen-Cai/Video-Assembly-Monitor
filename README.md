# Video Assembly Monitor: Dual-Model System with YOLO and EOID for Real-Time and Zero-Shot Assembly Monitoring

Welcome to the **Video Assembly Monitor** project! This repository provides the core model pipeline for accurate **hand tracking**, **object detection**, and **action recognition** in manufacturing assembly processes, such as rice cooker assembly. It combines a fast real-time model (YOLO) for online processing and a slow zero-shot model (EOID, a Vision-Language Model) for offline semantic analysis.

This work is part of the larger system **Hand Tracking and Object Detection for Assembly Stage Verification and Improvement**, developed in collaboration with frontend and backend modules for a complete end-to-end solution.

For the full system:
- **Frontend**: https://github.com/LtSeed/VideoAssemblyMonitorWeb (Built with Vite, React, TypeScript, and Ant Design for responsive UI and real-time feedback)
- **Backend**: https://github.com/LtSeed/VideoAssemblyMonitor (Built with Spring Boot for API handling and integration)

The custom dataset used in this project is available at: https://github.com/Yunchen-Cai/RiceCooker-Assembly-Dataset

EOID model integration is based on: https://github.com/mrwu-mac/EoID (AAAI 2023 paper on End-to-End Zero-Shot HOI Detection via Vision and Language Knowledge Distillation)

---

## 📦 Project Overview

This module focuses on the AI-driven foundation for assembly monitoring:
- **Capturing and annotating multi-perspective videos** from assembly tasks.
- **Defining object classes, action sequences, and standard operating times**.
- **Training and evaluating dual models**: YOLOv11 for fast, real-time detection (online processing) and EOID for slow, zero-shot semantic classification (offline analysis).
- **Providing ground-truth annotations** for gesture recognition, sequence validation, and ergonomic analysis.

The system verifies assembly correctness and efficiency by combining:
- **YOLO (Fast Model)**: High-speed frame-level predictions for known objects and actions, enabling responsive UI feedback.
- **EOID (Slow VLM Model)**: Zero-shot classification using CLIP-based vision-language alignment for unseen gestures or tools, enhancing generalization.

Experimental results: YOLOv11 achieved mAP over 83% with real-time performance; EOID provided broader coverage for ambiguous actions.

This architecture supports intelligent quality assurance in human-in-the-loop manufacturing, with potential for hybrid reasoning and domain adaptation.

---

## 🎥 Dataset Construction

A custom multi-view, multi-task dataset was built for rice cooker assembly, focusing on hand-object interactions.

### 1. 📹 Video Recording

Four camera angles were used for comprehensive coverage:
- **Front View**: Overall assembly actions and hand movements (using PowerConf C300 camera).
- **Operator's View**: Workbench, parts, and steps (mounted on operator).
- **Left-Side View**: Detailed hand-screw interactions (mobile phone camera).
- **Right-Side View**: Additional hand movement details.

Software: OBS Studio for simultaneous multi-camera recording. Environment: Even lighting to avoid shadows.

The full dataset, including videos, images, and annotations, is hosted at: https://github.com/Yunchen-Cai/RiceCooker-Assembly-Dataset

### 2. ✏️ Annotation Process

#### 🟠 Roboflow
- Extracted key frames from videos for manual bounding box annotation.
- Defined labels for objects (e.g., hands, screws, base) and actions.
- Applied auto-labeling to expand the dataset.
- Exported in YOLO-compatible format.

#### 🔵 VIA (VGG Image Annotator)
- Annotated keyframes for temporal gesture segmentation.
- Defined action intervals (e.g., "install heating plate screw") and batch-labeled steps.
- Exported JSON for vector-based sequence analysis.

Challenges addressed: Insufficient specificity in existing datasets like Assembly101, leading to custom creation for tasks like screw tightening and wiring.

---

## 🧩 Assembly Semantics Definition

### 1. 📦 Object & Action Classes

Components: Hands, base, heating plate, screws, lid, tools, etc.  
Actions: Handling (e.g., Steps 1,3,5), Insertion (e.g., Steps 2,4,6), categorized for sequence validation.

### 2. 🔁 Assembly Sequence & Preset

Pre-defined preset plans for rice cooker assembly, used for:
- Vector sequence matching.
- Gesture classification.
- Real-time order verification (e.g., demo for correct/wrong orders).

### 3. ⏱️ Standard Time Definition

Standard Operating Times (SOT) per step for monitoring delays, timeouts, and performance optimization.

📄 View detailed assembly sequence, quotas, and disassembly guide in the dataset repo: https://github.com/Yunchen-Cai/RiceCooker-Assembly-Dataset

---

## ⚙️ Model Configuration and Training

### Preprocessing and Augmentation
- **Preprocessing**: Auto-orient, resize for normalization.
- **Data Augmentation**:
  | Technique      | Description                                      |
  |----------------|--------------------------------------------------|
  | Flip           | Random horizontal/vertical flip                  |
  | Rotation       | ±15° variation for viewpoint robustness         |
  | Grayscale      | Applied to 15% of training images                |
  | Hue Adjustment | Random ±18° color shift                          |
  | Gaussian Blur  | Up to 2.6px for motion/focus noise resistance   |

### YOLOv11/YOLOv12 (Fast Real-Time Model)
- Pre-trained on MS COCO (47.0% mAP).
- Transfer learning from Roboflow Universe.
- Backbone: Efficient for real-time detection.
- Training: Box, class, and object losses converged to ~0 after 30 batches.
- Speed: ~30 FPS; Stable on small datasets.

### EOID (Slow Zero-Shot VLM Model)
- Based on: https://github.com/mrwu-mac/EoID (Integrates CLIP for vision-language alignment and DETR for detection).
- Zero-shot learning: Recognizes unseen tools/actions without retraining.
- Architecture: Multi-head for object + verb + position; Distills knowledge from CLIP/DETR.
- Performance: Verb match precision ~95%; Handles HOI (Human-Object Interactions).
- Limitations: Less stable on small datasets; Needs accurate temporal annotations; Struggles with occlusions.

Hybrid Strategy: YOLO for agility, EOID for semantic depth.

---

## 📊 Model Evaluation

### YOLO Results
- Confusion Matrix: High accuracy for most classes; Confusion in similar actions (e.g., "adjust" vs. "tighten").
- Average Precision: Object ~65%, Action ~57% (short/invisible movements challenging).
- Losses: Stabilized below 0.2 for objects, under 2.0 for actions.

### EOID Results
- Verb precision ~95%; Consistent bounding box localization.
- Supports zero-shot for unseen categories; No inference applied in initial phase.

Demos: Right/wrong order videos in `assets/videos` show sequence verification.

Future: Unlock EOID zero-shot fully; Improve robustness to lighting/occlusions.

---

## 🔄 Integration & Deployment

Integrates EOID engine with YOLO for modular deployment.

### Key Files
- `transfer.py`: Roboflow-to-EOID conversion.
- `app.py`: Client for EOID API.
- YOLO scripts in `src/yolo/`; EOID in `src/eoid/`.

Deployment: Adapt formats, map fields, align with backend logic.

Full System: Combines with frontend for UI feedback and backend for API/processing.

---

## 🤖 YOLO Inference Sample

Test locally:
```bash
python src/yolo/detect.py --weights src/yolo/best.pt --img 640 --conf 0.4 --source assets/videos/your_video.mp4
```

For EOID: Follow https://github.com/mrwu-mac/EoID for setup and evaluation scripts.

---

## 🛠️ Technology Stack

- **Models**: YOLOv11/YOLOv12 (Ultralytics), EOID (PyTorch-based with CLIP and DETR).
- **Libraries**: PyTorch, OpenCV, NumPy, SciPy, Pandas (for data processing).
- **Tools**: Roboflow (annotation/export), VIA (temporal annotation), OBS Studio (recording).
- **Dependencies**: See `requirements.txt` for full list.

No internet access required for core operations; Pre-trained models downloadable.

---

## 🚀 Setup

1. **Clone Repository**:
   ```
   git clone https://github.com/Yunchen-Cai/Video-Assembly-Monitor.git
   cd Video-Assembly-Monitor
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   - YOLO: From Roboflow or MS COCO.
   - EOID: CLIP and DETR as per https://github.com/mrwu-mac/EoID.

4. **Run Training/Evaluation**:
   - YOLO: Use scripts in `src/yolo/`.
   - EOID: Follow `train.sh` and `test.sh` from EOID repo.

5. **Integrate with Full System**:
   - Setup backend and frontend from linked repos.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Supervisor**: Prof. Ong Soh Khim for guidance and feedback.
- **Examiner**: Prof. Andrew Nee Yeh Ching for insightful suggestions.
- **Teammate**: Bing Hong Liu for contributions in model selection, dataset design, and architecture.
- **Libraries & Frameworks**: YOLO (Ultralytics), EOID (mrwu-mac), CLIP (OpenAI), DETR (Facebook AI), Roboflow, VIA.
- **Datasets**: Custom RiceCooker; Inspired by Assembly101, HICO-DET, V-COCO.
- Special thanks to myself for exploring new domains and persevering.

This is part of my Bachelor of Engineering in Mechanical Engineering at NUSRI@Suzhou, Session 2024/2025.

For questions, open an issue on GitHub.
