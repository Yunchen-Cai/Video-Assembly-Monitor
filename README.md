# Video Assembly Monitor: Dataset & Model Pipeline

Welcome to the data and model development module of the **Video Assembly Monitor** project!  
This section focuses on enabling accurate **hand tracking** and **object detection**, forming the foundational layer of assembly stage recognition and real-time verification.

This work is part of the larger system **Hand Tracking and Object Detection for Assembly Stage Verification and Improvement**, jointly developed with backend, frontend, and vector analysis modules.

---

## 📦 Project Overview

This module is responsible for:

- Capturing and annotating multi-perspective assembly videos
- Defining and segmenting object classes and action sequences
- Training and evaluating object detection models (YOLOv11/YOLOv12)
- Providing hand-annotated ground-truth for gesture and action recognition

---

## 🎥 Dataset Construction

### 1. 📹 Video Recording

Two camera views were used to simulate realistic assembly processes:

- **Main View**: Frontal angle capturing full-hand motion
- **Side View**: Profile perspective ensuring spatial validation

> All recordings follow standard environment setup: consistent indoor lighting, neutral background, standard uniform.

### 2. ✏️ Annotation Process

#### 🟠 Roboflow

- Extract key frames from video for manual annotation
- Define object bounding boxes and labels
- Use auto-labeling to expand and validate dataset
- Export in YOLO format

#### 🔵 VIA (VGG Image Annotator)

- Annotate video keyframes for gesture segmentation
- Define action intervals and batch-label steps
- Export JSON annotations for vector machine use

---

## 🧩 Assembly Semantics Definition

### 1. 📦 Object & Action Classes

Defined components include hands, base, screws, lid, tools, etc.

### 2. 🔁 Assembly Sequence & Preset

Each assembly task is pre-defined in a **Preset Assembly Plan**. These are used in:

- Vector sequence validation
- Gesture classification
- Task step matching

### 3. ⏱️ Standard Time Definition

Standard operating times (SOT) are defined for each step to:

- Monitor delay or timeout
- Enable feedback for performance tuning

📄 [Click here to view the full demo document (assembly sequence, definitions, timing)](your-demo-link-here)

---

## ⚙️ Roboflow Configuration Guide

### 1. 🧹 Preprocessing

- **Auto-Orient**: Remove EXIF rotation artifacts
- **Resize**: Normalize image dimensions

### 2. 🔄 Data Augmentation

| Technique        | Description                                      |
|------------------|--------------------------------------------------|
| Flip             | Random horizontal/vertical flip                  |
| Rotation         | ±15° variation for viewpoint robustness          |
| Grayscale        | Applied to 15% of training images                |
| Hue Adjustment   | Random ±18° color shift                          |
| Gaussian Blur    | Up to 2.6px for motion/focus noise resistance    |

---

## 🧠 Model Training

### YOLOv11 / YOLOv12

- **Pre-trained** on MS COCO v28 (47.0% mAP)
- **Transfer Learning** from Roboflow Universe
- **Backbone**: Efficient, real-time object detection

---

## 📊 Model Evaluation

### 1. Confusion Matrix

- Diagonal: Correct class predictions  
- Off-diagonal: Misclassification indicators  
- Use for quality assessment and mislabeling detection

### 2. Vector Analysis

- Project F1 scores of samples into 2D semantic space  
- Reveal clusters and anomalies in action recognition  
- Helps align with vector-based sequence validation

---

## 🔄 EOID Integration & Deployment

This module integrates into the EOID (Event-Oriented Intelligent Detection) engine.

### 📂 Deployment Steps

- Adapt data format for EOID  
- Migrate and map dataset fields  
- Ensure evaluation logic and result exports align with backend logic

### 🧾 Key Files

- `transfer.py`: Handles Roboflow-to-EOID conversion  
- `app.py`: Client for interacting with EOID API endpoints  

---

## 🤖 YOLO Inference Sample

To test model predictions locally:

```bash
python yolov5/detect.py --weights path/to/best.pt --img 640 --conf 0.4 --source your_image_or_video.mp4
>>>>>>> 20a7ade521077e019b83fba7f9826cc36c85c885
