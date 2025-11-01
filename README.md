<<<<<<< HEAD
# EoID: A Pytorch Implementation
The official code for our paper [End-to-End Zero-Shot HOI Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2204.03541) (AAAI2023).

## Installation
Install the dependencies. The code is tested with Pytorch 1.9.0. 
```
bash prepare.sh
```

## Data preparation

### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
data
 └─ hico_20160224_det
     |─ annotations
     |   |─ trainval_hico.json
     |   |─ test_hico.json
     |   └─ corre_hico.npy
     :
```

### V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
CDN
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.



## Pre-trained model
Download the pretrained model of CLIP for [CLIP50x16](https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt), and put it to the `ckpt` directory.

Download the pretrained model of DETR detector for [ResNet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth), and put it to the `params` directory.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2stage-q64.pth \
        --num_queries 64

python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2stage.pth \
        --dataset vcoco
```

## Training
After the preparation, you can start training with the following commands. The trainings of EoID for HICO-DET under UA setting.

### HICO-DET
```
bash train.sh
```

## Evaluation

### HICO-DET
You can conduct the evaluation with trained parameters for HICO-DET under UA setting as follows.
```
bash test.sh
```

## Results

### HICO-DET under UA setting
| |Full |Seen |Unseen |Download|
| :--- | :---: | :---: | :---: | :---: |
|EoID(R50)| 29.22 | 30.46 | 23.04 | [model](https://drive.google.com/file/d/1UrTkE0BGpCDnvmmHp9mtV2WoVtjEmHkR/view?usp=sharing) |

### HICO-DET under UC setting
| |UC_Type |Full | Seen |Unseen |Download|
| :--- | :---: | :---: | :---: | :---: | :---: |
|EoID(R50)|default |28.91 $\pm$ 0.33 |30.39 $\pm$ 0.40 |23.01 $\pm$ 1.98| model |
|EoID(R50)|rare_first |29.52 |31.39 |22.04| model |
|EoID(R50)*|rare_first |29.27 |31.72 |21.07| [model](https://drive.google.com/file/d/10lIQRZm4N92Ihsr30Vw5RlTBY-Uh893x/view?usp=sharing) |
|EoID(R50)|non_rare_first |26.69 |26.66 |26.77| model |
|EoID(R50)*|non_rare_first |25.64 |25.35 |26.80| [model](https://drive.google.com/file/d/1h4D4yYhItlrLwHz-LqvzLHkJ_7-bp2bL/view?usp=sharing) |

\* The original version of the model has been lost. This is a reimplement version on 2 RTX3090 with smaller batch size and learning rate.
## Acknowledge
This repo is based on [CDN](https://github.com/YueLiao/CDN), [CLIP](https://github.com/openai/CLIP) and [ConsNet](https://github.com/yeliudev/ConsNet).

=======
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
