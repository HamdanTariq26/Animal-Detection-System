# 🐾 Animal Detection & Tracking — Mixture of Experts

<p align="center">
  <strong>Dynamic Per-Frame Expert Routing for Robust Wildlife Video Detection</strong>
</p>

<p align="center">
  <a href="https://hamdantariq26.github.io/projects/animal-detection">
    <img src="https://img.shields.io/badge/Portfolio-Project%20Page-111827?style=for-the-badge" alt="Portfolio">
  </a>
  <img src="https://img.shields.io/badge/Computer%20Vision-Applied%20Deep%20Learning-2563EB?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/RT--DETR--X-Precision%20Expert-111827?style=for-the-badge" alt="RT-DETR-X">
  <img src="https://img.shields.io/badge/YOLO11m-Kinetic%20Expert-111827?style=for-the-badge" alt="YOLO11m">
  <img src="https://img.shields.io/badge/Mixture%20of%20Experts-MoE-7C3AED?style=for-the-badge" alt="MoE">
  <img src="https://img.shields.io/badge/BoT--SORT-Tracking-059669?style=for-the-badge" alt="BoT-SORT">
  <img src="https://img.shields.io/badge/Python-Gradio-F97316?style=for-the-badge" alt="Python">
</p>

---

## Overview

Wildlife video captured in natural habitats can vary dramatically from frame to frame. Clear and well-lit footage may contain rich visual detail, while fast animal movement, motion blur, foliage, mist, low contrast, and other degradations can make detection significantly more difficult.

This project addresses that problem with a **Mixture-of-Experts (MoE) object detection architecture**.

Instead of forcing a single detector to handle every visual condition, the system contains two specialized experts:

* **Precision Expert — RT-DETR-X:** optimized for clear, sharp frames where detailed feature extraction and precise localization are important.
* **Kinetic Expert — YOLO11m:** trained on degraded visual conditions to improve robustness when frames contain blur, noise, occlusion, or other difficult conditions.

A lightweight **Laplacian variance router** evaluates every incoming frame and dynamically chooses which expert should process it. The resulting detections are passed to **BoT-SORT** for persistent multi-object tracking.

```text
                    Video Stream
                         │
                         ▼
              ┌─────────────────────┐
              │ Laplacian Variance   │
              │ Frame Quality Router │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
        Sharp / Clear          Blur / Degraded
              │                     │
              ▼                     ▼
       RT-DETR-X Expert        YOLO11m Expert
       Precision Expert         Kinetic Expert
              │                     │
              └──────────┬──────────┘
                         ▼
                  BoT-SORT Tracking
                         │
                         ▼
                Annotated Video Output
                         │
                         ▼
                    Gradio Demo
```

---

## 🎯 Motivation

A single model is expected to perform well across fundamentally different visual regimes in conventional wildlife detection systems.

A sharp animal standing in open terrain provides strong edges, textures, and fine-grained appearance cues. In contrast, rapid movement, poor lighting, vegetation, mist, or occlusion can remove much of that information.

Training one detector to be equally optimal in both regimes creates a difficult trade-off.

The MoE approach instead specializes the models:

> **Use the precision-oriented expert when the visual information is clean, and switch to a robustness-oriented expert when the frame quality deteriorates.**

The router itself is deliberately lightweight and does not require a third neural network.

---

## 🧠 Mixture-of-Experts Architecture

### Precision Expert

**RT-DETR-X**

The precision expert is trained on clean wildlife imagery and is intended for favorable visual conditions.

It specializes in:

* Fine-grained species discrimination
* Precise bounding-box regression
* Clear and high-frequency visual features
* High-quality frames with good visibility

### Kinetic Expert

**YOLO11m**

The kinetic expert is trained using a degraded version of the dataset designed to simulate challenging field conditions.

The degradation pipeline includes:

* Motion blur
* Noise injection
* Image compression
* Other challenging visual conditions

This expert is intended to remain effective when visual quality decreases.

---

## 🔀 Frame-Level Routing

The router uses **Laplacian variance**, a classical image-sharpness measure.

For every incoming frame:

```python
score = cv2.Laplacian(
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    cv2.CV_64F
).var()
```

The score is compared against a configurable threshold:

```text
                 Laplacian Variance
                         │
             ┌───────────┴───────────┐
             │                       │
        score ≥ τ                score < τ
             │                       │
             ▼                       ▼
       Precision Expert         Kinetic Expert
         RT-DETR-X                 YOLO11m
```

This provides:

* Deterministic routing
* Very low computational overhead
* An interpretable decision rule
* No additional neural gating network

The threshold can be adjusted directly from the interactive application.

---

## 🎥 Persistent Wildlife Tracking

Detection is combined with **BoT-SORT** to maintain animal identities across video frames.

The tracking layer provides:

* Persistent track IDs
* Multi-object tracking
* Motion prediction
* Track continuity across frames
* Better identity stability when the selected expert changes

```python
results = active_model.track(
    frame,
    persist=True,
    tracker="botsort.yaml",
    conf=0.25,
    verbose=False
)[0]
```

This is particularly important for the MoE architecture because the detector can change between consecutive frames while the tracking system continues maintaining object identities.

---

## 📦 Dataset Curation

The original dataset contains **53 wildlife categories**.

Dataset inspection identified severe class imbalance associated with the **Butterfly** category. The project therefore removes this category and continuously re-indexes the remaining annotations.

```text
53 Classes
     │
     ▼
Remove Butterfly
     │
     ▼
Re-index labels
     │
     ▼
52 Wildlife Classes
```

This preprocessing is applied consistently before training the expert models.

---

## 🔬 Training Pipeline

### Phase 1 — Dataset Curation

The raw wildlife dataset is inspected and transformed into a cleaner training set by removing the heavily over-represented Butterfly category and re-indexing the remaining labels.

### Phase 2 — Precision Expert

The clean dataset is used to train the high-capacity detection model:

```text
Clean Wildlife Images
        │
        ▼
   RT-DETR-X
        │
        ▼
Precision Expert
```

### Phase 3 — Kinetic Expert

A degraded version of the dataset is generated using image transformations such as blur, noise, and compression.

```text
Clean Dataset
      │
      ▼
Synthetic Degradation
      │
      ├── Motion Blur
      ├── Noise
      └── Compression
      │
      ▼
YOLO11m
      │
      ▼
Kinetic Expert
```

### Phase 4 — MoE Router

The frame-level Laplacian variance estimator selects the appropriate expert during inference.

### Phase 5 — BoT-SORT

Detector outputs are passed to the tracking layer to maintain persistent animal identities.

### Phase 6 — Deployment

The complete pipeline is exposed through an interactive **Gradio** application.

---

## 🖥️ Interactive Demo

The application provides three operating modes:

| Mode               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| **MoE (Auto)**     | Automatically selects the expert using frame quality |
| **Precision Only** | Uses the precision expert for every frame            |
| **Kinetic Only**   | Uses the kinetic expert for every frame              |

The interface also exposes an adjustable quality threshold and displays the selected expert alongside the resulting tracking output.

---

## 🏗️ System Pipeline

```text
┌────────────────────┐
│   Wildlife Video   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Frame Quality      │
│ Estimation         │
│ Laplacian Variance │
└─────────┬──────────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌─────────┐ ┌─────────┐
│RT-DETR-X│ │ YOLO11m │
│Precision│ │ Kinetic │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
     ┌────────────┐
     │  BoT-SORT  │
     │  Tracking  │
     └─────┬──────┘
           ▼
     ┌────────────┐
     │   Gradio   │
     │ Application│
     └────────────┘
```

---

## 📊 Evaluation

The project evaluates the two experts independently using standard object-detection metrics:

* **mAP50**
* **Precision**
* **Recall**
* **F1-score**

Confusion matrices are also generated to inspect class-level behavior.

The evaluation stage is designed to compare the strengths of the two specialists rather than assuming that one detector should dominate across all visual conditions.

---

## 🛠️ Technologies

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Ultralytics-111111">
  <img src="https://img.shields.io/badge/RT--DETR-111111">
  <img src="https://img.shields.io/badge/YOLO11m-111111">
  <img src="https://img.shields.io/badge/BoT--SORT-6C5CE7">
  <img src="https://img.shields.io/badge/Albumentations-FF6F61">
  <img src="https://img.shields.io/badge/Gradio-F97316">
</p>

---

## 📁 Repository Structure

```text
Wildlife-MoE/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
│
├── scripts/
│   ├── __init__.py
│   └── prepare_dataset.py
│
├── inference/
│   ├── __init__.py
│   └── app.py
│
├── models/
│   ├── precision_expert/
│   │   └── best.pt
│   │
│   └── kinetic_expert/
│       └── best.pt
│
└── runs/
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare and train

```bash
python -m training.train
```

The training pipeline prepares the datasets, trains both specialists, and places their best checkpoints into:

```text
models/precision_expert/best.pt
models/kinetic_expert/best.pt
```

### 3. Evaluate

```bash
python -m training.evaluate
```

### 4. Launch inference

```bash
python -m inference.app
```

The Gradio application will load both experts and provide the automatic MoE routing interface.

---

## ⚙️ Model Checkpoint Organization

Training outputs are separated from inference models.

```text
Ultralytics Training
        │
        ├── Precision Expert
        │       │
        │       ▼
        │   best.pt
        │       │
        │       ▼
        │ models/precision_expert/
        │
        └── Kinetic Expert
                │
                ▼
             best.pt
                │
                ▼
          models/kinetic_expert/
```

This keeps inference independent from automatically generated experiment-directory names.

---

## 🔒 GitHub Notes

The repository is intended to contain:

* Dataset preparation code
* Training code
* Evaluation code
* Inference code
* Configuration
* Documentation

---

## 🌱 Key Engineering Ideas

### Specialized Models Instead of One Universal Model

The system intentionally separates **precision** and **robustness** into different experts rather than forcing a single detector to optimize conflicting objectives.

### Classical Computer Vision as a Lightweight Router

The routing mechanism does not require another neural model. Laplacian variance provides a fast and interpretable way to estimate frame sharpness.

### Tracking Across Expert Switching

BoT-SORT maintains object identities even while the underlying detector can switch between experts from one frame to another.

### Designed Around Real-World Video

The architecture is motivated by the heterogeneous conditions found in real wildlife footage rather than assuming every frame has equivalent visual quality.

---

## 📌 Project Information

**Domain:** Computer Vision & Applied Deep Learning
**Year:** 2026
**Type:** Deployed Experiment

### Technologies

`Computer Vision` `RT-DETR-X` `YOLO11m` `Mixture of Experts` `Laplacian Variance` `BoT-SORT` `Python` `Gradio`

---

## 👤 Author

### Hamdan Tariq

[Portfolio](https://hamdantariq26.github.io/) · [Project Page](https://hamdantariq26.github.io/projects/animal-detection)

---

<p align="center">
  <i>Adaptive perception for challenging wildlife environments.</i>
</p>
