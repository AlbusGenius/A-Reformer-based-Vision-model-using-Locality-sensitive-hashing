# Vision Reformer for Diabetic Retinopathy Classification

This repository contains a Vision Reformer model adapted from the Reformer architecture for image classification.  
We apply it to a **high-resolution medical imaging dataset** (diabetic retinopathy fundus photos) and train it with **DeepSpeed** to handle long token sequences efficiently.

---

## 1. Overview

- **Backbone:** Reformer-style Transformer with LSH attention.
- **Vision front-end:** Lightweight CNN stem + patch embedding (image → patches → tokens).
- **Task:** 5-class diabetic retinopathy (No DR, Mild, Moderate, Severe, Proliferative).
- **Efficiency:** Uses DeepSpeed (FP16 + ZeRO) and **gradient accumulation** to keep GPU memory around ~12 GB while simulating a larger effective batch size.

---

## 2. Requirements

Install the main dependencies (example for Python 3.10+):

```bash
pip install torch torchvision deepspeed kagglehub matplotlib scikit-learn
pip install reformer-pytorch
