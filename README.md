# Metal & Non-Metal Particle Classification and Segmentation

**Course:** Industrial Data Science 2 (M.Sc. Data Science, TU Dortmund University, Apr–Aug 2024)  
**Industry Partner:** [PartikelART Solution GmbH](https://www.partikel-art.de/) (TU Dortmund / IPS spin-off)  
**Team (Group 3):** Raj Anilbhai Pawar, Harsh Yadav, Ananya Pal, Saptarsi Bhattacharya, Purvanshi Sharma

---

## Overview

In automotive, battery, and electronics manufacturing, microscopic metallic chips and non-metallic contaminants left on components can cause short circuits or premature wear. Standard technical cleanliness inspection depends on lab microscopes and slow manual review.

Partnering with **PartikelART Solution GmbH**, our 5-student team built and benchmarked deep learning models for classifying and segmenting metallic (`met`) vs. non-metallic (`nichtmet`) particles across two very different image domains:
- **53,880** high-resolution microscope crops (**19,206** metal, **34,674** non-metal)
- **660** low-resolution smartphone captures (**288** metal, **372** non-metal)
- **1,024** manually annotated images (`640×640`) labeled with bounding boxes and masks in `makesense.ai` (70% train / 15% val / 15% test) for detection and segmentation

> **From classroom PoC to production:** During the final presentation of this project, PartikelART co-founder Dr.-Ing. Ronny Zwinkau sat in the audience. Based on these results and the next-step proposals presented, I joined PartikelART in October 2024 as a Machine Learning & Software Engineer, where we scaled particle inspection under VDA 19.1 / ISO 16232 and won **1st Place at the TU Start-up Award 2025**. (Production company code and datasets are proprietary and not part of this university repository.)

---

## My Contributions

Within Group 3, my primary work covered:
- **Custom Keras U-Net Segmentation (`train/unet/`):** Built the YOLO-bounding-box-to-mask conversion pipeline (`yolo-annotations-to-mask.ipynb`) and iterated through five `640×640` U-Net training variants (`Raj-U-Net*.ipynb`) with L2 regularization, synchronized image/mask augmentation, `ReduceLROnPlateau`, and `EarlyStopping`.
- **Mobile Classifier & TFLite Export (`train/cnn/cnn_detect_mobile.ipynb`):** Built the two-stage frozen/fine-tuned MobileNetV2 transfer-learning pipeline for low-resolution mobile images and exported/validated `.tflite` inference.
- **Interactive Gradio Prototype (`app.py`, `src/`):** Built the 3-tab Gradio interface combining YOLOv8 detection, OpenCV contour + CNN classification (`src/ClassifyCnn.py`), and a 17-transform augmentation utility (`src/ImageProcessor.py`).
- **Evaluation Report:** Authored the individual technical evaluation report comparing architectures across microscope and mobile image domains.

---

## Benchmark Results

| Model | Data Source | Train Acc | Val Acc / mAP@0.5 | Val Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Custom CNN (128×128 Grayscale)** | Microscope (53,880 crops) | **97%** | **97% (AP 0.97 / 1.00)** | **0.20** |
| MobileNetV2 (Two-Stage Fine-Tune) | Mobile Phone (660 images) | 93% | 88% | 0.30 |
| YOLOv8n (Object Detection) | Microscope (100 epochs) | 75% | 75% · 0.756 mAP@0.5 | 3.32 |
| YOLOv8n (Instance Segmentation) | Microscope (100 epochs) | 71% | 71% · 0.717 Mask mAP | 1.61 |
| YOLOv8n (Object Detection) | Mobile Phone (200 epochs) | 66%* | 0.820 mAP@0.5 (peak ep. 144) | 1.41 |
| Custom U-Net (Box-Derived Masks) | Mobile Phone (50 epochs) | 49–55% | 32–92% (unstable) | 0.28 |

*\*Mobile YOLOv8n peaked at `0.825 mAP@0.5` around epoch 144 before overfitting by epoch 200 (`0.665 mAP@0.5`), highlighting the 660-image data bottleneck.*

---

## Visual Outputs & Diagnostics

### Microscope Instance Segmentation (YOLOv8n-Seg)
![YOLOv8n Instance Segmentation on Microscope Images](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/Yolo%20microscope%20train%20segment/train3/val_batch0_pred.jpg)

### Mobile Phone Detections (YOLOv8n)
![YOLOv8n Validation Predictions on Mobile Phone Images](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/Yolo%20mobile%20train/train11-yolov8n-200epocs-mobile/val_batch2_pred.jpg)

### Microscope Bounding-Box Detection (YOLOv8n)
![YOLOv8n Object Detection on Microscope Images](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/Yolo%20microscope%20train%20detect/results%20yolo%20v8%20high%20res/val_batch2_pred.jpg)

### Precision-Recall Curves (Custom CNN Microscope & YOLOv8n Mobile)
| Custom CNN (Microscope, AP 0.97 / 1.00) | YOLOv8n (Mobile, 0.820 mAP@0.5) |
| :---: | :---: |
| ![CNN Precision-Recall Curve](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/CNN%20microscope%20results/Precision_Recall_Curve_Microscope_HD.png) | ![YOLOv8n Mobile Precision-Recall Curve](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/Yolo%20mobile%20train/train11-yolov8n-200epocs-mobile/Precision_Recall_Curve_for_YOLOv8_on_Mobile_Images.png) |

### U-Net Training Diagnostics
![U-Net Training and Validation Loss and Accuracy](https://raw.githubusercontent.com/r-a-j/Particle-Classification-and-Segmentation/main/results/U-net/training_validation_loss_accuracy.png)

---

## Key Takeaways

1. **Crop classification vs. full-frame detection:** The custom 3-block CNN reached 97% accuracy on pre-cropped microscope particles, while YOLOv8n sat at 0.756 mAP@0.5 because textured filter backgrounds add localization noise.
2. **Mask supervision quality matters most:** Training U-Net on masks approximated from rectangular YOLO bounding boxes produced 32%–92% validation accuracy swings despite smooth loss convergence. True polygon/pixel masks are required for reliable particle boundaries.
3. **Small-dataset overfitting:** On 660 mobile phone images, YOLOv8n peaked at epoch 144 (`0.825 mAP@0.5`) and degraded by epoch 200 (`0.665 mAP@0.5`), showing why early checkpointing and larger active-learning datasets are necessary before deployment.

---

## Repository Structure

```text
.
├── app.py                        # 3-tab Gradio UI (YOLO detect, CNN classify, augmentation)
├── src/
│   ├── ClassifyCnn.py            # OpenCV contour detection + Keras CNN classifier
│   ├── ImageProcessor.py         # Batch resizing and 17-transform augmentation utilities
│   └── ClusteringAlgorithm.py    # Unsupervised K-Means exploration helper
├── train/
│   ├── cnn/                      # Microscope CNN & MobileNetV2 + TFLite export notebooks
│   ├── unet/                     # YOLO-to-mask conversion and 5 custom Keras U-Net runs
│   ├── yolo/                     # YOLOv8 / YOLOv9 training and evaluation notebooks
│   └── explore-other-architectures/
├── results/                      # Validation batches, PR/F1 curves, CSV training logs
├── docs/Presentations/           # Final & milestone presentation decks (PDF & PPTX)
└── requirements.txt
```

## Running the Gradio Prototype

```bash
git clone https://github.com/r-a-j/Particle-Classification-and-Segmentation.git
cd Particle-Classification-and-Segmentation
pip install -r requirements.txt
python app.py
```

*Note: Raw industrial particle datasets remain confidential to PartikelART Solution GmbH and are excluded from this repository. Sample validation batches, logs, notebooks, and [presentation decks](docs/Presentations/IDS_2_Group3_Presentation_3.pdf) are included for full reproducibility of the methodology.*
