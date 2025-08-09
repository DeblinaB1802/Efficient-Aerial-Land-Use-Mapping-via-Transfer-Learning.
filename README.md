# 🛰️ Land Use Classification with Transfer Learning & Model Optimization

This project focuses on **land-use classification** using the **UCMerced Land Use Dataset** (21 classes), applying **transfer learning** with ResNet-50, DenseNet-121, and MobileNetV2.  
The goal was to benchmark different architectures for accuracy–latency trade-offs, then optimize the best-performing model for **deployment efficiency** using **Post-Training Quantization (PTQ)**.

---

## 📌 Project Overview

- **Dataset**: [UCMerced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)  
  - 21 classes (e.g., Forest, Freeway, Buildings, River)  
  - Each class contains 100 RGB images of **256×256** resolution.  

- **Objective**:  
  1. Train and evaluate multiple CNN architectures with transfer learning.  
  2. Benchmark models for **accuracy, latency, and parameter count**.  
  3. Deploy an optimized model with **faster inference** while minimizing accuracy drop.

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Resized images to `224×224` to match ImageNet pre-trained weights.
   - Applied standard augmentations: random horizontal flips, random rotations, normalization.
   - Split data into:
     - **80% Training**
     - **10% Validation**
     - **10% Testing**

2. **Model Selection**
   - **ResNet-50** – Deep residual connections for robust feature learning.
   - **DenseNet-121** – Dense connectivity for efficient feature reuse.
   - **MobileNetV2** – Lightweight and fast architecture suitable for deployment.

3. **Training Setup**
   - Transfer learning with ImageNet pre-trained weights.
   - Fine-tuned last layers for 21-class classification.
   - Used **Cross-Entropy Loss** and **Adam optimizer**.
   - Learning rate scheduling with `ReduceLROnPlateau`.
   - Early stopping to prevent overfitting.

4. **Benchmarking**
   - **Accuracy** – Classification accuracy on test set.
   - **Latency** – Average inference time per image (CPU).
   - **Parameters** – Total trainable parameters.

5. **Optimization**
   - Applied **Post-Training Quantization (PTQ)** on the best model.
   - Reduced latency by converting weights from FP32 to INT8.
   - Measured accuracy drop and performance gain.

---

## 📊 Results

| Model         | Accuracy (%) | Latency (ms) | Params (M) |
|---------------|-------------|--------------|------------|
| ResNet-50     | 93.2        | 152.6        | 25.6       |
| DenseNet-121  | 92.5        | 184.7        | 7.98       |
| **MobileNetV2** | **91.1**    | **54.2**     | **2.25**   |

### 🚀 Post-Training Quantization on MobileNetV2
- **Latency Reduction**: 48.4% faster inference.  
- **Accuracy Drop**: ~1.2% (91.1% → 90.9%).  
- **Final Latency**: ~28 ms/image on CPU.  

---

## 🔍 Key Insights

- **MobileNetV2** offered the best **accuracy–latency trade-off** with only **2.25M parameters**, making it ideal for edge deployment.  
- **PTQ optimization** significantly reduced latency with negligible accuracy loss, proving effective for **real-time inference scenarios**.
- ResNet-50 had the highest accuracy but was **~3× slower** than MobileNetV2.

---

## 📈 Visualization
- Loss vs Epoch plot visualization.
- Accuracy, precision, recall, F1 score - classification prediction performance.
- Confusion Matrix – Class-wise prediction performance.
- Latency Plots – Pre- and post-quantization inference speeds.
