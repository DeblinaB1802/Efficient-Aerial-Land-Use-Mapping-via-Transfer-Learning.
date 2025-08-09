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
  - Images resized to 224×224 (ImageNet input size).
  - Normalized with ImageNet statistics:
      - Mean = [0.485, 0.456, 0.406]
      - Std = [0.229, 0.224, 0.225]
  - Train/Test Split: 70% / 30%

---

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

| Model        | Accuracy (%) | Latency (ms/sample) | Parameters | FLOPs         |
|--------------|--------------|--------------------|------------|---------------|
| resnet50     | 92.06        | 133.02              | 23.60 M    | 4.13 GMac     |
| densenet121  | 88.73        | 193.55              | 7.06 M     | 2.90 GMac     |
| mobilenet_v2 | 90.32        | 42.64               | 2.28 M     | 325.37 MMac   |


### 🚀 Post-Training Quantization on MobileNetV2
| Model                    | Accuracy (%) | Latency (ms/sample) | Parameters | FLOPs       |
|--------------------------|--------------|---------------------|------------|-------------|
| mobilenet_v2              | 90.32        | 50.39               | 2.28 M     | 325.37 MMac |
| mobilenet_v2_quantized    | 89.12        | 24.47               | 2.21 M     | 62.72 KMac  |

- **Latency Reduction**: 51.44% faster inference.  
- **Accuracy Drop**: ~1.2% (90.32% → 89.12%).  
- **Final Latency**: ~24.47 ms/image on CPU.  

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
