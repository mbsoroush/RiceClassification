# 🍚 Rice Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green)

---

## 🔰 Overview

This project focuses on **Rice Classification** using Deep Learning and Computer Vision. It explores CNN architectures, compares deep learning frameworks, and evaluates transfer learning approaches using pretrained models.

The dataset contains **75,000 rice grain images** across five different varieties:

- 🌾 Arborio
- 🌾 Basmati
- 🌾 Ipsala
- 🌾 Jasmine
- 🌾 Karacadag

The project consists of two main notebooks:

1. **Rice Classification | TensorFlow vs. PyTorch** → Building and comparing CNN models using TensorFlow/Keras and PyTorch.
2. **Rice Classification | Transfer Learning** → Applying transfer learning with different pretrained CNN and Vision Transformer architectures.

---

## 📦 Dataset

The project uses a rice image dataset containing **75,000 images** distributed equally across five rice varieties.

- **Total Images:** 75,000
- **Number of Classes:** 5
- **Images per Class:** 15,000

### Dataset Source

👉 [Rice Classification Dataset – Kaggle](https://www.kaggle.com/datasets/mbsoroush/rice-images-dataset)

---

## 🧠 Project 1: Rice Classification | TensorFlow vs. PyTorch

This notebook explores CNN implementation and performance comparison using two popular deep learning frameworks: **TensorFlow/Keras and PyTorch**.

### ⚙️ Implementation

1. ✅ **CNN Architecture Development**
   - Built two CNN architectures: a baseline model and a regularized model.
   - Implemented both architectures independently using TensorFlow/Keras and PyTorch.

2. ✅ **Controlled Training**
   - Used the same data split, architecture, optimizer, hyperparameters, and early-stopping strategy across frameworks.

3. ✅ **Model Evaluation**
   - Training and validation loss/accuracy curves.
   - Confusion matrices.
   - ROC curves and AUC.
   - Precision-Recall analysis.

4. ✅ **Framework Comparison**
   - Training speed and memory consumption.
   - Input pipeline performance.
   - Inference latency and throughput.
   - CPU inference and cold-start performance.
   - Statistical significance testing.

5. ✅ **Final Comparison**
   - Compared model accuracy, training behavior, and computational performance.
   - Summarized the results in a final comparison scorecard.

---

## 🚀 Project 2: Rice Classification | Transfer Learning

This notebook explores **Transfer Learning** by evaluating seven ImageNet-pretrained architectures for rice variety classification.

### 🏗️ Pretrained Architectures

Seven different backbones from various architecture families were evaluated:

| Architecture Family | Pretrained Models |
|---|---|
| Classic CNNs | ResNet-50, DenseNet-121 |
| Efficient CNNs | EfficientNet-B0, MobileNetV3-Large |
| Modern CNNs | ConvNeXt-Tiny |
| Vision Transformers | Swin-Tiny, DeiT-Small |

### ⚙️ Implementation

1. ✅ **Transfer Learning**
   - Linear probing with frozen pretrained backbones.
   - Full fine-tuning with differential learning rates.

2. ✅ **Model Evaluation**
   - Accuracy and balanced accuracy.
   - Precision, Recall, and F1-score.
   - ROC-AUC and confusion matrices.
   - Model calibration and reliability analysis.

3. ✅ **Performance Benchmarking**
   - Training time and convergence behavior.
   - Inference latency and throughput.
   - Model size and computational complexity.
   - Memory consumption.

4. ✅ **Additional Experiments**
   - Statistical comparison between models.
   - Data-efficiency analysis.
   - Robustness testing under different image conditions.
   - Grad-CAM visualizations for model explainability.

5. ✅ **Final Comparison**
   - Compared all seven pretrained architectures.
   - Analyzed trade-offs between accuracy, efficiency, robustness, and model complexity.
   - Presented the results through comparison charts and a final scorecard.

---

## 📊 Technologies & Libraries

The following technologies and libraries were used throughout the project:

- **Programming:** Python
- **Deep Learning Frameworks:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Model Architectures:** CNNs, ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt, Swin Transformer, DeiT

---

## 📁 Repository Structure

```text
Rice-Classification/
│
├── Rice Classification - TensorFlow vs PyTorch.ipynb
│
├── Rice Classification - Transfer Learning.ipynb
│
└── README.md
```

---

## 🚀 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/mbsoroush/RiceClassification.git
   ```

2. Navigate to the project directory:

   ```bash
   cd RiceClassification
   ```

3. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn
   pip install tensorflow torch torchvision opencv-python scikit-learn
   ```

4. Download the dataset from Kaggle and configure its path in the notebooks.

5. Open either notebook using Jupyter Notebook or Kaggle Notebooks and run the cells.

---

## 🏁 Conclusion

This project provides a comparative study of deep learning approaches for rice variety classification.

1. ✅ **Framework Comparison**
   - Explored CNN implementation using TensorFlow/Keras and PyTorch.

2. ✅ **Transfer Learning**
   - Evaluated seven pretrained architectures, including CNNs and Vision Transformers.

3. ✅ **Comprehensive Evaluation**
   - Investigated model accuracy, computational efficiency, robustness, and reliability.

4. ✅ **Practical Experience**
   - Gained hands-on experience in CNN development, transfer learning, model benchmarking, and computer vision.

