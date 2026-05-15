# Handwritten Digit Recognition — MNIST

A Multilayer Perceptron (MLP) built with **PyTorch** to classify handwritten digits from the MNIST dataset.
Trained across 3 experiments varying activation functions, network size, learning rate, dropout, and batch normalization.

---

## Problem Description

The goal is to correctly classify grayscale images of handwritten digits (0–9).
Each image is 28x28 pixels, flattened into a 784-dimensional input vector and passed through a fully connected MLP.

This is a **10-class classification problem** evaluated using Accuracy and Cross-Entropy Loss.

---

## Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Source:** [PyTorch MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- **Size:** 60,000 training samples / 10,000 testing samples
- **Split used:** 54,000 train / 6,000 validation / 10,000 test

The dataset is automatically downloaded via `torchvision.datasets.MNIST`.

**Preprocessing applied:**
- Converted to tensor
- Normalized with mean=0.5 and std=0.5 — pixel values scaled to [-1, 1]

---

## Model Architecture

A flexible `FlexibleMLP` class was implemented supporting:
- Configurable hidden layers and neuron counts
- Swappable activation functions (ReLU / Sigmoid)
- Optional Dropout regularization
- Optional Batch Normalization

**General structure:**
```
Input (784) -> Hidden Layers -> Output (10 classes)
```

**Loss Function:** Cross-Entropy Loss  
**Optimizer:** Adam  
**Epochs:** 10

---

## Experiments & Results

| # | Experiment | Neurons | Activation | LR | Dropout | BatchNorm | Accuracy | Loss |
|---|------------|---------|------------|----|---------|-----------|----------|------|
| 1 | Baseline ReLU | [128, 64] | ReLU | 0.001 | No | No | 96.93% | 0.0956 |
| 2 | Sigmoid | [128, 64] | Sigmoid | 0.001 | No | No | 97.16% | 0.0882 |
| 3 | Large Net + Low LR + Dropout + BatchNorm | [256, 128] | ReLU | 0.0005 | 0.3 | Yes | **98.15%** | **0.0626** |

**Best Model:** Experiment 3 — 98.15% accuracy with the lowest loss of 0.0626

**Key Takeaways:**
- Sigmoid slightly outperformed the ReLU baseline with the same architecture
- Scaling up neurons + adding Dropout + BatchNorm gave the biggest jump in performance
- Lower learning rate (0.0005) combined with a larger network led to more stable and accurate training

---

## Visualizations

The notebook includes:
- Training vs. Validation Loss curves for all 3 experiments
- Training vs. Validation Accuracy curves for all 3 experiments
- Bar chart comparison of final test accuracy and loss across experiments

---

## Enhancement Techniques Used (Exp 3)

| Technique | Reason |
|-----------|--------|
| **Dropout (0.3)** | Prevents overfitting by randomly deactivating neurons during training |
| **Batch Normalization** | Stabilizes training, allows faster convergence, reduces sensitivity to weight initialization |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install torch torchvision matplotlib pandas numpy
```

### 3. Run the notebook
```bash
jupyter notebook notebook.ipynb
```

The MNIST dataset will be automatically downloaded on first run — no manual setup needed.

---

## Repository Structure

```
├── notebook.ipynb       # Full project notebook
├── README.md            # This file
└── data/                # Auto-created by PyTorch on first run
```

