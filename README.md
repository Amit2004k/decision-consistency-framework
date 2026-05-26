# Decision Consistency Framework (DCF)

A framework for measuring **prediction robustness** of deep learning classifiers under non-adversarial perturbations. Standard accuracy metrics can hide systematic failure modes — this work introduces three complementary metrics that expose them.

---

## Overview

When a model confidently predicts the same class across a 2-degree rotation, a brightness tweak, and a JPEG compression, that prediction is *consistent*. When it flips, that is a reliability problem that accuracy alone won't surface. DCF formalises this intuition into three metrics and validates them across four datasets and four CNN architectures.

**Datasets evaluated:**
- CIFAR-10 (32×32, 10 classes)
- STL-10 (96×96, 10 classes)
- Intel Image Classification / Scene Recognition (150×150, 6 classes)
- OCTMNIST (retinal OCT scans, 4 classes — medical domain)

**Models evaluated:**
- ResNet-18, ResNet-50
- VGG-16
- MobileNetV2

---

## Metrics

| Symbol | Name | Definition |
|--------|------|------------|
| **S** | Label Stability | Fraction of perturbations where the predicted class is unchanged |
| **D** | Confidence Deviation | Normalised mean shift in confidence across perturbations |
| **C** | Consistency Score | `α·S + (1−α)·(1−D)`, default α = 0.5 |

An additional diagnostic metric **CAG** (Consistency–Accuracy Gap = Accuracy − C) separates two failure modes:
- **CAG > 0** — Overconfident: model is right but brittle
- **CAG < 0** — Consistently Wrong: model makes the same mistake reliably (dangerous in high-stakes settings)

---

## Perturbations

All six perturbations are non-adversarial (no gradient-based attack):

| Perturbation | Parameters |
|---|---|
| Rotation | ±2° |
| Brightness | factor 1.1 |
| Gaussian noise | σ = 0.01 |
| Contrast | factor 1.1 |
| Gaussian blur | kernel 3×3, σ = 0.5 |
| JPEG compression | quality 75 |

---

## Repository Structure

```
decision-consistency-framework/
├── notebooks/
│   ├── 01_cifar_evaluation.ipynb          # CIFAR-10/100 — core results
│   ├── 02_stl10_evaluation.ipynb          # STL-10 — higher resolution
│   ├── 03_intel_scene_evaluation.ipynb    # Scene recognition — cross-domain
│   └── 04_theoretical_extensions.ipynb   # Lemmas, DINOv2, ViT, medical case study
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Amit2004k/decision-consistency-framework.git
cd decision-consistency-framework
pip install -r requirements.txt
```

Open any notebook in Jupyter and run all cells. Dataset paths are clearly marked at the top of each notebook — update them to match your local directory before running.

---

## Key Results

**CIFAR-10 (N=1000 samples)**

| Model | Label Stability ↑ | Confidence Deviation ↓ | Consistency Score ↑ |
|---|---|---|---|
| ResNet-18 | 0.876 | — | 0.782 |
| ResNet-50 | 0.865 | — | 0.789 |
| VGG-16 | 0.856 | — | 0.777 |
| MobileNetV2 | 0.874 | — | 0.776 |

**Intel Scene (cross-domain, ImageNet-pretrained on scene images)**

All models show CAG < 0 — high consistency but consistently wrong, revealing that the CAG diagnostic catches failure modes that accuracy alone misses.

---

## Theoretical Results

**Lemma 1 (Lipschitz Bound):** For a classifier with Lipschitz constant L and decision margin m, Label Stability satisfies S ≥ 1 − L·ε/m for perturbation magnitude ε.

**Lemma 2 (CAG Dichotomy):** Across all 16 evaluated configurations (4 datasets × 4 models), every configuration falls into either the Overconfident or Consistently Wrong regime. No well-calibrated configuration was found, which motivates DCF as a necessary complement to accuracy.

---

## Ablation Study

Model rankings under the Consistency Score are stable across α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, validating that results are not sensitive to the specific weighting choice.

---

## Dependencies

See `requirements.txt`. Core: PyTorch ≥ 1.13, torchvision, NumPy, SciPy, matplotlib, tqdm, Pillow.

---

## License

MIT
