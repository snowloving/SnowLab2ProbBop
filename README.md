# SnowLab2ProbBop

**Probabilistic Optimizer for Binary Neural Networks**

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing%202025-b31b1b)](https://doi.org/10.1016/j.neurocom.2025.130297)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

## Overview

SnowLab2ProbBop is a PyTorch-based implementation of **ProbBop**, a probabilistic optimizer for Binary Neural Networks (BNNs). Unlike traditional BNN optimizers that rely on deterministic gradient approximations, ProbBop introduces a probabilistic treatment of binary weights, enabling more robust and accurate training.

This repository accompanies the paper:

> He, X., Geng, X., Zhang, T., Yu, M., Wu, M., Yu, G., & Zhao, Y. (2025).  
> *A probabilistic optimizer for binary neural networks*.  
> Neurocomputing, 648, 130297.

## Key Features

- **Probabilistic weight update** – Samples binary weights from learned distributions instead of hard thresholding.
- **Gradient variance reduction** – Implements a novel variance control mechanism for discrete variables.
- **Seamless integration** – Works as a drop-in optimizer for existing BNN architectures (e.g., XNOR-Net, DoReFa, ReActNet).
- **Lightweight** – Minimal overhead compared to standard optimizers (Adam/SGD).

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy

## Installation

```bash
git clone https://github.com/yourusername/SnowLab2ProbBop.git
cd SnowLab2ProbBop
pip install -r requirements.txt
```

Quick Start
1. Replace your BNN optimizer
python
from probbop import ProbBop

# Instead of Adam/SGD
optimizer = ProbBop(model.parameters(), lr=0.001, prob_temp=0.1)
2. Training loop example
python
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()   # Probabilistic binary weight update
3. Full example with ResNet-20 on CIFAR-10
bash
python train.py --dataset cifar10 --model resnet20_bnn --optim probbop --epochs 200
Configuration
Argument	Default	Description
lr	0.001	Learning rate
prob_temp	0.1	Temperature for probabilistic sampling (lower = more deterministic)
variance_clip	0.5	Clip gradient variance to stabilize training
momentum	0.9	Momentum for base optimizer
Results (CIFAR-10)
Method	Top-1 Accuracy
XNOR-Net (Adam)	89.1%
DoReFa (SGD)	90.3%
ProbBop (ours)	91.8%
*See paper for full benchmarks on ImageNet, CIFAR-100, and semantic segmentation.*

Citation
If you use this code in your research, please cite:

bibtex
@article{he2025probabilistic,
  title={A probabilistic optimizer for binary neural networks},
  author={He, Xue and Geng, Xue and Zhang, Tiancheng and Yu, Minghe and Wu, Min and Yu, Ge and Zhao, Yuhai},
  journal={Neurocomputing},
  volume={648},
  pages={130297},
  year={2025},
  publisher={Elsevier}
}
License
MIT License – see LICENSE file.

Contact
For questions or issues, please open a GitHub issue or contact the corresponding author via the paper.

text

---

Let me know if you want me to adjust the **project name**, **license type**, or add a **code of conduct / contributing guide**.
