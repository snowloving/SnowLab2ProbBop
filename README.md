# SnowLab2ProbBop

This repository shows our neurocomputing paper "A probabilistic optimizer for binary neural networks".


## Overview

SnowLab2ProbBop is a PyTorch-based implementation of **ProbBop**, a probabilistic optimizer for Binary Neural Networks (BNNs). Unlike traditional BNN optimizers that rely on deterministic gradient approximations, ProbBop introduces a probabilistic treatment of binary weights, enabling more robust and accurate training.

---

## 📂 Repository Structure
```text
SnowBench/
├── README.md                   
├── requirements.txt            
├── data.py                     # Dataset loader: CIFAR-10/100, Tiny-ImageNet, ImageNet (with path configuration)
├── preprocess.py               # Data augmentation & preprocessing pipelines
├── utils.py                    # Logger, metrics, checkpointing, optimizer adjustment
│
├── datasets/
├── results/
├── optimizers/                 # ProbBop/ProbBop2ndOrder
│   ├── ProbBop.py/          
│   └── ProbBop3ndOrder.py/  
│
├── models_sgdat/               # Binary models (SGDAT-style)
│   ├── __init__.py.py/         # __all__ = ['binarynet', 'resnet_binary']
│   ├── binarized_modules.py/   # BinarizeLinear, BinarizeConv2d (1w1a / 1w32a)
│   ├── binarynet.py/          
│   └── resnet_binary.py/      
│  
├── main_binary_probbop.py      # Entry: binary & full-precision (models_sgdat) — nearly identical to main_binary_sgdat.py
└
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/snpwloving/SnowBench4Quant.git
cd SnowBench4Quant
pip install -r requirements.txt
```

---


## 🖥️ Experiments

This benchmark supports two main experimental tracks:

### 🧩 Experiments on CIFAR and Tiny-imagenet

#### 📊 Results on BinaryNet

| Optimizer | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-----------|:--------:|:---------:|:-------------:|
| ProbBop | ⌛️ | ⌛️ | ⌛️ |
| ProbBop2ndOrder | ⌛️ | ⌛️ | ⌛️  |

> 📝 **Notes:**
> All results are from a single run with a fixed random seed (`seed_value=2020`). No hyperparameter tuning was performed.
> 
> ℹ️ **Note:** The "BinaryNet" used throughout this experiment refers to a compact VGG-style architecture (a.k.a. **VGG-Small**), implemented as `vgg_small` in `models_full_cifar/` and the "ResNet" used throughout this experiment refers to a ResNet18 architecture modified for ImageNet, implemented as `resnet18` in `models_full_cifar/`.

#### 📋 Quick Example Command

```bash
python main_binary_sgdat.py \
  --model binarynet \
  --save binarynet_cifar10_SGD \
  --dataset cifar10 \
  --bin_regime "{0: {'optimizer': 'SGD', 'lr': 1e-4}}" \
  --binarization det \
  --gpus 0
```

<details> <summary>🔁 All Reproducible Commands on BinaryNet</summary>

---

**CIFAR-10 with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 with SGDM** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDM --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```
</details>

#### 🔨 Optimizer Configuration Reference

| Optimizer | `--bin_regime` Configuration |
|-----------|---------------------------|
| ProbBop | `"{0: {'optimizer': 'Bop','lr':1e-4}}"` |
| ProbBop2ndOrder | `"{0: {'optimizer': 'Bop2ndOrder','lr':1e-4}}"` |

---
## 📝 Citation
If you use this code in your research, please cite:

```bibtex
@article{he2025probabilistic,
  title={A probabilistic optimizer for binary neural networks},
  author={He, Xue and Geng, Xue and Zhang, Tiancheng and Yu, Minghe and Wu, Min and Yu, Ge and Zhao, Yuhai},
  journal={Neurocomputing},
  volume={648},
  pages={130297},
  year={2025},
  publisher={Elsevier}
}
```

## 🙏 Acknowledgements

This project builds upon [SGDAT](https://github.com/gushan/SGDAT). All baseline implementations (SGD, Adam, Bop, Bop2ndOrder, SGDAT) are unified in **SnowBench4Quant — [🔬 Experiment 1]: Binary Network Optimizer Comparison (SGDAT-style)**. We thank all open-source contributors.

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmail.com.

Contact
For questions or issues, please open a GitHub issue or contact the corresponding author via the paper.

text

---

Let me know if you want me to adjust the **project name**, **license type**, or add a **code of conduct / contributing guide**.
