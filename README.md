# SnowLab2ProbBop

This repository shows our neurocomputing paper "A probabilistic optimizer for binary neural networks".


## 📌 Overview

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
│   └── ProbBop2ndOrder.py/  
│
├── models_sgdat/               # Binary models (SGDAT-style)
│   ├── __init__.py.py/         # __all__ = ['binarynet', 'resnet_binary']
│   ├── binarized_modules.py/   # BinarizeLinear, BinarizeConv2d (1w1a / 1w32a)
│   ├── binarynet.py/          
│   └── resnet_binary.py/      
│
├── main_probs_cifar.py         # Entry: binary (models_sgdat) — nearly identical to main_binary_sgdat.py except bin_optimizer.step(last_step=len(data_loader))
├── main_probs_imagenet.py      # Entry: binary (models_sgdat)
└
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/snowloving/SnowLab2ProbBop.git
cd SnowLab2ProbBop
pip install -r requirements.txt
```

---


## 🖥️ Experiments

This benchmark supports two main experimental tracks:

### 🧩 Experiments on CIFAR and Tiny-imagenet

#### 📊 Results on BinaryNet

| Optimizer | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-----------|:--------:|:---------:|:-------------:|
| ProbBop | 89.48 | 64.17 | 45.19 |
| ProbBop2ndOrder | 89.81 | 65.56 | 46.58  |
> 📝 **Notes:**
> All results are from a single run with a fixed random seed (`seed_value=2020`). No hyperparameter tuning was performed.
> 
> ℹ️ **Note:** The "BinaryNet" used throughout this experiment refers to a compact VGG-style architecture (a.k.a. **VGG-Small**), implemented as `vgg_small` in `models_full_cifar/` and the "ResNet" used throughout this experiment refers to a ResNet18 architecture modified for ImageNet, implemented as `resnet18` in `models_full_cifar/`.

#### 📋 Quick Example Command

```bash
python main_probs_cifar.py \
  --model binarynet \
  --save binarynet_cifar10_ProbBop_f0 \
  --dataset cifar10 \
  --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':0}}" \
  --binarization det \
  --gpus 0
```

<details> <summary>🔁 All Reproducible Commands on BinaryNet</summary>

---

**cifar10 with ProbBop** 
```
python main_probs_cifar.py --model binarynet  --save binarynet_cifar10_ProbBop_f0 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':0}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

**cifar10 with ProbBop2ndOrder** 
```
python main_probs_cifar.py --model binarynet --save binarynet_cifar10_ProbBop2ndOrder_f0 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':0}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

**cifar100 with ProbBop** 
```
python main_probs_cifar.py --model binarynet  --save binarynet_cifar100_ProbBop_f0 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':0}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

**cifar100 with ProbBop2ndOrder** 
```
python main_probs_cifar.py --model binarynet --save binarynet_cifar100_ProbBop2ndOrder_f0 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':0}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

**tiny_imagenet with ProbBop** 
```
python main_probs_cifar.py --model binarynet  --save binarynet_tiny_imagenet_ProbBop_f0 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':0}}"  --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 1
```

**tiny_imagenet with ProbBop2ndOrder** 
```
python main_probs_cifar.py --model binarynet --save binarynet_tiny_imagenet_ProbBop2ndOrder_f0 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':0}}"  --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2
```

</details>

#### 📊 Results on ResNet

| Optimizer | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-----------|:--------:|:---------:|:-------------:|
| SGD | ⌛️ | ⌛️ | ⌛️ |
| SGDM | ⌛️ | ⌛️ | ⌛️ |
| Adam | ⌛️ | ⌛️ | ⌛️ |
| Bop | ⌛️ | ⌛️ | ⌛️ |
| Bop2ndOrder | ⌛️ | ⌛️ | ⌛️ |
| SGDT | ⌛️ | ⌛️ | ⌛️ |
| SGDAT | ⌛️ | ⌛️ | ⌛️ |
| ProbBop | ⌛️ | ⌛️ | ⌛️ |
| ProbBop2ndOrder | ⌛️ | ⌛️ | ⌛️  |


<details> <summary>🔁 All Reproducible Commands on ResNet</summary>

---

**CIFAR-10 with SGD** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 with SGDM** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_SGDM --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with Adam** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_Adam --dataset cifar10 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with Bop** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_Bop --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_Bop2ndOrder --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 with SGDT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_SGDT --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with SGDAT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar10_SGDAT --dataset cifar10 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 with SGD** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with SGDM** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_SGDM --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 with Adam** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with Bop** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with SGDT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_SGDT --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 with SGDAT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**tiny-imagenet with SGD** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_SGD --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

**tiny-imagenet with SGDM** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_SGDM --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
```

**tiny-imagenet with Adam** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_Adam --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

**tiny-imagenet with Bop** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_Bop --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
```

**tiny-imagenet with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_Bop2ndOrder --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 2
```

**tiny-imagenet with SGDT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_SGDT --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 64 --epochs 100 -b 256 --gpus 2
```

**tiny-imagenet with SGDAT** 
```bash
python main_binary_sgdat.py --model resnet_binary --save resnet_binary_tiny_imagenet_SGDAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3
```
</details>


#### 🔨 Optimizer Configuration Reference

| Optimizer | `--bin_regime` Configuration |
|-----------|---------------------------|
| ProbBop | `"{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"` |
| ProbBop2ndOrder | `"{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}"` |

## 🎨 Experiments on ImageNet

<!-- **Command** 
- Benchmark_imagenet_alexnet1w1a_cos.sh (4.0.1)
- Benchmark_imagenet_alexnet1w1a_MSteps.sh (4.0.2)
- Benchmark_imagenet_birealnet1w1a_cos.sh (4.1.1)
- Benchmark_imagenet_birealnet1w1a_MSteps.sh (4.1.2)
-->


|Number        | Model         | Optimizer         | Learning Scheduler| Acc@1(e200)   |
| :---         | :---          |    :----          |        :---       |         :---  |
| 4.0.0        | alexnet       | Bop2ndOrder       | -                 | 46.90         |
| 4.0.1        | alexnet       | ProbBop2ndOrder   | cos               | 45.94         |
| 4.0.2        | alexnet       | ProbBop2ndOrder   | MSteps            |               | 
| 4.1.0        | birealnet     | Bop2ndOrder       | -                 | 57.20         |
| 4.1.1        | birealnet     | ProbBop2ndOrder   | cos               | 56.87         |
| 4.1.2        | birealnet     | ProbBop2ndOrder   | MSteps            |               |



**SGDAT: XnorNet + ProbBop2ndOrder + cos**

```
#!/bin/bash

wd=1e-4
lr=0.1
minlr=1e-4

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=28678 main_probs_imagenet.py -a alexnet_1w1a_sgdat --dali_cpu \
-save imagenet_alexnet1w1a_benchmark_ProbBop2ndOrder_Recu_cos  --wd ${wd} --lr ${lr} --minlr ${minlr} \
--epochs 200 -b 1024 -j 8 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.5, 'formula':14}}" --lr_decay cos /home/hexue/bnn_vi-master/datasets/imagenet

```

**SGDAT: BirealNet + ProbBop2ndOrder + cos**

```
#!/bin/bash

wd=1e-4
lr=0.1
minlr=1e-4

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port=24778 main_probs_imagenet.py -a resnet18_1w1a_sgdat  --dali_cpu \
-save imagenet_birealnet1w1a_benchmark_ProbBop2ndOrder_Recu_cos  --wd ${wd} --lr ${lr} --minlr ${minlr} \
--epochs 200 -b 1024 -j 8 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.5, 'formula':14}}" --lr_decay cos /home/hexue/bnn_vi-master/datasets/imagenet

```


**SGDAT: XnorNet + ProbBop2ndOrder + MSteps**

```
#!/bin/bash

wd=1e-4
lr=0.1
minlr=1e-4

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=28678 main_probs_imagenet.py -a alexnet_1w1a_sgdat --dali_cpu \
-save imagenet_alexnet1w1a_benchmark_ProbBop2ndOrder_Recu_MSteps  --wd ${wd} --lr ${lr} --minlr ${minlr} \
--epochs 200 -b 1024 -j 8 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.5, 'formula':14}}" --lr_decay MSteps /home/hexue/bnn_vi-master/datasets/imagenet

```


**SGDAT: XnorNet + ProbBop + cos**

```
#!/bin/bash

wd=1e-4
lr=0.1
minlr=1e-4

# Use paste and process substitution to iterate over seeds and GPU_ids simultaneously
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=28678 main_probs_imagenet.py -a alexnet_1w1a_sgdat --dali_cpu \
-save imagenet_alexnet1w1a_benchmark_ProbBop2ndOrder_Recu_cos  --wd ${wd} --lr ${lr} --minlr ${minlr} \
--epochs 200 -b 1024 -j 8 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-3,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --lr_decay cos /home/hexue/bnn_vi-master/datasets/imagenet

```

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
