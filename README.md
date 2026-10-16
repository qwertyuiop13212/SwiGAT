# SwiGAT

**SwiGAT: A Gaze-Guided Swin Transformer with Graph-Based Relational Reasoning for Chest X-ray Classification**

## Overview

Chest X-ray classification remains challenging because subtle radiographic patterns and overlapping anatomical structures can obscure diagnostically relevant evidence. Moreover, disease-related findings may be distributed across spatially separated regions.

SwiGAT is a **gaze-guided graph-based framework** that incorporates radiologist gaze maps as continuous spatial priors and performs relational reasoning among regional visual features.

SwiGAT consists of three main components:

- **Dual-Branch Feature Refinement**  
  Independently refines the X-ray and gaze representations using branch-specific instances of the proposed FocusFilterBlock (FFB).

- **Gaze-Guided Feature Weighting**  
  Uses the refined gaze map to weight the refined X-ray map through element-wise multiplication, producing gaze-modulated region-level features.

- **Graph-Based Relational Reasoning**  
  Constructs a fully connected graph over the regional features and employs a graph attention network (GAT) to model long-range inter-regional dependencies among spatially distant but diagnostically related regions.

## Architecture

![SwiGAT architecture](fig/model.png)

## Data Preparation

SwiGAT is evaluated on two gaze-annotated chest X-ray datasets:

- **EGD-CXR**  
  A gaze-annotated subset of MIMIC-CXR containing Normal, Congestive Heart Failure (CHF), and Pneumonia cases. We use 976 images for training and 107 images for testing.

- **SIIM-ACR**  
  A binary chest X-ray classification dataset for distinguishing pneumothorax from normal cases. We use 1,000 images for training and 250 images for testing.

The repository does not redistribute chest X-ray images or radiologist gaze data. Please obtain the datasets from their respective official sources and comply with their licenses and data-use requirements.

Organize each dataset as follows:

```text
data/
├── egd_cxr/
│   ├── images/
│   │   ├── sample_0001.png
│   │   └── ...
│   ├── gaze_maps/
│   │   ├── sample_0001.png
│   │   └── ...
│   ├── train.csv
│   └── test.csv
│
└── siim_acr/
    ├── images/
    │   ├── sample_0001.png
    │   └── ...
    ├── gaze_maps/
    │   ├── sample_0001.png
    │   └── ...
    ├── train.csv
    └── test.csv
```

Each CSV file should contain the following columns:

```text
image,gaze,label
sample_0001.png,sample_0001.png,0
sample_0002.png,sample_0002.png,1
```

The `image` and `gaze` entries indicate the corresponding filenames in the `images` and `gaze_maps` directories. The `label` column contains the integer class label.

All chest X-ray images and gaze maps are resized to `224 × 224`. Gaze maps are normalized and provided to the model as single-channel inputs.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The implementation requires Python, PyTorch, Torchvision, NumPy, Pandas, scikit-learn, and the graph-learning dependencies listed in `requirements.txt`.

## Training and Evaluation

`main.py` is the single entry point for both training and evaluation.

### EGD-CXR

```bash
python main.py --config configs/egd_cxr.yaml
```

### SIIM-ACR

```bash
python main.py --config configs/siim_acr.yaml
```

The configuration files specify the dataset path, number of classes, training hyperparameters, and output directory.

To evaluate a trained checkpoint without retraining:

```bash
python main.py --config configs/egd_cxr.yaml --mode test --checkpoint checkpoints/egd_cxr_best.pt
```

Replace the configuration and checkpoint paths as needed for SIIM-ACR.

## Results

SwiGAT achieves the highest overall performance among the compared image-based and gaze-guided methods on both datasets.

![SwiGAT result](fig/result.png) 

The reported results are averaged over three runs with different random seeds.

## Repository Structure

```text
SwiGAT/
├── main.py
├── configs/
│   ├── egd_cxr.yaml
│   └── siim_acr.yaml
├── models/
│   ├── swigat.py
│   ├── focus_filter_block.py
│   └── graph_attention.py
├── datasets/
│   └── chest_xray_dataset.py
├── utils/
│   ├── metrics.py
│   └── seed.py
├── fig/
│   ├── frame.png
│   └── result.png
├── requirements.txt
└── README.md
```

Users only need to execute `main.py`. Model construction, dataset loading, training, evaluation, and metric calculation are handled internally according to the selected configuration file.

