# DSLNDD-Net: A Multi-Scale Edge-Aware Attention Network for Nutrient Deficiency Detection in Dense Strawberry Leaves

This repository provides the official implementation of the paper:

**DSLNDD-Net: A Multi-Scale Edge-Aware Attention Network for Nutrient Deficiency Detection in Dense Strawberry Leaves**

---

## Overview

DSLNDD-Net is designed for nutrient deficiency detection in dense strawberry leaves. The framework integrates:

- **MEPN**: Multi-Scale Edge Perception Network
- **CSEAB**: Cascaded Squeeze-and-Excitation Attention Block
- **DS_UIoU**: Dynamic Shape-Unified IoU loss

The proposed model improves detection performance for dense, small, and highly similar deficiency targets, including **Ca**, **Fe**, and **P** deficiency symptoms in strawberry leaves.

---

## Repository Structure

~~~text
.
├── block.py              # Model blocks and network modules
├── loss.py               # Loss function definitions
├── two69+all.yaml        # Model configuration file
├── train.py              # Training script
├── test.py               # Testing / inference script
└── README.md
~~~

### File Description

- **block.py**: implementation of the proposed network modules
- **loss.py**: implementation of the DS_UIoU loss
- **two69+all.yaml**: model architecture and configuration file

---

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional, recommended for training and inference)

Please install the required packages according to your environment.

---

## Dataset

The dataset used in this study is publicly available at:

[Dataset Download Link](https://drive.google.com/drive/folders/1HeB-U_AUlxknExO_jJNN7_J5-YbYxvSS?usp=sharing)

After downloading the dataset, please unzip it and place the `datasets/` directory in the root of this repository.

---

## Pretrained Weights

The pretrained weights can be downloaded from:

[Weights Download Link](https://drive.google.com/drive/folders/1v1X2GWuixjs3Yw7hXA8t8CkVQxaANol-?usp=drive_link)

---

## Training

To train DSLNDD-Net on your own machine:

1. Download the dataset from the link above.
2. Unzip the dataset and move the `datasets/` folder to the root directory of this repository.
3. Check the configuration file and training settings.
4. Run:

~~~bash
python train.py
~~~

---

## Testing

To evaluate the model or perform inference, please check the arguments in `test.py` and run:

~~~bash
python test.py
~~~

---

## Code Availability

The source code and pretrained model weights are publicly available at:

[GitHub Repository](xxxxx.git)

---

## Data Availability

The dataset, code, and pretrained weights are publicly available through the links provided above.

---

## Citation

If you find this repository useful for your research, please cite our paper.

~~~bibtex
@article{DSLNDDNet,
  title={DSLNDD-Net: A Multi-Scale Edge-Aware Attention Network for Nutrient Deficiency Detection in Dense Strawberry Leaves},
  author={...},
  journal={...},
  year={...}
}
~~~

---

## Contact

For questions regarding the dataset or code, please contact:

**Email:** 18154346943@163.com

---

## License

This code is released for academic research purposes only.
