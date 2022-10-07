# EcoFormer Environment Installation

## Manual installation

If the exported anaconda environment file does not work on your machine, please consider installing the environment manually by following the instructions below.

* Install a python 3.8 environment

```bash
conda create -n ecoformer python=3.8
conda activate ecoformer
```

* Install PyTorch 1.10.1. You can also choose another CUDA version from [here](https://pytorch.org/get-started/previous-versions/#v1101).

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install related python dependencies

```bash
pip install timm==0.4.9 mmcv==1.3.8 einops==0.4.1 scipy==1.8.0
```