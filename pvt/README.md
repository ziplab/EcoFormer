# Code for EcoFormer on PVTv2

## Dataset Preparation

Download the ImageNet 2012 dataset from [here](http://image-net.org/), and prepare the dataset based on this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). The file structure should look like:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Training

1. Activate your python environment

```bash
conda activate ecoformer
```

2. Train a PVTv2 model (e.g., PVTv2 B0) with standard self-attention under 100 epochs. The model is initialized with corresponding pre-trained models in [PVT](https://github.com/whai362/PVT/tree/v2/classification).

```bash
# train with 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_msa.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --epochs 100 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune [path of pvt_v2 pre-trained models]
```

3. Finetune the pre-trained models obtained in Step 2 with our EcoFormer.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --epochs 30 \ # note the difference
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune [path of the pre-trained model in Step 2]
```

## Evaluation

To evaluate a model, you can

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path/to/trained_weights] \
    --eval
```

To test the throughput, you can

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --throughput
```

To obtain the number of multiplication, addition and energy, run

```bash
python get_energy.py --config configs/pvt_v2/pvt_v2_b0_ecoformer.py
```

## Results and Models

| Model    | #Mul. (B) | #Add. (B) | Energy (B pJ) | Throughput (images/s) | Top-1 Acc. (%) | Download                                                                                    |
| -------- | --------- | --------- | ------------- | --------------------- | -------------- | ------------------------------------------------------------------------------------------- |
| PVTv2-B0 | 0.54      | 0.56      | 2.5           | 1379                  | 70.44          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/pvtv2_b0_ecoformer.pth) |
| PVTv2-B1 | 2.03      | 2.09      | 9.4           | 874                   | 78.38          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/pvtv2_b1_ecoformer.pth) |
| PVTv2-B2 | 3.85      | 3.97      | 17.8          | 483                   | 81.28          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/pvtv2_b2_ecoformer.pth) |
| PVTv2-B3 | 6.54      | 6.75      | 30.25         | 325                   | 81.96          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/pvtv2_b3_ecoformer.pth) |
| PVTv2-B4 | 9.57      | 9.82      | 44.25         | 249                   | 81.90          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/pvtv2_b4_ecoformer.pth) |

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](../LICENSE) file.

## Acknowledgement

This repository is built upon [PVT](https://github.com/whai362/PVT). We thank the authors for their open-sourced code.