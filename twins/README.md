# Code for EcoFormer on Twins

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

2. Train a Twins-SVT model (e.g., Twins-SVT-S) with standard self-attention under 100 epochs. The model is initialized with corresponding pre-trained models in [Twins](https://github.com/Meituan-AutoML/Twins).

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 \
  --use_env main.py \
  --model alt_gvt_small \
  --batch-size 32 \
  --data-path [path of imagenet] \
  --dist-eval \
  --drop-path 0.2 \
  --epochs 100 \
  --finetune [path of Twins_SVT pretrained weights]
  --lr 5e-5 \
  --warmup-lr 1e-7 \
  --min-lr 1e-6 \
  --train_msa
```

3. Finetune the pre-trained models obtained in Step 2 with our EcoFormer.


```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ecoformer \
  --batch-size 32 \
  --data-path [path of imagenet] \
  --data-set IMNET \
  --dist-eval \
  --nbits 16 \
  --anchor 25 \
  --topk 10 \
  --k 300 \
  --epochs 30 \
  --lr 25e-5 \
  --warmup-lr 5e-7 \
  --min-lr 1e-6 \
  --drop-path 0.2 \
  --finetune [path of the pre-trained model in Step 2] \
  --output_dir checkpoints/alt_gvt_small_ecoformer
```

## Evaluation

To evaluate a model, you can

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --model alt_gvt_small_ecoformer \
    --batch-size 32 \
    --data-path [path of imagenet] \
    --data-set IMNET \
    --resume [path/to/trained_weights] \
    --eval
```

To obtain the number of multiplication, addition and energy, run

```bash
python get_energy.py --model alt_gvt_small_ecoformer
```

## Results and Models

| Model       | #Mul. (B) | #Add. (B) | Energy (B pJ) | Throughput (images/s) | Top-1 Acc. (%) | Download                                                                                       |
| ----------- | --------- | --------- | ------------- | --------------------- | -------------- | ---------------------------------------------------------------------------------------------- |
| Twins-SVT-S | 2.72      | 2.81      | 12.6          | 576                   | 80.22          | [Github](https://github.com/ziplab/EcoFormer/releases/download/v1.0/twins_svt_s_ecoformer.pth) |

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](../LICENSE) file.

## Acknowledgement

This repository is built upon [Twins](https://github.com/Meituan-AutoML/Twins). We thank the authors for their open-sourced code.