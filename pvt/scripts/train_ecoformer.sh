python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path path_of_imagenet \
    --data-set IMNET \
    --epochs 30 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune path_of_pretrained_model


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path /home/jliu/dl65/m3_imagenet \
    --data-set IMNET \
    --epochs 30 \
    --lr 5e-5 \
    --warmup-lr 1e-7 \
    --min-lr 1e-6 \
    --finetune /scratch/dl65/jing/Codes/transformer/EcoFormer_models/full_precision/pvt_v2_b0_msa/best_checkpoint.pth