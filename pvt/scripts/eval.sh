python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b0_ecoformer.py \
    --batch-size 32 \
    --data-path /home/jliu/dl65/m3_imagenet \
    --data-set IMNET \
    --resume /scratch/dl65/jing/Codes/transformer/EcoFormer_models/ecoformer/pvt_v2_b0_performer_share_qk_ksh_hard_m25_top10_k300/best_checkpoint.pth \
    --eval

python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 \
    --use_env main.py \
    --config configs/pvt_v2/pvt_v2_b4_ecoformer.py \
    --batch-size 32 \
    --data-path /home/jliu/dl65/m3_imagenet \
    --data-set IMNET \
    --resume /scratch/dl65/jing/Codes/transformer/EcoFormer_models/ecoformer/pvt_v2_b4_performer_share_qk_ksh_hard_m25_top10_k300/best_checkpoint.pth \
    --eval