CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ksh \
  --batch-size 32 \
  --data-path /home/jliu/dl65/liujing/dataset/cifar \
  --data-set CIFAR \
  --dist-eval \
  --drop-path 0.2 \
  --nbits 16 \
  --anchor 30 \
  --topk 10 \
  --k 300 \
  --output_dir checkpoints/alt_gvt_small_ksh_n16_m25_topk10_k300

# imagenet from scratch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ksh \
  --batch-size 32 \
  --data-path /projects/dl65/m3_imagenet \
  --data-set IMNET \
  --dist-eval \
  --nbits 16 \
  --anchor 25 \
  --topk 10 \
  --k 300 \
  --output_dir checkpoints/alt_gvt_small_kshv2_n16_m25_topk10_k300_from_scratch

# imagenet finetune
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ksh \
  --batch-size 32 \
  --data-path /projects/dl65/m3_imagenet \
  --data-set IMNET \
  --dist-eval \
  --nbits 16 \
  --anchor 25 \
  --topk 10 \
  --k 300 \
  --epochs 100 \
  --lr 5e-5 \
  --warmup-lr 1e-7 \
  --min-lr 1e-6 \
  --drop-path 0.2 \
  --finetune /fs03/dl65/pzz/nips2022/after_submission/twins-ecoformer/outputs/alt_gvt_small_sra_to_msa_sra/checkpoint_best.pth \
  --output_dir checkpoints/alt_gvt_small_kshv2_n16_m25_topk10_k300_ft_e100


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ksh \
  --batch-size 32 \
  --data-path /projects/dl65/m3_imagenet \
  --data-set IMNET \
  --dist-eval \
  --nbits 16 \
  --anchor 25 \
  --topk 10 \
  --k 300 \
  --epochs 60 \
  --lr 25e-5 \
  --warmup-lr 5e-7 \
  --min-lr 1e-6 \
  --drop-path 0.2 \
  --finetune /home/zpan/dl65_scratch/pzz/nips2022/twins-liteformer/alt_gvt_small/checkpoint_best.pth \
  --output_dir checkpoints/alt_gvt_small_ksh_n16_m25_topk10_k300_ft_e60_5scale_lr


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
  --use_env main.py \
  --model alt_gvt_small_ksh \
  --batch-size 32 \
  --data-path /projects/dl65/m3_imagenet \
  --data-set IMNET \
  --dist-eval \
  --drop-path 0.2 \
  --nbits 16 \
  --anchor 25 \
  --topk 10 \
  --k 1 \
  --lr 5e-5 \
  --warmup-lr 1e-7 \
  --min-lr 1e-6 \
  --epochs 30 \
  --finetune /home/zpan/dl65_scratch/pzz/nips2022/twins-liteformer/alt_gvt_small/checkpoint_best.pth \
  --output_dir checkpoints/alt_gvt_small_ksh_n16_m25_topk10_k1_e30_i500_ft
