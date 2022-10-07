python -m torch.distributed.launch --nproc_per_node=1 \
  --use_env main.py \
  --model alt_gvt_small \
  --batch-size 32 \
  --data-path /home/datasets/imagenet \
   --dist-eval \
   --drop-path 0.2 \
   --epochs 100 \
   --finetune /data1/nips2022/models/twins_model/alt_gvt_small.pth \
   --lr 5e-5 \
   --warmup-lr 1e-7 \
   --min-lr 1e-6 \
