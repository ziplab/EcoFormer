import argparse
import collections
import datetime
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

import gvt
import utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from losses import DistillationLoss
from params import args
from samplers import RASampler

warnings.filterwarnings("ignore")


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint

        state_keys = list(checkpoint_model.keys())
        for k, p in model.named_parameters():
            if k in checkpoint_model:
                state_keys.remove(k)
                p.data = checkpoint_model[k].float()
            else:
                # load from pretrained twins
                if args.train_msa:
                    refer_k = k.split(".")
                    refer_k[-2] = "q"
                    target_key = ".".join(refer_k)
                    q_weight = checkpoint_model[target_key].float()
                    state_keys.remove(target_key)

                    refer_k[-2] = "kv"
                    target_key = ".".join(refer_k)
                    kv_weight = checkpoint_model[target_key].float()
                    state_keys.remove(target_key)
                    p.data = torch.cat((q_weight, kv_weight), dim=0)
                else:
                    # load from msa weight
                    refer_k = k.split(".")
                    source_k = refer_k[-2]
                    refer_k[-2] = "qkv"
                    target_key = ".".join(refer_k)
                    qkv_weight = checkpoint_model[target_key].float()
                    shapes = qkv_weight.shape
                    dim = shapes[0] // 3
                    if target_key in state_keys:
                        state_keys.remove(target_key)
                    if source_k == "to_qk":
                        p.data = qkv_weight[:dim, ...]
                    elif source_k == "to_v":
                        p.data = qkv_weight[-dim:, ...]

        if len(state_keys) > 0:
            for temp_key in state_keys:
                print("Not used: %s" % temp_key)
        else:
            print("Load from pretrained model - All matched")

    model.to(device)
    model_ema = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(criterion, None, "none", 0, 0)

    if not args.output_dir:
        args.output_dir = args.model
        if utils.is_main_process():
            import os

            if not os.path.exists(args.model):
                os.mkdir(args.model)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model" in checkpoint:
            model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint)

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

        if "epoch" in checkpoint:
            if hasattr(model, "module"):
                model.module.set_retrain_resume()
            else:
                model.set_retrain_resume()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    if args.throughout:
        from logger import create_logger

        logger = create_logger(
            output_dir=output_dir, dist_rank=utils.get_rank(), name=args.model
        )
        throughput(data_loader_val, model, logger)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if hasattr(args, "k"):
            if epoch % args.k == 0:
                if hasattr(model, "module"):
                    model.module.set_retrain()
                else:
                    model.set_retrain()

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune
            == "",  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        if test_stats["acc1"] > max_accuracy:
            if args.output_dir:
                checkpoint_paths = [output_dir / "checkpoint_best.pth"]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Twins training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
