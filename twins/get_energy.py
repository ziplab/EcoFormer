import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.models import create_model

import gvt
import utils
from params import args

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


def msa_flops(h, w, dim, heads):
    muls = 0
    adds = 0

    # q@k and attn@v
    muls += 2 * h * w * h * w * dim
    adds += 2 * h * w * h * w * dim

    # scale
    muls += heads * h * w * h * w

    return muls, adds


def fast_attn_ecoformer_flops(h, w, dim, heads):
    H = heads
    N = h * w
    C = dim // heads
    Nh = 16
    muls = 0
    adds = 0

    # kernel
    m = 25
    adds += H * N * m * C
    adds += H * N * m * C
    muls += H * N * m * 2

    # hashing, H, N, m, Nh
    muls += H * N * m * Nh
    adds += H * N * m * Nh

    # out = linear_attention(q, k, v)
    # k_cumsum = k.sum(dim=-2)
    adds += H * N * Nh

    # D_inv = 1. / (torch.einsum('...nd,...d->...n', q.float(), k_cumsum) + bottom_bias)
    muls += H * N * Nh + H * N
    adds += H * N * Nh

    # context = torch.einsum('...nd,...ne->...de', k, v)
    adds += H * N * Nh * C

    # out = torch.einsum('...de,...nd->...ne', context, q) + top_bias
    adds += H * N * Nh * C + H * N * C

    # out2 = torch.einsum('...ne,...n->...ne', out, D_inv)
    muls += H * N * C

    return muls, adds


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print(f"Creating model: {args.model}")
    args.nb_classes = 1000
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print(model)

    model.set_retrain_resume()
    try:
        from mmcv.cnn import get_model_complexity_info
        from mmcv.cnn.utils.flops_counter import get_model_complexity_info

        flops, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=False, print_per_layer_stat=False
        )
        H = W = 224
        # flop_func = attn_flops[args.attn_type]
        muls = flops
        adds = flops
        for i in range(4):
            # stage_muls, stage_adds = msa_flops(H // (4 * (2 ** i)), W // (4 * (2 ** i)), model.embed_dims[i],
            #                                    model.num_heads[i])
            if i != 3:
                if args.train_msa:
                    stage_muls, stage_adds = msa_flops(
                        H // (4 * (2 ** i)),
                        W // (4 * (2 ** i)),
                        model.embed_dims[i],
                        model.num_heads[i],
                    )
                else:
                    stage_muls, stage_adds = fast_attn_ecoformer_flops(
                        H // (4 * (2 ** i)),
                        W // (4 * (2 ** i)),
                        model.embed_dims[i],
                        model.num_heads[i],
                    )
            else:
                stage_muls, stage_adds = msa_flops(
                    H // (4 * (2 ** i)),
                    W // (4 * (2 ** i)),
                    model.embed_dims[i],
                    model.num_heads[i],
                )
            muls += stage_muls * model.depths[i]
            adds += stage_adds * model.depths[i]

        print("{:<30}  {:<8}".format("Mul: ", round(muls / 1e9, 2)))
        print("{:<30}  {:<8}".format("Add: ", round(adds / 1e9, 2)))
        print("{:<30}  {:<8}".format("Energy: ", (muls * 3.7 + adds * 0.9) / 1e9))
        print("{:<30}  {:<8}".format("Area: ", (muls * 7700 + adds * 4184) / 1e9))
    except ImportError:
        raise ImportError("Please upgrade mmcv to >0.6.2")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Twins training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
