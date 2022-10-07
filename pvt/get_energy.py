import warnings

import torch
from timm.models import create_model

# import models
import pvt
import pvt_v2
from params import args

warnings.filterwarnings("ignore")

try:
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import (
        flops_to_string,
        get_model_complexity_info,
        params_to_string,
    )
except ImportError:
    raise ImportError("Please upgrade mmcv to >0.6.2")


def msa_flops(h, w, sr_ratio, dim, heads):
    muls = 0
    adds = 0

    # q@k and attn@v
    muls += 2 * h * w * h * w * dim
    adds += 2 * h * w * h * w * dim

    # scale
    muls += heads * h * w * h * w

    return muls, adds


def fast_attn_ecoformer_flops(h, w, sr_ratio, dim, heads):
    H = heads
    N = h * w
    C = dim // heads
    Nh = args.nbits
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


attn_flops = {
    "msa": msa_flops,
    "ecoformer": fast_attn_ecoformer_flops,
}


def get_energy(model, input_shape):
    print(args.attn_type)
    flops, params = get_model_complexity_info(
        model, input_shape, as_strings=False, print_per_layer_stat=False
    )
    H = W = 224
    flop_func = attn_flops[args.attn_type]
    muls = flops
    adds = flops
    for i in range(4):
        if i == 3:
            stage_muls, stage_adds = msa_flops(
                H // (4 * (2 ** i)),
                W // (4 * (2 ** i)),
                model.sr_ratios[i],
                model.embed_dims[i],
                model.num_heads[i],
            )
        else:
            stage_muls, stage_adds = flop_func(
                H // (4 * (2 ** i)),
                W // (4 * (2 ** i)),
                model.sr_ratios[i],
                model.embed_dims[i],
                model.num_heads[i],
            )

        muls += stage_muls * model.depths[i]
        adds += stage_adds * model.depths[i]

    print("{:<30}  {:<8}".format("Mul: ", round(muls / 1e9, 2)))
    print("{:<30}  {:<8}".format("Add: ", round(adds / 1e9, 2)))
    print("{:<30}  {:<8}".format("Energy: ", round((muls * 3.7 + adds * 0.9) / 1e9, 2)))
    print("{:<30}  {:<8}".format("Area: ", round((muls * 7700 + adds * 4184) / 1e9, 0)))
    return flops_to_string(muls), params_to_string(params)


def main():
    input_shape = (3, 224, 224)

    model = create_model(args.model, pretrained=False, num_classes=1000)
    print(model)
    model.set_retrain_resume()
    model.name = args.model
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    get_energy(model, input_shape)


if __name__ == "__main__":
    main()
