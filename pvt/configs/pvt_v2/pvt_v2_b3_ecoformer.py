cfg = dict(
    model="pvt_v2_b3",
    drop_path=0.1,
    clip_grad=None,
    attn_type="ecoformer",
    output_dir="checkpoints/pvt_v2_b3_ecoformer",
    nbits=16,
    m=25,
    k=300,
    topk=10,
)
