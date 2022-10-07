import torch


def create_gen(seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


def inner_product_kernel(x, y):
    kernel_matrix = torch.einsum("...id,...jd->...ij", x, y)
    return kernel_matrix


def softmax_kernel(x, y):
    kernel_matrix = torch.einsum("...id,...jd->...ij", x, y).exp()
    return kernel_matrix


def square_dist(x, y):
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-3)
    sqd = torch.sum((x - y) ** 2, -1)
    return sqd


def gaussian_kernel(x, y, variance=1.0):
    sqd = square_dist(x, y)
    K = torch.exp(-sqd / (2 * variance))
    return K


def compute_mean(value_list):
    return sum(value_list) / len(value_list)


def create_gen(seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return gen


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input
