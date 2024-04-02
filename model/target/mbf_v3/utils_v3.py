import torch
from torch import Tensor
import math
import torch.nn.functional as F


def to_ycbcr(x: Tensor, data_range: float = 255) -> Tensor:
    r"""
    Converts a Tensor from RGB color space to YCbCr color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding an RGB image in :math:`(\ldots, C, H ,W)` format (where :math:`\ldots` indicates an arbitrary number of dimensions).
    data_range : float
        The range of the input/output data. i.e., 255 indicates pixels in [0, 255], 1.0 indicates pixels in [0, 1]. Only 1.0 and 255 are supported.

    Returns
    -------
    Tensor
        The YCbCr result of the same shape as the input and with the same data range.

    Note
    -----
    This function implements the "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which
    many libraries (excluding PIL) use as the default definition of YCbCr. This conversion (for [0, 255]) is given by:

    .. math::
        \begin{aligned}
        Y&=&0&+(0.299&\cdot R)&+(0.587&\cdot G)&+(0.114&\cdot B) \\
        C_{B}&=&128&-(0.168736&\cdot R)&-(0.331264&\cdot G)&+(0.5&\cdot B) \\
        C_{R}&=&128&+(0.5&\cdot R)&-(0.418688&\cdot G)&-(0.081312&\cdot B)
        \end{aligned}

    """
    assert data_range in [1.0, 255]

    # fmt: off
    ycbcr_from_rgb = torch.tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ],
    device=x.device).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([0, 128, 128], device=x.device).view(3, 1, 1)
    else:
        b = torch.tensor([0, 0.5, 0.5], device=x.device).view(3, 1, 1)

    x = torch.einsum("cv,...cxy->...vxy", [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def _normalize(N: int) -> Tensor:
    r"""
    Computes the constant scale factor which makes the DCT orthonormal
    """
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N: int) -> Tensor:
    r"""
    Computes the cosine harmonics for the DCT transform
    """
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(blocks: Tensor) -> Tensor:
    N = blocks.shape[3]

    n = _normalize(N).to(blocks.device)
    h = _harmonics(N).to(blocks.device)

    coeff = (2 / N) * n * (h.t() @ blocks @ h)

    return coeff


def discrete_cosine_transform(x):
    x = (x + 1) / 2 * 255
    x = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=True)
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return
    x = to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    bs, ch, h, w = x.shape
    block_num = h // 8
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0, stride=(8, 8))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)
    dct_block = block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    dct_block = dct_block[:, :, 1:, :, :]  # remove DC
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)
    return dct_block
