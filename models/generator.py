# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """
    Patch-based Generator for generating images based on input and previous outputs.

    The generator takes in two inputs, `x` and `y`, and produces an image by applying a series of convolutional blocks.
    The output is a combination of the generated image and the previous output (`y`), allowing for a skip connection
    to refine the generated result. A temperature parameter is used in the softmax function for controlling the diversity
    of generated images.

    Args:
        opt (Namespace): Contains hyperparameters for the generator:
            - opt.nc_current (int): Number of input/output channels.
            - opt.nfc (int): Number of feature channels.
            - opt.ker_size (int): Size of the convolutional kernels.
            - opt.num_layer (int): Number of layers in the network.

    Attributes:
        head (ConvBlock): Initial convolutional block.
        body (nn.Sequential): Series of convolutional blocks forming the body of the network.
        tail (nn.Sequential): Final convolutional layer that generates the output image.

    Methods:
        forward(x, y, temperature=1):
            - x: Input tensor for the generator.
            - y: Previous output tensor (used in the skip connection).
            - temperature: Scaling factor for softmax to control image diversity.
    """

    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(
            opt.nc_current, N, (opt.ker_size, opt.ker_size), 0, 1
        )  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(
            nn.Conv2d(
                N,
                opt.nc_current,
                kernel_size=(opt.ker_size, opt.ker_size),
                stride=1,
                padding=0,
            )
        )

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = F.softmax(
            x * temperature, dim=1
        )  # Softmax is added here to allow for the temperature parameter
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind : (y.shape[2] - ind), ind : (y.shape[3] - ind)]
        return x + y
