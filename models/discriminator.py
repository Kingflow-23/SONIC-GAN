# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock


class Level_WDiscriminator(nn.Module):
    """
    Patch-based discriminator for evaluating the quality of generated images.

    This discriminator uses a series of convolutional blocks to process input images and classify them as real or fake.
    It operates in a hierarchical fashion, with each layer progressively evaluating smaller patches of the input image.

    Args:
        opt (Namespace): Contains hyperparameters for the discriminator:
            - opt.nc_current (int): Number of input channels.
            - opt.nfc (int): Number of feature channels.
            - opt.ker_size (int): Size of the convolutional kernels.
            - opt.num_layer (int): Number of layers in the network.

    Attributes:
        head (ConvBlock): Initial convolutional block.
        body (nn.Sequential): Series of convolutional blocks forming the body of the network.
        tail (nn.Conv2d): Final convolutional layer outputting the discriminator's prediction.
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

        self.tail = nn.Conv2d(
            N, 1, kernel_size=(opt.ker_size, opt.ker_size), stride=1, padding=0
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
