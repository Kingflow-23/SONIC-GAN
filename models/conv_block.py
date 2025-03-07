# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """
    A convolutional block that includes Conv2d, BatchNorm2d, and LeakyReLU layers.

    This block is used for building convolutional layers in a neural network with batch normalization and LeakyReLU activations.

    Args:
        in_channel (int): Number of input channels (e.g., 3 for RGB images).
        out_channel (int): Number of output channels (e.g., 64 for feature maps).
        ker_size (int or tuple): Size of the convolutional kernel (e.g., 3 for a 3x3 kernel).
        padd (int or tuple): Padding added to the input (e.g., 1 for a padding of 1).
        stride (int): Stride for the convolution operation (e.g., 2 for downsampling).

    Attributes:
        conv (nn.Conv2d): The convolutional layer.
        norm (nn.BatchNorm2d): The batch normalization layer.
        LeakyRelu (nn.LeakyReLU): The LeakyReLU activation function with negative slope of 0.2.
    """

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=ker_size,
                stride=stride,
                padding=padd,
            ),
        ),
        self.add_module("norm", nn.BatchNorm2d(out_channel)),
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))
