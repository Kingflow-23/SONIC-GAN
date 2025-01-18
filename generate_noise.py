# Code based on https://github.com/tamarott/SinGAN
import torch


def generate_spatial_noise(size, device, *args, **kwargs):
    """
    Generates a spatial noise tensor using a normal distribution.

    Args:
        size (list or tuple): The shape of the noise tensor to generate.
        device (torch.device): The device where the tensor will be allocated (e.g., 'cuda' or 'cpu').
        *args: Additional arguments for future extensions or custom noise generation.
        **kwargs: Additional keyword arguments for future extensions or custom noise generation.

    Returns:
        torch.Tensor: A tensor of the specified size filled with random values from a normal distribution.
    """

    # noise = generate_noise([size[0], *size[2:]], *args, **kwargs)
    # return noise.expand(size)
    return torch.randn(size, device=device)
