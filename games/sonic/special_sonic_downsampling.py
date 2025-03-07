import torch
from torch.nn.functional import interpolate
from torch.nn import Softmax

from games.sonic.tokens import TOKEN_DOWNSAMPLING_HIERARCHY as HIERARCHY


def special_sonic_downsampling(num_scales, scales, image, token_list):
    """
    Special Downsampling Method designed for Sonic Token based levels.

    Args:
        num_scales (int): The number of scales to which the image will be downscaled.
        scales (list of tuples): A list of tuples representing downsampling scales (scale_x, scale_y).
                                Each tuple corresponds to one of the downsampling steps, and there should be
                                'num_scales' tuples in total.
        image (torch.Tensor): The original level image to be scaled down. It is expected to be a torch tensor
                              representing a multi-channel image (e.g., one-hot encoding).
        token_list (list of str): A list of tokens that appear in the image in the order of channels
                                  in the `image` tensor.

    Returns:
        list: A list of downscaled image tensors. Each element in the list corresponds to an image at a
              different downscale level.

    Description:
        This method downscales a given image through multiple hierarchical scales while maintaining the
        integrity of different token groups. The downsampling is done in a multi-step process where:

        1. The image is first downscaled using bilinear interpolation.
        2. Each pixel in the downscaled image is analyzed, and the corresponding token hierarchy is
           identified based on a predefined hierarchy `HIERARCHY` defined in tokens.py.
        3. After finding the appropriate token hierarchy for each pixel, tokens that belong to the correct
           group in the hierarchy are retained.
        4. A Softmax function is applied to each pixel in the scaled image to make the output resemble
           the generator's output, promoting smooth transitions between token groups.

        The final output is a list of tensors representing the downscaled images, one for each scale in
        the downsampling hierarchy, with the list in reverse order of scaling.
    """

    scaled_list = []
    for sc in range(num_scales):
        scale_v = scales[sc][1]
        scale_h = scales[sc][0]

        # Initial downscaling of one-hot level tensor is normal bilinear scaling
        bil_scaled = interpolate(
            image,
            (int(image.shape[-2] * scale_v), int(image.shape[-1] * scale_h)),
            mode="bilinear",
            align_corners=False,
        )

        # Init output level
        img_scaled = torch.zeros_like(bil_scaled)

        for x in range(bil_scaled.shape[-2]):
            for y in range(bil_scaled.shape[-1]):
                curr_h = 0
                curr_tokens = [
                    tok
                    for tok in token_list
                    if bil_scaled[:, token_list.index(tok), x, y] > 0
                ]
                for h in range(
                    len(HIERARCHY)
                ):  # find out which hierarchy group we're in
                    for token in HIERARCHY[h].keys():
                        if token in curr_tokens:
                            curr_h = h

                for t in range(bil_scaled.shape[-3]):
                    if not (token_list[t] in HIERARCHY[curr_h].keys()):
                        # if this token is not on the correct hierarchy group, set to 0
                        img_scaled[:, t, x, y] = 0
                    else:
                        # if it is, keep original value
                        img_scaled[:, t, x, y] = bil_scaled[:, t, x, y]

                # Adjust level to look more like the generator output through a Softmax function.
                img_scaled[:, :, x, y] = Softmax(dim=1)(30 * img_scaled[:, :, x, y])

        scaled_list.append(img_scaled)

    scaled_list.reverse()
    return scaled_list
