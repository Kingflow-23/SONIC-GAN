import torch
from torch.nn.functional import interpolate
from torch.nn import Softmax

from games.mariokart.tokens import TOKEN_DOWNSAMPLING_HIERARCHY as HIERARCHY


def special_mariokart_downsampling(num_scales, scales, image, token_list):
    """
    Special Downsampling Method designed for Mario Kart token-based levels.

    This function downscales the given level image at various scales, applying hierarchical token adjustments
    according to a predefined token hierarchy. The scaled versions of the image are returned in reverse order.

    Args:
        num_scales (int): Number of scales to which the image should be downscaled.
        scales (list of tuple): A list of tuples (scale_x, scale_y), each defining the scaling factors
                                for the x and y dimensions respectively.
        image (torch.Tensor): The original level image to be scaled down, expected to be a tensor with one-hot encoded channels.
        token_list (list of str): A list of tokens corresponding to the channels in the input image.

    Returns:
        list of torch.Tensor: A list of scaled down versions of the input image, adjusted for token hierarchy.
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
