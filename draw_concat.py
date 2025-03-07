# Code based on https://github.com/tamarott/SinGAN
from torch.nn.functional import interpolate

from generate_noise import generate_spatial_noise
from games.mario.level_utils import group_to_token


def format_and_use_generator(
    curr_img, G_z, count, mode, Z_opt, pad_noise, pad_image, noise_amp, G, opt
):
    """
    Formats the input for the generator correctly and runs it through the generator.

    Args:
        curr_img (torch.Tensor): The current image or noise map being processed.
        G_z (torch.Tensor): The generator's output from the previous scale.
        count (int): The current scale index.
        mode (str): The mode of operation ('rand' for random noise, 'rec' for reconstruction).
        Z_opt (torch.Tensor): The optimized noise used for reconstruction.
        pad_noise (function): Function to apply padding to the noise input.
        pad_image (function): Function to apply padding to the generator's output.
        noise_amp (float): The amplitude of the noise added to the input.
        G (torch.nn.Module): The generator model for the current scale.
        opt (argparse.Namespace): Configuration options.

    Returns:
        torch.Tensor: The output of the generator for the current scale.
    """

    if curr_img.shape != G_z.shape:
        G_z = interpolate(
            G_z, curr_img.shape[-2:], mode="bilinear", align_corners=False
        )
    if count == (opt.token_insert + 1):  # (opt.stop_scale - 1):
        G_z = group_to_token(G_z, opt.token_list)
    if mode == "rand":
        curr_img = pad_noise(curr_img)  # Curr image is z in this case
        z_add = curr_img
    else:
        z_add = Z_opt
    G_z = pad_image(G_z)
    z_in = noise_amp * z_add + G_z
    G_z = G(z_in.detach(), G_z)
    return G_z


def draw_concat(
    generators,
    noise_maps,
    reals,
    noise_amplitudes,
    in_s,
    mode,
    pad_noise,
    pad_image,
    opt,
):
    """
    Draws and concatenates the output from the previous scale with a new noise map,
    passing it through the generators at each scale.

    Args:
        generators (list of torch.nn.Module): The list of generators for each scale.
        noise_maps (list of torch.Tensor): The list of noise maps for each scale.
        reals (list of torch.Tensor): The list of real images for each scale.
        noise_amplitudes (list of float): The noise amplitude for each scale.
        in_s (torch.Tensor): The input to the current scale, usually the previous scale's output.
        mode (str): The mode of operation ('rand' for random noise, 'rec' for reconstruction).
        pad_noise (function): Function to apply padding to the noise input.
        pad_image (function): Function to apply padding to the generator's output.
        opt (argparse.Namespace): Configuration options.

    Returns:
        torch.Tensor: The output of the final generator after all scales are processed.
    """

    G_z = in_s
    if len(generators) > 0:
        if mode == "rand":
            noise_padding = 1 * opt.num_layer
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)
            ):
                if count < opt.stop_scale:  # - 1):
                    z = generate_spatial_noise(
                        [
                            1,
                            real_curr.shape[1],
                            Z_opt.shape[2] - 2 * noise_padding,
                            Z_opt.shape[3] - 2 * noise_padding,
                        ],
                        device=opt.device,
                    )
                G_z = format_and_use_generator(
                    z,
                    G_z,
                    count,
                    "rand",
                    Z_opt,
                    pad_noise,
                    pad_image,
                    noise_amp,
                    G,
                    opt,
                )

        if mode == "rec":
            for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
                zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)
            ):
                G_z = format_and_use_generator(
                    real_curr,
                    G_z,
                    count,
                    "rec",
                    Z_opt,
                    pad_noise,
                    pad_image,
                    noise_amp,
                    G,
                    opt,
                )

    return G_z
