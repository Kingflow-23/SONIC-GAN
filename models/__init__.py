# Code based on https://github.com/tamarott/SinGAN
import os
import torch

from games.mario.tokens import TOKEN_GROUPS

from .generator import Level_GeneratorConcatSkip2CleanAdd
from .discriminator import Level_WDiscriminator


def weights_init(m):
    """
    Initialize weights for Conv and Norm layers.

    - Convolution layers are initialized with normal distribution (mean=0, std=0.02).
    - Normalization layers are initialized with normal distribution (mean=1, std=0.02) and bias filled with zeros.

    Args:
        m (torch.nn.Module): The module (layer) to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Norm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_models(opt):
    """
    Initialize the Generator and Discriminator models.

    Initializes the models with weights and, if specified, loads the state dictionaries from pre-trained models.

    Args:
        opt: The options containing model parameters, including file paths for pre-trained networks.

    Returns:
        D (torch.nn.Module): The discriminator model.
        G (torch.nn.Module): The generator model.
    """
    # generator initialization:
    G = Level_GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    G.apply(weights_init)
    if opt.netG != "":
        G.load_state_dict(torch.load(opt.netG))
    print(G)

    # discriminator initialization:
    D = Level_WDiscriminator(opt).to(opt.device)
    D.apply(weights_init)
    if opt.netD != "":
        D.load_state_dict(torch.load(opt.netD))
    print(D)

    return D, G


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    """
    Calculate the gradient penalty for Wasserstein GAN training.

    The penalty is used to enforce the 1-Lipschitz constraint on the discriminator.

    Args:
        netD (torch.nn.Module): The discriminator model.
        real_data (torch.Tensor): A batch of real data.
        fake_data (torch.Tensor): A batch of generated fake data.
        LAMBDA (float): The gradient penalty coefficient.
        device (torch.device): The device (CPU or GPU).

    Returns:
        torch.Tensor: The computed gradient penalty.
    """
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def save_networks(G, D, z_opt, opt):
    """
    Save the state dictionaries of the Generator, Discriminator, and noise map.

    Args:
        G (torch.nn.Module): The generator model.
        D (torch.nn.Module): The discriminator model.
        z_opt (torch.Tensor): The optimized noise map.
        opt: The options containing the output directory path.
    """
    torch.save(G.state_dict(), "%s/G.pth" % (opt.outf))
    torch.save(D.state_dict(), "%s/D.pth" % (opt.outf))
    torch.save(z_opt, "%s/z_opt.pth" % (opt.outf))


def restore_weights(D_curr, G_curr, scale_num, opt):
    """
    Restore the weights for the Generator and Discriminator from a previous scale.

    This function helps with progressive training of the models at different scales.

    Args:
        D_curr (torch.nn.Module): The current discriminator model.
        G_curr (torch.nn.Module): The current generator model.
        scale_num (int): The scale number to restore from.
        opt: The options containing the output directory path.

    Returns:
        D_curr (torch.nn.Module): The updated discriminator model.
        G_curr (torch.nn.Module): The updated generator model.
    """
    G_state_dict = torch.load("%s/%d/G.pth" % (opt.out_, scale_num - 1))
    D_state_dict = torch.load("%s/%d/D.pth" % (opt.out_, scale_num - 1))

    G_head_conv_weight = G_state_dict["head.conv.weight"]
    G_state_dict["head.conv.weight"] = G_curr.head.conv.weight
    G_tail_weight = G_state_dict["tail.0.weight"]
    G_state_dict["tail.0.weight"] = G_curr.tail[0].weight
    G_tail_bias = G_state_dict["tail.0.bias"]
    G_state_dict["tail.0.bias"] = G_curr.tail[0].bias
    D_head_conv_weight = D_state_dict["head.conv.weight"]
    D_state_dict["head.conv.weight"] = D_curr.head.conv.weight

    for i, token in enumerate(opt.token_list):
        for group_idx, group in enumerate(TOKEN_GROUPS):
            if token in group:
                G_state_dict["head.conv.weight"][:, i] = G_head_conv_weight[
                    :, group_idx
                ]
                G_state_dict["tail.0.weight"][i] = G_tail_weight[group_idx]
                G_state_dict["tail.0.bias"][i] = G_tail_bias[group_idx]
                D_state_dict["head.conv.weight"][:, i] = D_head_conv_weight[
                    :, group_idx
                ]
                break

    G_state_dict["head.conv.weight"] = (
        G_state_dict["head.conv.weight"].detach().requires_grad_()
    )
    G_state_dict["tail.0.weight"] = (
        G_state_dict["tail.0.weight"].detach().requires_grad_()
    )
    G_state_dict["tail.0.bias"] = G_state_dict["tail.0.bias"].detach().requires_grad_()
    D_state_dict["head.conv.weight"] = (
        D_state_dict["head.conv.weight"].detach().requires_grad_()
    )

    G_curr.load_state_dict(G_state_dict)
    D_curr.load_state_dict(D_state_dict)

    G_curr.head.conv.weight = torch.nn.Parameter(
        G_curr.head.conv.weight.detach().requires_grad_()
    )
    G_curr.tail[0].weight = torch.nn.Parameter(
        G_curr.tail[0].weight.detach().requires_grad_()
    )
    G_curr.tail[0].bias = torch.nn.Parameter(
        G_curr.tail[0].bias.detach().requires_grad_()
    )
    D_curr.head.conv.weight = torch.nn.Parameter(
        D_curr.head.conv.weight.detach().requires_grad_()
    )

    return D_curr, G_curr


def reset_grads(model, require_grad):
    """
    Reset the requires_grad flag for all parameters in the model.

    Args:
        model (torch.nn.Module): The model whose parameters' requires_grad flag will be updated.
        require_grad (bool): Whether to enable or disable gradients for the model parameters.

    Returns:
        model (torch.nn.Module): The model with updated gradient requirements.
    """
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def load_trained_pyramid(opt):
    """
    Load the trained models and data for the pyramid structure.

    If no trained models exist, a message is printed indicating that training must be done first.

    Args:
        opt: The options containing the output directory path.

    Returns:
        Gs (list): List of trained generator models.
        Zs (list): List of noise maps used for generating images.
        reals (list): List of real training images.
        NoiseAmp (list): List of noise amplitudes.
    """
    dir = opt.out_
    if os.path.exists(dir):
        reals = torch.load("%s/reals.pth" % dir)
        Gs = torch.load("%s/generators.pth" % dir)
        Zs = torch.load("%s/noise_maps.pth" % dir)
        NoiseAmp = torch.load("%s/noise_amplitudes.pth" % dir)

    else:
        print("no appropriate trained model exists, please train first")
    return Gs, Zs, reals, NoiseAmp
