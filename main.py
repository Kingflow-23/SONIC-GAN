# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
from train import train
from games.mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from games.mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from games.sonic.tokens import REPLACE_TOKENS as SONIC_REPLACE_TOKENS
from games.mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from games.mariokart.special_mariokart_downsampling import (
    special_mariokart_downsampling,
)
from games.mario.level_image_gen import LevelImageGen as MarioLevelGen
from games.mario.special_mario_downsampling import special_mario_downsampling
from games.sonic.special_sonic_downsampling import special_sonic_downsampling
from games.sonic.level_image_gen import LevelImageGen as SonicLevelGen
from games.mario.level_utils import read_level, read_level_from_file
from config import get_arguments, post_config
from loguru import logger
import wandb
import sys
import torch


def get_tags(opt):
    """
    Generate tags for logging based on the input name.

    Args:
        opt (argparse.Namespace): The parsed command-line arguments.

    Returns:
        list: A list containing the game name (input name) used for logging.
    """
    return [opt.input_name.split(".")[0]]


def main():
    """
    Main training function that handles argument parsing, logger initialization, training process,
    and sample generation.

    - Parses input arguments and configurations.
    - Initializes the logger and Weights & Biases (wandb) for experiment tracking.
    - Initializes game-specific configurations (sprite paths, token replacements, and downsampling).
    - Reads the level data and trains the model.
    - Generates random samples based on the trained model.

    Side effects:
        - Logs training progress and results to wandb.
        - Saves generated samples to disk.
    """

    # torch.autograd.set_detect_anomaly(True)

    # Logger init
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        + "<level>{level}</level> | "
        + "<light-black>{file.path}:{line}</light-black> | "
        + "{message}",
    )

    # Parse and configure arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    # Initialize  wandb
    run = wandb.init(project="mario", tags=get_tags(opt), config=opt, dir=opt.out)
    opt.out_ = run.dir

    # Initialize game-specific parameters based on the selected game
    replace_tokens = {}
    sprite_path = "games/" + opt.game + "/sprites"
    if opt.game == "mario":
        opt.ImgGen = MarioLevelGen(sprite_path)
        replace_tokens = MARIO_REPLACE_TOKENS
        downsample = special_mario_downsampling
    elif opt.game == "mariokart":
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling
    elif opt.game == "sonic":
        opt.ImgGen = SonicLevelGen(sprite_path)
        replace_tokens = SONIC_REPLACE_TOKENS
        downsample = special_sonic_downsampling
    else:
        NameError("name of --game not recognized. Supported: mario, mariokart, sonic")

    # Read level according to input arguments
    real = read_level(opt, None, replace_tokens).to(opt.device)

    # Train!
    generators, noise_maps, reals, noise_amplitudes = train(real, opt)

    # Generate Samples of same size as level
    logger.info("Finished training! Generating random samples...")
    in_s = None
    generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s)

    # Generate samples of smaller size than level
    logger.info("Generating arbitrary sized random samples...")
    scale_v = 0.8  # Arbitrarily chosen scales
    scale_h = 0.4
    real_down = downsample(1, [[scale_v, scale_h]], real, opt.token_list)
    real_down = real_down[0]
    # necessary for correct input shape
    in_s = torch.zeros(real_down.shape, device=opt.device)
    generate_samples(
        generators,
        noise_maps,
        reals,
        noise_amplitudes,
        opt,
        in_s=in_s,
        scale_v=scale_v,
        scale_h=scale_h,
        save_dir="arbitrary_random_samples",
    )


if __name__ == "__main__":
    main()
