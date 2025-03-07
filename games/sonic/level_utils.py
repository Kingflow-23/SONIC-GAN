import torch
from loguru import logger

from games.sonic.tokens import TOKENS_GROUPS, REPLACE_TOKENS


def group_to_token(tensor, tokens, token_groups=TOKENS_GROUPS):
    """
    Converts a token group tensor back to a full token tensor.

    Args:
        tensor (torch.Tensor): The input tensor with token groups.
        tokens (list): A list of all possible tokens.
        token_groups (list of lists): Groups of related tokens (default is `TOKENS_GROUPS`).

    Returns:
        torch.Tensor: A tensor with the same shape, but with each group replaced by its corresponding token.
    """
    new_tensor = torch.zeros(tensor.shape[0], len(tokens), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, i] = tensor[:, group_idx]
                break
    return new_tensor


def token_to_group(tensor, tokens, token_groups=TOKENS_GROUPS):
    """
    Converts a full token tensor to a token group tensor.

    Args:
        tensor (torch.Tensor): The input tensor with all tokens.
        tokens (list): A list of all possible tokens.
        token_groups (list of lists): Groups of related tokens (default is `TOKENS_GROUPS`).

    Returns:
        torch.Tensor: A tensor with each token replaced by its corresponding token group.
    """
    new_tensor = torch.zeros(tensor.shape[0], len(token_groups), *tensor.shape[2:]).to(
        tensor.device
    )
    for i, token in enumerate(tokens):
        for group_idx, group in enumerate(token_groups):
            if token in group:
                new_tensor[:, group_idx] += tensor[:, i]
                break
    return new_tensor


def load_level_from_text(path_to_level_txt, replace_tokens=REPLACE_TOKENS):
    """
    Loads an ascii level from a text file and replaces tokens as specified.

    Args:
        path_to_level_txt (str): The path to the level text file.
        replace_tokens (dict): A dictionary of tokens to replace in the file (default is `REPLACE_TOKENS`).

    Returns:
        list: A list of strings representing the level, with the tokens replaced.
    """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in replace_tokens.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def ascii_to_one_hot_level(level, tokens):
    """
    Converts an ASCII level into a one-hot encoded tensor.

    Args:
        level (list): The level as a list of strings (ASCII format).
        tokens (list): A list of all possible tokens.

    Returns:
        torch.Tensor: A one-hot encoded tensor with shape (num_tokens, height, width).
    """
    oh_level = torch.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[tokens.index(token), i, j] = 1
    return oh_level


def one_hot_to_ascii_level(level, tokens):
    """
    Converts a one-hot encoded tensor back into an ASCII level.

    Args:
        level (torch.Tensor): A one-hot encoded tensor.
        tokens (list): A list of all possible tokens.

    Returns:
        list: A list of strings representing the level in ASCII format.
    """
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level


def read_level(opt, tokens=None, replace_tokens=REPLACE_TOKENS):
    """
    Reads a level from a file and updates the options object with the token list.

    Args:
        opt (object): The options object, which contains input file details.
        tokens (list, optional): A list of tokens (default is `None`).
        replace_tokens (dict, optional): A dictionary of tokens to replace in the file (default is `REPLACE_TOKENS`).

    Returns:
        list: The level as a one-hot encoded tensor.
    """
    level, uniques = read_level_from_file(
        opt.input_dir, opt.input_name, tokens, replace_tokens
    )
    opt.token_list = uniques
    logger.info("Tokens in level {}", opt.token_list)
    opt.nc_current = len(uniques)
    return level


def read_level_from_file(
    input_dir, input_name, tokens=None, replace_tokens=REPLACE_TOKENS
):
    """
    Loads a level from a file and returns a one-hot encoded tensor and unique tokens.

    Args:
        input_dir (str): The directory where the level file is located.
        input_name (str): The name of the level file.
        tokens (list, optional): A list of tokens (default is `None`).
        replace_tokens (dict, optional): A dictionary of tokens to replace in the file (default is `REPLACE_TOKENS`).

    Returns:
        tuple: A tuple containing a one-hot encoded tensor and a list of unique tokens found in the level.
    """
    txt_level = load_level_from_text("%s/%s" % (input_dir, input_name), replace_tokens)
    uniques = set()
    for line in txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in replace_tokens.items():
                uniques.add(token)
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = ascii_to_one_hot_level(txt_level, uniques if tokens is None else tokens)
    return oh_level.unsqueeze(dim=0), uniques


def place_a_sonic_token(level):
    """
    Places Sonic at the first valid position in the level.

    Searches the level for a suitable spot for Sonic ('@') to be placed. It first checks predefined positions,
    then looks from left to right for the first available spot.

    Args:
        level (list): The level as a list of strings (ASCII format).

    Returns:
        list: The level with Sonic placed at a valid position.
    """
    # First check if default spot is available
    for j in range(1, 4):
        if level[-3][j] == "-" and level[-2][j] in [
            "X",
            "#",
            "S",
            "%",
            "t",
            "?",
            "@",
            "!",
            "C",
            "D",
            "U",
            "L",
        ]:
            tmp_slice = list(level[-3])
            tmp_slice[j] = "@"
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == "-" and level[i][j] in ["#", "&", "/", "!"]:
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = "@"
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario
