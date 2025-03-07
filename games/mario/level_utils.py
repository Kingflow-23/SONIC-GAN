import torch
from loguru import logger

from .tokens import TOKEN_GROUPS, REPLACE_TOKENS


# Miscellaneous functions to deal with ascii-token-based levels.


def group_to_token(tensor, tokens, token_groups=TOKEN_GROUPS):
    """
    Converts a token group level tensor back to a full token level tensor.

    Args:
        tensor (torch.Tensor): The input tensor representing grouped tokens.
        tokens (list): The list of tokens corresponding to the tensor dimensions.
        token_groups (list of lists): Groups of tokens that are combined in the input tensor.

    Returns:
        torch.Tensor: A new tensor representing the full token level.
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


def token_to_group(tensor, tokens, token_groups=TOKEN_GROUPS):
    """
    Converts a full token tensor to a token group tensor.

    Args:
        tensor (torch.Tensor): The input tensor representing full tokens.
        tokens (list): The list of tokens corresponding to the tensor dimensions.
        token_groups (list of lists): Groups of tokens that the output tensor will represent.

    Returns:
        torch.Tensor: A new tensor representing the grouped token level.
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
    Loads an ASCII level from a text file, replacing specified tokens as necessary.

    Args:
        path_to_level_txt (str): Path to the text file containing the level.
        replace_tokens (dict): Dictionary of tokens to be replaced and their replacements.

    Returns:
        list: A list of strings representing the level in ASCII.
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
    Converts an ASCII level to a one-hot encoded tensor.

    Args:
        level (list): List of strings representing the level in ASCII.
        tokens (list): List of tokens to be one-hot encoded.

    Returns:
        torch.Tensor: One-hot encoded tensor of the level.
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
    Converts a one-hot encoded level tensor back to an ASCII representation.

    Args:
        level (torch.Tensor): One-hot encoded tensor of the level.
        tokens (list): List of tokens corresponding to the one-hot encoding.

    Returns:
        list: A list of strings representing the level in ASCII.
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
    Wrapper function for reading a level using specified options.

    Args:
        opt (namespace): Namespace containing input options such as input_dir and input_name.
        tokens (list, optional): Predefined list of tokens for reading the level. If None, it is computed.
        replace_tokens (dict): Dictionary of tokens to be replaced and their replacements.

    Returns:
        torch.Tensor: One-hot encoded tensor of the level.
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
    Reads a level from a .txt file and returns it as a one-hot encoded tensor.

    Args:
        input_dir (str): Directory containing the level file.
        input_name (str): Name of the level file.
        tokens (list, optional): Predefined list of tokens. If None, it is computed from the level.
        replace_tokens (dict): Dictionary of tokens to be replaced and their replacements.

    Returns:
        tuple: A tuple containing the one-hot encoded tensor and a list of unique tokens found in the level.
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


def place_a_mario_token(level):
    """
    Places the 'M' token representing Mario at the first plausible position in the level.

    Args:
        level (list): List of strings representing the level in ASCII.

    Returns:
        list: Modified level with Mario placed in a suitable position.
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
            tmp_slice[j] = "M"
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == "-" and level[i][j] in [
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
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = "M"
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario
