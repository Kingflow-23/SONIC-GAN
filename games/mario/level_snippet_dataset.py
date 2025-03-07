import math
import os

import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm

from .level_utils import load_level_from_text, ascii_to_one_hot_level


class LevelSnippetDataset(Dataset):
    """
    Converts a folder containing ASCII-based level representations into a PyTorch Dataset.
    Each level is divided into slices of a specified width to be used in machine learning models.

    Attributes:
        level_idx (optional int): If None, returns the actual index with the retrieved slice.
                                  Otherwise, uses the provided index.
        ascii_levels (list): List of levels in ASCII representation.
        token_list (list): List of unique tokens found in the levels.
        level_names (list): List of the level filenames used.
        levels (list of torch.Tensor): List of levels converted to one-hot encoded tensors.
        slice_width (int): Width of each slice extracted from the levels.
        missing_slices_per_level (int): Number of slices that are missing due to slicing.
        missing_slices_l (int): Number of missing slices on the left.
        missing_slices_r (int): Number of missing slices on the right.
        level_lengths (list): List of effective lengths of each level after accounting for missing slices.
    """

    def __init__(
        self,
        level_dir,
        slice_width=16,
        token_list=None,
        level_idx=None,
        level_name=None,
    ):
        """
        Initializes the LevelSnippetDataset by reading levels from a directory, converting them to one-hot encoded tensors,
        and preparing them for slicing.

        Args:
            level_dir (str): Directory containing level .txt files.
            slice_width (int): Width of the slices to be extracted from the levels. Defaults to 16.
            token_list (list, optional): Predefined list of tokens for one-hot encoding.
                                         If None, it is computed from the levels.
            level_idx (int, optional): Predefined index for levels. If None, indices are returned based on the dataset.
            level_name (str, optional): Specific level file to use. If None, all levels in the directory are used.
        """
        super(LevelSnippetDataset, self).__init__()
        self.level_idx = level_idx
        self.ascii_levels = []
        uniques = set()
        self.level_names = []
        logger.debug("Reading levels from directory {}", level_dir)
        for level in tqdm(sorted(os.listdir(level_dir))):
            if not level.endswith(".txt") or (
                level_name is not None and level != level_name
            ):
                continue
            self.level_names.append(level)
            curr_level = load_level_from_text(os.path.join(level_dir, level))
            for line in curr_level:
                for token in line:
                    if token != "\n" and token != "M" and token != "F":
                        # if token != "M" and token != "F":
                        uniques.add(token)
            self.ascii_levels.append(curr_level)

        logger.trace("Levels: {}", self.level_names)
        if token_list is not None:
            self.token_list = token_list
        else:
            self.token_list = list(sorted(uniques))

        logger.trace("Token list: {}", self.token_list)

        logger.debug("Converting ASCII levels to tensors...")
        self.levels = []
        for i, level in tqdm(enumerate(self.ascii_levels)):
            self.levels.append(ascii_to_one_hot_level(level, self.token_list))

        self.slice_width = slice_width
        self.missing_slices_per_level = slice_width - 1
        self.missing_slices_l = math.floor(self.missing_slices_per_level / 2)
        self.missing_slices_r = math.ceil(self.missing_slices_per_level / 2)

        self.level_lengths = [
            x.shape[-1] - self.missing_slices_per_level for x in self.levels
        ]

    def get_level_name(self, file_name):
        """
        Extracts the base name of the level file, removing the file extension.

        Args:
            file_name (str): Name of the level file.

        Returns:
            str: Base name of the file without extension.
        """
        return file_name.split(".")[0]

    def __getitem__(self, idx):
        """
        Retrieves a slice of a level and its corresponding index.

        Args:
            idx (int): Index of the slice to retrieve.

        Returns:
            tuple: A tuple containing the level slice (torch.Tensor) and its index (torch.Tensor).
        """
        i_l = 0
        while sum(self.level_lengths[0:i_l]) < (idx + 1) < sum(self.level_lengths):
            i_l += 1
        i_l -= 1

        level = self.levels[i_l]
        idx_lev = idx - sum(self.level_lengths[0:i_l]) + self.missing_slices_l
        lev_slice = level[
            :, :, idx_lev - self.missing_slices_l : idx_lev + self.missing_slices_r + 1
        ]
        return (
            lev_slice,
            torch.tensor(i_l if self.level_idx is None else self.level_idx),
        )

    def __len__(self):
        """
        Returns the total number of slices available in the dataset.

        Returns:
            int: Total number of slices.
        """
        return sum(self.level_lengths) - 1
