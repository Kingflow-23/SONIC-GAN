import os

from PIL import Image, ImageOps, ImageEnhance


class LevelImageGen:
    """
    Generates PIL Image files from Mario Kart ASCII levels.

    This class initializes with a path to sprite files and provides the `render` method to convert ASCII levels into
    visual representations using the specified sprites.
    """

    def __init__(self, sprite_path):
        """
        Initializes the LevelImageGen with the given sprite path.

        Args:
            sprite_path (str): Path to the folder containing sprite files, e.g., 'mariokart/sprites/'.

        This method loads the sprite sheet and slices it into individual sprite images stored in a dictionary.
        """

        # Load Graphics
        mapsheet = Image.open(os.path.join(sprite_path, "mapsheet.png"))

        # Cut out the actual sprites:
        sprite_dict = dict()

        # Map Sheet
        map_names = [
            "W_r",
            "W_g",
            "W_y",
            "W_b",
            "W_0",
            "-",
            "R",
            "S",
            "C",
            "empty",
            "O_ul",
            "O_ur",
            "Q_ul",
            "Q_ur",
            "Q",
            "O_dl",
            "O_dr",
            "Q_dl",
            "Q_dr",
            "O",
            "<_ul",
            "<_ur",
            "<",
            "empty",
            "empty",
            "<_dl",
            "<_dr",
            "empty",
            "empty",
            "empty",
        ]

        sheet_length = (6, 5)
        sprite_counter = 0
        for i in range(sheet_length[0]):
            for j in range(sheet_length[1]):
                sprite_dict[map_names[sprite_counter]] = mapsheet.crop(
                    (j * 8, i * 8, (j + 1) * 8, (i + 1) * 8)
                )
                sprite_counter += 1

        self.sprite_dict = sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """
        Prepares the sprite and bounding box for a specific position in the ASCII level.

        Args:
            ascii_level (list of str): The ASCII representation of the level.
            sprite_key (str): The key identifying the sprite to be placed.
            curr_x (int): The current x-coordinate in the level.
            curr_y (int): The current y-coordinate in the level.

        Returns:
            (Image.Image, tuple): The sprite image and the bounding box tuple (left, top, right, bottom).

        Handles special cases for sprites that span multiple tiles.
        """

        # Init default size
        new_left = curr_x * 8
        new_top = curr_y * 8
        new_right = (curr_x + 1) * 8
        new_bottom = (curr_y + 1) * 8

        # Handle sprites depending on their type:
        if sprite_key in ["O", "Q", "<"]:
            if curr_x > 0 and ascii_level[curr_y][curr_x - 1] == sprite_key:
                if curr_y > 0 and ascii_level[curr_y - 1][curr_x] == sprite_key:
                    if ascii_level[curr_y - 1][curr_x - 1] == sprite_key:
                        # 4 Sprites of the same type! use big sprite
                        new_left -= 8
                        new_top -= 8
                        actual_sprite = Image.new("RGBA", (2 * 8, 2 * 8))
                        actual_sprite.paste(
                            self.sprite_dict[sprite_key + "_ul"], (0, 0, 8, 8)
                        )
                        actual_sprite.paste(
                            self.sprite_dict[sprite_key + "_ur"], (8, 0, 2 * 8, 8)
                        )
                        actual_sprite.paste(
                            self.sprite_dict[sprite_key + "_dl"], (0, 8, 8, 2 * 8)
                        )
                        actual_sprite.paste(
                            self.sprite_dict[sprite_key + "_dr"], (8, 8, 2 * 8, 2 * 8)
                        )
                        return actual_sprite, (new_left, new_top, new_right, new_bottom)

            actual_sprite = self.sprite_dict[sprite_key]

        elif sprite_key == "W":
            walls = [
                ["W_r", "W_g", "W_y", "W_b"],
                ["W_g", "W_y", "W_b", "W_r"],
                ["W_y", "W_b", "W_r", "W_g"],
                ["W_b", "W_r", "W_g", "W_y"],
            ]
            curr_col = curr_x % 16
            curr_row = curr_y % 16
            w_col = curr_col // 4
            w_row = curr_row // 4
            actual_sprite = self.sprite_dict[walls[w_col][w_row]]

        else:
            actual_sprite = self.sprite_dict[sprite_key]

        return actual_sprite, (new_left, new_top, new_right, new_bottom)

    def render(self, ascii_level):
        """
        Renders the ASCII level as a PIL Image.

        Args:
            ascii_level (list of str): The ASCII representation of the level to be rendered.

        Returns:
            Image.Image: The rendered image of the level.

        This method creates a new image filled with the sky tile and overlays the appropriate sprites based on the level.
        """
        len_level = len(ascii_level[-1])
        height_level = len(ascii_level)

        # Fill base image with sky tiles
        dst = Image.new("RGB", (len_level * 8, height_level * 8))
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(
                    self.sprite_dict["-"], (x * 8, y * 8, (x + 1) * 8, (y + 1) * 8)
                )

        # Fill with actual tiles
        for y in range(height_level):
            for x in range(len_level):
                curr_sprite = ascii_level[y][x]
                sprite, box = self.prepare_sprite_and_box(
                    ascii_level, curr_sprite, x, y
                )
                dst.paste(sprite, box, mask=sprite)

        return dst
