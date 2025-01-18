import os
from PIL import Image


class LevelImageGen:
    """
    Generates level images from ASCII maps for Sonic levels.
    """

    def __init__(self, sprite_path):
        """
        Initializes the LevelImageGen object and loads the necessary sprites.

        Args:
            sprite_path (str): Path to the folder containing sprite sheets.
        """
        self.sprite_dict = self.load_sprites(sprite_path)

    def load_sprites(self, sprite_path):
        """
        Loads and crops all necessary sprites for the Sonic level generation.

        Args:
            sprite_path (str): Path to the folder containing sprite sheets.

        Returns:
            dict: A dictionary of loaded and resized sprites keyed by their names.
        """
        sheets = {
            "sonic": [
                Image.open(os.path.join(sprite_path, r"sonic\Sonic.png")).convert(
                    "RGBA"
                )
            ],
            "enemies": [
                Image.open(
                    os.path.join(sprite_path, r"enemies\buzz_bomber.png")
                ).convert("RGBA"),
                Image.open(
                    os.path.join(sprite_path, r"enemies\enemycube_face_1.png")
                ).convert("RGBA"),
                Image.open(
                    os.path.join(sprite_path, r"enemies\enemycube_face_2.png")
                ).convert("RGBA"),
                Image.open(
                    os.path.join(sprite_path, r"enemies\enemycube_face_3.png")
                ).convert("RGBA"),
                Image.open(os.path.join(sprite_path, r"enemies\Traps.png")).convert(
                    "RGBA"
                ),
            ],
            "snail": [
                Image.open(os.path.join(sprite_path, r"enemies\snails.png")).convert(
                    "RGBA"
                )
            ],
            "trap": [
                Image.open(os.path.join(sprite_path, r"enemies\Traps.png")).convert(
                    "RGBA"
                )
            ],
            "floor": [
                Image.open(os.path.join(sprite_path, r"floor\floor_1.png")),
                Image.open(os.path.join(sprite_path, r"floor\floor_2.png")),
            ],
            "flower": [
                Image.open(os.path.join(sprite_path, r"flower\flower_body.png")),
                Image.open(os.path.join(sprite_path, r"flower\flower_head_1.png")),
                Image.open(os.path.join(sprite_path, r"flower\flower_head_2.png")),
            ],
            "ground_block": [
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_bl.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_bm.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_br.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_ml.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_mm.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_mr.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_tr.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_tl.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"ground_block\ground_block_tm.png")
                ),
            ],
            "curved_ground": [
                Image.open(
                    os.path.join(sprite_path, r"curved_ground\curved_ground_tr.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"curved_ground\curved_ground_tm.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"curved_ground\curved_ground_br.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"curved_ground\curved_ground_bm.png")
                ),
                Image.open(
                    os.path.join(sprite_path, r"curved_ground\curved_ground_bl.png")
                ),
            ],
            "stage_block": [
                Image.open(os.path.join(sprite_path, r"stage_block\stage_block_l.png")),
                Image.open(
                    os.path.join(sprite_path, r"stage_block\stage_block_mid.png")
                ),
                Image.open(os.path.join(sprite_path, r"stage_block\stage_block_r.png")),
            ],
            "pipe": [Image.open(os.path.join(sprite_path, r"objects\pipe.png"))],
            "ring": [Image.open(os.path.join(sprite_path, r"objects\Ring.png"))],
            "rock": [Image.open(os.path.join(sprite_path, r"objects\rock.png"))],
            "tree": [
                Image.open(os.path.join(sprite_path, r"tree\tree_bm.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_ml.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_mm.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_mr.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_tl.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_tm.png")),
                Image.open(os.path.join(sprite_path, r"tree\tree_tr.png")),
            ],
            "Goalpost": [
                Image.open(os.path.join(sprite_path, r"goalpost\Goalpost.png"))
            ],
            "background": [
                Image.open(
                    os.path.join(sprite_path, r"background\tiled_background.png")
                ),
                Image.open(os.path.join(sprite_path, r"background\background.jpg")),
            ],
        }

        sprite_dict = {
            "Sonic": sheets["sonic"][0]
            .crop((0, 0, 40, 40))
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_bl": sheets["ground_block"][0]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_bm": sheets["ground_block"][1]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_br": sheets["ground_block"][2]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_ml": sheets["ground_block"][3]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_mm": sheets["ground_block"][4]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_mr": sheets["ground_block"][5]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_tr": sheets["ground_block"][6]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_tl": sheets["ground_block"][7]
            .resize((32, 32))
            .convert("RGBA"),
            "ground_block_tm": sheets["ground_block"][8]
            .resize((32, 32))
            .convert("RGBA"),
            "stage_block_l": sheets["stage_block"][0].resize((32, 32)).convert("RGBA"),
            "stage_block_m": sheets["stage_block"][1].resize((32, 32)).convert("RGBA"),
            "stage_block_r": sheets["stage_block"][2].resize((32, 32)).convert("RGBA"),
            "curved_ground_tr": sheets["curved_ground"][0]
            .resize((32, 32))
            .convert("RGBA"),
            "curved_ground_tm": sheets["curved_ground"][1]
            .resize((32, 32))
            .convert("RGBA"),
            "curved_ground_br": sheets["curved_ground"][2]
            .resize((32, 32))
            .convert("RGBA"),
            "curved_ground_bm": sheets["curved_ground"][3]
            .resize((32, 32))
            .convert("RGBA"),
            "curved_ground_bl": sheets["curved_ground"][4]
            .resize((32, 32))
            .convert("RGBA"),
            "flower_body": sheets["flower"][0].resize((32, 32)).convert("RGBA"),
            "flower_head_1": sheets["flower"][1].resize((32, 32)).convert("RGBA"),
            "flower_head_2": sheets["flower"][2].resize((32, 32)).convert("RGBA"),
            "buzz_bomber": sheets["enemies"][0].resize((32, 32)).convert("RGBA"),
            "trap": sheets["enemies"][4]
            .crop((0, 0, 50, 50))
            .resize((32, 32))
            .convert("RGBA"),
            "enemycube_face_1": sheets["enemies"][1].resize((32, 32)).convert("RGBA"),
            "enemycube_face_2": sheets["enemies"][2].resize((32, 32)).convert("RGBA"),
            "enemycube_face_3": sheets["enemies"][3].resize((32, 32)).convert("RGBA"),
            "ring": sheets["ring"][0]
            .crop((0, 0, 50, 50))
            .resize((32, 32))
            .convert("RGBA"),
            "tree_bm": sheets["tree"][0].resize((32, 32)).convert("RGBA"),
            "tree_ml": sheets["tree"][1].resize((32, 32)).convert("RGBA"),
            "tree_mm": sheets["tree"][2].resize((32, 32)).convert("RGBA"),
            "tree_mr": sheets["tree"][3].resize((32, 32)).convert("RGBA"),
            "tree_tl": sheets["tree"][4].resize((32, 32)).convert("RGBA"),
            "tree_tm": sheets["tree"][5].resize((32, 32)).convert("RGBA"),
            "tree_tr": sheets["tree"][6].resize((32, 32)).convert("RGBA"),
            "background_1": sheets["background"][0].resize((32, 32)).convert("RGBA"),
            "background_2": sheets["background"][1]
            .crop((425, 196, 468, 224))
            .resize((32, 32))
            .convert("RGBA"),
            "Goal Post": sheets["Goalpost"][0]
            .crop((0, 0, 50, 50))
            .resize((32, 32))
            .convert("RGBA"),
        }

        return sprite_dict

    def prepare_sprite_and_box(self, ascii_level, sprite_key, curr_x, curr_y):
        """
        Prepares the sprite and the box for pasting in the level image based on the ASCII map.

        Args:
            ascii_level (list): 2D list representing the ASCII map of the level.
            sprite_key (str): Key that determines which sprite to use.
            curr_x (int): The current x-coordinate of the tile in the ASCII map.
            curr_y (int): The current y-coordinate of the tile in the ASCII map.

        Returns:
            tuple: A tuple containing the sprite (PIL Image) and the bounding box (tuple).
        """

        def has_three_ground_blocks_below(ascii_level, curr_x, curr_y):
            """
            Checks if there are at least 3 ground blocks directly below the given position.
            Args:
                ascii_level: 2D list of the ASCII level.
                curr_x: X-coordinate of the current position.
                curr_y: Y-coordinate of the current position.

            Returns:
                bool: True if there are at least 3 ground blocks directly below, False otherwise.
            """
            if curr_y + 1 >= len(ascii_level):
                return False  # No row below the current one

            row_below = ascii_level[curr_y + 1]

            return (
                curr_x - 1 >= 0
                and row_below[curr_x - 1] == "#"
                and row_below[curr_x] == "#"
                and curr_x + 1 < len(row_below)
                and row_below[curr_x + 1] == "#"
            )

        box = ()
        sprite = None
        b_multiplier = 256

        if sprite_key not in [
            "#",
            "&",
            "F",
            "f",
            "|",
            "P",
            "O",
            "B",
            "^",
            "-",
            "@",
            "/",
        ]:
            return None, None

        # Handling sorite depending on their type and logical problems

        elif sprite_key == "#":

            if curr_x == 0 or ascii_level[curr_y][curr_x - 1] != "#":
                sprite = self.sprite_dict["stage_block_l"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            elif (
                curr_x == len(ascii_level[curr_y]) - 1
                or ascii_level[curr_y][curr_x + 1] != "#"
            ):
                sprite = self.sprite_dict["stage_block_r"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            else:
                sprite = self.sprite_dict["stage_block_m"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "|":

            if (curr_y + 1 < len(ascii_level)) and (
                ascii_level[curr_y + 1][curr_x] in ["#", "&"]
            ):

                if curr_y - 1 >= 0 and ascii_level[curr_y - 1][curr_x] == "-":

                    sprite = Image.new("RGBA", (3 * 32, 3 * 32))
                    sprite.paste(self.sprite_dict["tree_tl"], (0, 0))
                    sprite.paste(self.sprite_dict["tree_tm"], (32, 0))
                    sprite.paste(self.sprite_dict["tree_tr"], (64, 0))
                    sprite.paste(self.sprite_dict["tree_ml"], (0, 32))
                    sprite.paste(self.sprite_dict["tree_mm"], (32, 32))
                    sprite.paste(self.sprite_dict["tree_tl"], (64, 32))
                    sprite.paste(self.sprite_dict["tree_bm"], (32, 64))
                    sprite = sprite.resize((b_multiplier, b_multiplier))

                    left = curr_x * b_multiplier
                    top = curr_y * b_multiplier
                    new_left = left + b_multiplier
                    new_top = top + b_multiplier
                    box = (left, top, new_left, new_top)

        elif sprite_key == "F":

            if (
                (curr_y > 0 and ascii_level[curr_y - 1][curr_x] in ["#", "&"])
                or (
                    curr_y < len(ascii_level) - 1
                    and ascii_level[curr_y + 1][curr_x] in ["#", "&"]
                )
                or (curr_x > 0 and ascii_level[curr_y][curr_x - 1] in ["#", "&"])
                or (
                    curr_x < len(ascii_level[curr_y]) - 1
                    and ascii_level[curr_y][curr_x + 1] in ["#", "&"]
                )
            ):

                if (
                    ascii_level[curr_y - 1][curr_x] == "#"
                    or ascii_level[curr_y - 1][curr_x] == "&"
                ):
                    sprite = Image.new("RGBA", (32, 2 * 32))
                    sprite.paste(self.sprite_dict["flower_body"], (0, 32))
                    sprite.paste(self.sprite_dict["flower_head_1"], (0, 0))
                    sprite = sprite.resize((b_multiplier, b_multiplier))

            else:
                sprite = self.sprite_dict["background_1"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "f":

            if (curr_y + 1 < len(ascii_level)) and (
                ascii_level[curr_y + 1][curr_x] in ["#", "&"]
            ):

                if curr_y - 1 >= 0 and ascii_level[curr_y - 1][curr_x] == "-":

                    sprite = Image.new("RGBA", (32, 2 * 32))
                    sprite.paste(self.sprite_dict["flower_body"], (0, 32))
                    sprite.paste(self.sprite_dict["flower_head_2"], (0, 0))
                    sprite = sprite.resize(
                        (b_multiplier, b_multiplier)
                    )  # Resize to the appropriate size

                # Position for pasting the sprite in the level image
                left = curr_x * b_multiplier
                top = curr_y * b_multiplier
                new_left = left + b_multiplier
                new_top = top + b_multiplier
                box = (left, top, new_left, new_top)

            else:
                sprite = self.sprite_dict["background_1"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "P":  # Goal post
            sprite = self.sprite_dict["Goal Post"]
            sprite = sprite.resize((b_multiplier, b_multiplier))
            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "O":  # Ring
            sprite = self.sprite_dict["ring"]
            sprite = sprite.resize((b_multiplier, b_multiplier))
            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "^":  # Sprikes/Basic trap

            if (curr_y + 1 < len(ascii_level)) and (
                ascii_level[curr_y + 1][curr_x] in ["#", "&"]
            ):
                sprite = self.sprite_dict["trap"]
                sprite = sprite.resize((b_multiplier, b_multiplier))

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "B":  # Basic enemy / Buzz bomber
            sprite = self.sprite_dict["buzz_bomber"]
            sprite = sprite.resize((b_multiplier, b_multiplier))
            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "-":  # Empty tile
            sprite = self.sprite_dict["background_1"]
            sprite = sprite.resize((b_multiplier, b_multiplier))
            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "@":  # Sonic sprite
            sprite = self.sprite_dict["Sonic"]
            sprite = sprite.resize((b_multiplier, b_multiplier))
            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)

        elif sprite_key == "&":  # Flat ground / floating ground
            sprite = Image.new("RGBA", (3 * 32, 3 * 32))
            sprite.paste(self.sprite_dict["ground_block_tl"], (0, 0))
            sprite.paste(self.sprite_dict["ground_block_tm"], (32, 0))
            sprite.paste(self.sprite_dict["ground_block_tr"], (64, 0))
            sprite.paste(self.sprite_dict["ground_block_ml"], (0, 32))
            sprite.paste(self.sprite_dict["ground_block_mm"], (32, 32))
            sprite.paste(self.sprite_dict["ground_block_mr"], (64, 32))
            sprite.paste(self.sprite_dict["ground_block_bl"], (0, 64))
            sprite.paste(self.sprite_dict["ground_block_bm"], (32, 64))
            sprite.paste(self.sprite_dict["ground_block_br"], (64, 64))

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            new_left = left + b_multiplier
            new_top = top + b_multiplier
            box = (left, top, new_left, new_top)
            sprite = sprite.resize((b_multiplier, b_multiplier))

        elif sprite_key == "/":  # Curved ground
            sprite = Image.new("RGBA", (3 * 32, 2 * 32))

            if has_three_ground_blocks_below(ascii_level, curr_x, curr_y):
                sprite.paste(self.sprite_dict["curved_ground_tr"], (64, 0))
                sprite.paste(self.sprite_dict["curved_ground_tm"], (32, 0))
                sprite.paste(self.sprite_dict["curved_ground_bl"], (0, 32))
                sprite.paste(self.sprite_dict["curved_ground_bm"], (32, 32))
                sprite.paste(self.sprite_dict["curved_ground_br"], (64, 32))

            else:
                sprite = self.sprite_dict["background_1"].resize(
                    (b_multiplier, b_multiplier)
                )

            left = curr_x * b_multiplier
            top = curr_y * b_multiplier
            box = (left, top, left + b_multiplier, top + b_multiplier)

            sprite = sprite.resize((b_multiplier, b_multiplier))

        return sprite, box

    def render(self, ascii_level):
        """
        Generates the full level image from the ASCII map.

        Args:
            ascii_level (list): 2D list representing the ASCII map of the level.

        Returns:
            Image: The generated level image as a PIL Image.
        """
        b_multiplier = 256
        len_level = len(ascii_level[0])
        height_level = len(ascii_level)
        dst = Image.new("RGBA", (len_level * b_multiplier, height_level * b_multiplier))

        # Render background first
        background = self.sprite_dict["background_1"].resize(
            (b_multiplier, b_multiplier)
        )
        for y in range(height_level):
            for x in range(len_level):
                dst.paste(
                    background,
                    (
                        x * b_multiplier,
                        y * b_multiplier,
                        (x + 1) * b_multiplier,
                        (y + 1) * b_multiplier,
                    ),
                )

        # Render each sprite based on ASCII map
        for y, row in enumerate(ascii_level):
            for x, char in enumerate(row):
                sprite_key = char
                if sprite_key:
                    sprite, box = self.prepare_sprite_and_box(
                        ascii_level, sprite_key, x, y
                    )
                    if sprite and box:
                        dst.paste(
                            sprite, box, mask=sprite
                        )  # Use sprite as its own mask

        return dst
