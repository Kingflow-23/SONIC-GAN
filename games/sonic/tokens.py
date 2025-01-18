from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.

GROUND_TOKENS = OrderedDict({"#": "Base Ground Block", "&": "Flat Ground"})


SPECIAL_GROUND_TOKENS = OrderedDict(
    {
        "o": "Loop Ground",
        "/": "Curved ground",
    }
)


SKY_TOKENS = OrderedDict(
    {
        "-": "Empty",
    }
)

ENEMY_TOKENS = OrderedDict(
    {"M": "Moto Bug", "B": "Buzz Bomber", "R": "Crabmeat", "^": "Spkikes"}
)

SPECIAL_TOKENS = OrderedDict(
    {
        "O": "Ring",
        "@": "Sonic Starting Position",
        "P": "Exit Panel. Sonic Finish line.",
        "I": "Item Box",
    }
)

EXTRA_SPECIAL_TOKENS = OrderedDict(
    {
        "|": "Palm Tree",
        "F": "Simple Flower",
        "f": "Little Flower",
    }
)

TOKEN_DOWNSAMPLING_HIERARCHY = [
    SKY_TOKENS,
    GROUND_TOKENS,
    SPECIAL_GROUND_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_TOKENS,
    EXTRA_SPECIAL_TOKENS,
]

TOKENS = OrderedDict(
    {
        **GROUND_TOKENS,
        **SPECIAL_GROUND_TOKENS,
        **SKY_TOKENS,
        **ENEMY_TOKENS,
        **SPECIAL_TOKENS,
        **EXTRA_SPECIAL_TOKENS,
    }
)

TOKENS_GROUPS = [
    SKY_TOKENS,
    GROUND_TOKENS,
    SPECIAL_GROUND_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_TOKENS,
    EXTRA_SPECIAL_TOKENS,
]

REPLACE_TOKENS = {"@": "-", "P": "-"}
