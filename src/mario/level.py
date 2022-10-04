"""Provides the MarioLevel class."""
import json
from pathlib import Path

import numpy as np

with (Path(__file__).parent / "index2str.json").open("r") as f:
    index2str = {int(k): v for k, v in json.load(f).items()}
    str2index = {v: k for k, v in index2str.items()}


class MarioLevel:
    """Utilities for handling Mario levels.

    Args:
        data: 2D numpy array of integers where the integers represent the object
            types. The array must be of shape (lvl_height, lvl_width).
    """

    def __init__(self, data: np.ndarray):
        self.data = data

        # For backwards compatibility with measure_calculate, this is set to a
        # JSON string which stores an integer representation of the level.
        self.level = json.dumps(data.tolist())

    @property
    def lvl_height(self):
        return self.data.shape[0]

    @property
    def lvl_width(self):
        return self.data.shape[1]

    @staticmethod
    def str_to_number(lvl_str):
        """Converts list of strings (each string is a row) to numpy array."""
        np_lvl = np.zeros((len(lvl_str), len(lvl_str[0])), dtype=int)
        for x, row in enumerate(lvl_str):
            row = row.strip()
            for y, tile in enumerate(row):
                np_lvl[x, y] = str2index[tile]
        return np_lvl

    def to_str_grid(self):
        """Converts level to grid of characters."""
        return [[index2str[x] for x in row] for row in self.data]

    def to_str(self):
        """Converts level to single string."""
        return "\n".join(["".join(row) for row in self.to_str_grid()])

    def __str__(self):
        return self.to_str()
