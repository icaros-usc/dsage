"""Networks for MarioEmulationModel."""
from typing import Tuple

import gin
import torch

from src.mario.level import str2index
from src.maze.emulation_model.networks import MazeConvolutional, \
    MazeAugResnetOccupancy
from src.utils.network import int_preprocess


@gin.configurable
class MarioConvolutional(MazeConvolutional):
    """Conv network for predicting objective, measures on Mario levels."""

    def predict_objs_and_measures(
            self,
            lvls: torch.Tensor,
            aug_lvls: torch.Tensor = None) -> Tuple[torch.Tensor]:
        inputs = int_preprocess(lvls, self.i_size, self.nc, str2index["-"])
        if aug_lvls is not None:
            inputs[:, -aug_lvls.shape[1]:, ...] = aug_lvls
        return self(inputs)


@gin.configurable
class MarioAugResnetOccupancy(MazeAugResnetOccupancy):
    """Resnet for predicting the agent cell occupancy on Mario levels."""

    def int_to_logits(self, lvls: torch.Tensor) -> torch.Tensor:
        _, lvl_height, lvl_width = lvls.shape
        outputs = self.int_to_no_crop(lvls)
        return outputs[:, :, :lvl_height, :lvl_width]

    def int_to_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        inputs = int_preprocess(lvls, self.i_size, self.nc, str2index["-"])
        return self(inputs)
