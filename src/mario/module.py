"""MarioConfig and MarioModule.

Usage:
    # Run as a script to demo the MarioModule.
    python -m src.mario.module
"""
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Optional

import fire
import gin
import numpy as np

from src.mario import measure_calculate
from src.utils.logging import setup_logging

from .level import MarioLevel
from .mario_result import MarioResult

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class MarioConfig:
    """Config for Mario."""

    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = "mean"


class MarioModule:
    """Module for Mario."""

    # The objective is the completion ratio / percentage, which ranges 0-1.
    MIN_SCORE = 0.0
    MAX_SCORE = 1.0

    def __init__(self, config: MarioConfig):
        self.config = config
        self.measure_funcs = [
            getattr(measure_calculate, name)
            for name in self.config.measure_names
        ]

    @staticmethod
    def calc_occupancy_grid(level: MarioLevel, game_result):
        """Creates an occupancy grid based on the game result."""
        occupancy_grid = np.zeros_like(level.data, dtype=int)

        agent_events = game_result.getAgentEvents()

        # Convert from Java ArrayList to Python list.
        agent_events = [agent_events.get(i) for i in range(agent_events.size())]

        # Each event is a MarioAgentEvent
        # https://github.com/amidos2006/Mario-AI-Framework/blob/master/src/engine/core/MarioAgentEvent.java
        for e in agent_events:
            # 16 seems to be the width and height of all tiles, judging by the
            # code for MarioWorld.
            # https://github.com/amidos2006/Mario-AI-Framework/blob/master/src/engine/core/MarioWorld.java
            x = int(e.getMarioX() / 16.0)
            y = int(e.getMarioY() / 16.0)

            # Only set the occupancy grid if we are within bounds.
            if (0 <= y < occupancy_grid.shape[0] and
                    0 <= x < occupancy_grid.shape[1]):
                occupancy_grid[y, x] += 1

        return occupancy_grid

    def evaluate(
            self,
            level: np.ndarray,
            n_evals: int,
            render: Optional[bool] = False,
            img_name: Optional[callable] = None,
            seed: Optional[int] = None,  # pylint: disable = unused-argument
    ):
        """Evaluates the solution.

        Args:
            level: Integer array with shape (lvl_height, lvl_width)
                returned by gan.MarioGenerator.levels_from_latent()
            n_evals: Number of repetitions to aggregate over. Should be 1 if
                only using representation-level measures (e.g. count the number
                of enemies in a level) since these depend only on the generated
                environment. Otherwise, for agent-level measures, this should be
                greater than 1 since the agent and environment dynamics are
                stochastic.
            render: True if the env should be rendered
            img_name (callable): If passed in, this should be a callable that
                takes in a timestep and outputs the filename for an image. An
                image of the env will then be saved to this file. Note: `render`
                must be True for this callable to be used.
            seed: Seed for the evaluation. Note that the Mario simulator uses
                its own stochasticity, and we do not know how to control it, so
                passing the same seed will likely not lead to the same results.
        Returns:
            ObjectiveResult with n_evals solutions.
        """

        # Set up jnius classes. Note that when we put this at the top of the
        # file, it seems to result in core dump errors from jnius, perhaps
        # because we're trying to run this on the workers and Dask tries to
        # capture the global variables, including `autoclass`.
        os.environ['CLASSPATH'] = str(Path(__file__).parent / "Mario.jar")
        # jnius must be imported _after_ CLASSPATH is set, so it picks up
        # Mario.jar.
        from jnius import \
            autoclass  # pylint: disable = wrong-import-order, wrong-import-position, import-outside-toplevel

        # Level must be converted to JSON string for compatibility with
        # functions in measure_calculate.
        level = MarioLevel(level)

        objs = []
        measures = []
        all_info = []
        occupancy_grids = []
        for _ in range(n_evals):
            # Since we use a lot of Java things here, disable naming checks.
            # pylint: disable = invalid-name

            # Adapted from
            # https://github.com/icaros-usc/MarioGAN-LSI/blob/master/search/search.py#L61
            real_level = level.to_str()
            JString = autoclass("java.lang.String")
            agent = autoclass('agents.robinBaumgarten.Agent')()
            game = autoclass('engine.core.MarioGame')()

            # Note this requires having the Mario images available in an `img/`
            # directory next to where the code is run. We achieve this by
            # symlinking `img/` at the root of this repo.
            #
            # See here for function definition
            # https://github.com/amidos2006/Mario-AI-Framework/blob/510dba06cf2423283d7c0328e528b4bcb72cc133/src/engine/core/MarioGame.java#L175
            #
            # All versions of that function ultimately call this one
            # https://github.com/amidos2006/Mario-AI-Framework/blob/510dba06cf2423283d7c0328e528b4bcb72cc133/src/engine/core/MarioGame.java#L206
            game_result = game.runGame(
                agent,
                JString(real_level),
                20,  # 20 ticks for timer.
                0,  # Initial state for Mario - 0 for small mario.
                render,  # Whether to visualize the level.
            )

            occupancy_grid = self.calc_occupancy_grid(level, game_result)
            occupancy_grids.append(occupancy_grid)

            all_info.append({
                "CompletionPercentage": game_result.getCompletionPercentage(),
                "NumJumps": game_result.getNumJumps(),
                "KillsTotal": game_result.getKillsTotal(),
                "CurrentLives": game_result.getCurrentLives(),
                "NumCollectedTileCoins": game_result.getNumCollectedTileCoins(),
                "RemainingTime": game_result.getRemainingTime(),
            })

            # Technically this is a ratio since it is between 0 and 1.
            completionPercentage = float(
                str(game_result.getCompletionPercentage()))
            objs.append(completionPercentage)

            cur_measures = [mf(level, game_result) for mf in self.measure_funcs]
            measures.append(cur_measures)

        return MarioResult.from_raw(
            level.data,
            np.array(occupancy_grids).mean(axis=0),
            objs,
            measures,
            all_info,
            opts={
                "aggregation": self.config.aggregation_type,
            },
        )

    def actual_qd_score(self, objs: "array-like"):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= self.MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)


def module_demo(n_evals: int = 1, render: bool = False):
    """Demonstration of the MarioModule."""
    setup_logging(on_worker=False)

    module = MarioModule(
        MarioConfig(
            measure_names=[
                "calc_higher_level_non_empty_blocks",
                "calc_num_enemies",
                "calc_num_jumps",
            ],
            aggregation_type="mean",
        ))

    data = MarioLevel.str_to_number("""\
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
--------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX""".split("\n"))

    result = module.evaluate(
        level=data,
        n_evals=n_evals,
        render=render,
    )

    print(result)


if __name__ == "__main__":
    fire.Fire(module_demo)
