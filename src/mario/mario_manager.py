"""Provides MarioManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import gin
import numpy as np
from dask.distributed import Client

from src.device import DEVICE
from src.maze.emulation_model.buffer import Experience
from src.maze.emulation_model.aug_buffer import AugExperience
from src.utils.worker_state import init_mario_module

from .emulation_model.emulation_model import MarioEmulationModel
from .gan.dcgan import MarioGenerator
from .mario_result import MarioResult
from .module import MarioConfig, MarioModule
from .run import run_mario

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng"])
class MarioManager:
    """Manager for the Mario environment.

    Args:
        client: Dask client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        initial_sol_sigma: Initial solutions are selected from a distribution
            with sigma set to this parameter.
    """

    def __init__(
        self,
        client: Client,
        rng: np.random.Generator = None,
        n_evals: int = gin.REQUIRED,
        initial_sol_sigma: float = gin.REQUIRED,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.initial_sol_sigma = initial_sol_sigma

        self.emulation_model = None

        # Set up GAN.
        self.generator = MarioGenerator().load_from_saved_weights().to(
            DEVICE).eval()

        # Set up a module locally and on workers. During evaluations,
        # run_mario retrieves this module and uses it to evaluate the
        # function. Configuration is done with gin (i.e. the params are in the
        # config file).
        self.module = MarioModule(config := MarioConfig())
        client.register_worker_callbacks(lambda: init_mario_module(config))

    def em_init(self,
                seed: int,
                pickle_path: Path = None,
                pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = MarioEmulationModel(seed=seed + 420)
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        return (self.initial_sol_sigma *
                self.rng.standard_normal(size=size, dtype=np.float32))

    def em_train(self):
        self.emulation_model.train()

    def emulation_pipeline(self, sols):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures.

        Args:
            sols: Emitted solutions.

        Returns:
            lvls: Generated levels.
            objs: Predicted objective values.
            measures: Predicted measure values.
            success_mask: Array of size `len(lvls)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        lvls = self.generator.levels_from_latent(sols)
        success_mask = np.ones(len(lvls), dtype=bool)
        objs, measures = self.emulation_model.predict(lvls)
        return lvls, objs, measures, success_mask

    def eval_pipeline(self, sols):
        """Pipeline that takes a list of solutions and evaluates it.

        Args:
            sols: Emitted solutions.

        Returns:
            Results of the evaluation.
        """
        lvls = self.generator.levels_from_latent(sols)

        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=len(sols),
                                             endpoint=True)
        futures = [
            self.client.submit(
                run_mario,
                lvl,
                self.n_evals,
                seed,
                pure=False,
            ) for lvl, seed in zip(lvls, evaluation_seeds)
        ]
        logger.info("Collecting evaluations")
        results: List[MarioResult] = self.client.gather(futures)

        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution.
            result: Evaluation result.
        """
        obj = result.agg_obj
        meas = result.agg_measures
        if self.emulation_model.pre_network is not None:
            self.emulation_model.add(
                AugExperience(sol, result.level, obj, meas,
                              result.occupancy_grid))
        else:
            self.emulation_model.add(Experience(sol, result.level, obj, meas))

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "level": result.level,
            "log_message": result.log_message,
        }
        return failed_level_info
