"""Class representing the results of an evaluation."""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MarioResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).

    Note that the Mario AI Framework also has a MarioResult class which has no
    relation to this class:
    https://github.com/amidos2006/Mario-AI-Framework/blob/master/src/engine/core/MarioResult.java
    """

    ## Raw data ##

    level: np.ndarray = None  # (lvl_height, lvl_width) integer array.
    occupancy_grid: np.ndarray = None  # (lvl_height, lvl_width) integer array.
    objs: np.ndarray = None
    measures: np.ndarray = None
    all_info: List[dict] = None  # Additional info/metadata regarding each eval.

    ## Aggregate data ##

    agg_obj: float = None
    agg_measures: np.ndarray = None  # (behavior_dim,) array

    ## Measures of spread ##

    std_obj: float = None
    std_measure: np.ndarray = None  # (behavior_dim,) array

    ## Other data ##

    failed: bool = False
    log_message: str = None

    @staticmethod
    def from_raw(
        level: np.ndarray,
        occupancy_grid: np.ndarray,
        objs: "array-like",
        measures: "array-like",
        all_info: list,
        opts: dict = None,
    ):
        """Constructs a MarioResult from raw data.

        `opts` is a dict with several configuration options. It may be better as
        a gin parameter, but since MarioResult is created on workers, gin
        parameters are unavailable (unless we start loading gin on workers too).
        Options in `opts` are:

            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean measure
                - "median": Take the median, e.g. median measure (element-wise)
        """
        # Handle config options.
        opts = opts or {}
        opts.setdefault("aggregation", "mean")

        assert opts["aggregation"] == "mean", \
            "Only mean aggregation is currently supported."

        objs = np.array(objs)
        measures = np.array(measures)

        return MarioResult(
            level=level,
            occupancy_grid=occupancy_grid,
            objs=objs,
            measures=measures,
            all_info=all_info,
            agg_obj=np.mean(objs),
            agg_measures=np.mean(measures, axis=0),
            std_obj=np.std(objs),
            std_measure=np.std(measures, axis=0),
        )
