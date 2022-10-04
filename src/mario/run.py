"""Provides a generic function for executing Mario envs."""
import logging
import random
import time
import traceback

import numpy as np

from src.mario.mario_result import MarioResult
from src.utils.worker_state import get_mario_module

logger = logging.getLogger(__name__)


def run_mario(level: np.ndarray,
              n_evals: int,
              seed: int,
              eval_kwargs=None) -> MarioResult:
    """Grabs the Mario module and evaluates level n_evals times."""
    start = time.time()
    eval_kwargs = {} if eval_kwargs is None else eval_kwargs

    logger.info("seeding global randomness")
    np.random.seed(seed // np.int32(4))
    random.seed(seed // np.int32(2))

    logger.info("run_mario with %d n_evals and seed %d", n_evals, seed)
    mario_module = get_mario_module()

    try:
        result = mario_module.evaluate(level=level, n_evals=n_evals, seed=seed)
    except TimeoutError as e:
        logger.warning("Evaluate failed")
        logger.info("The level was %s", level)
        result = MarioResult(
            failed=True,
            log_message="Evaluate failed with following error\n"
            f"{''.join(traceback.TracebackException.from_exception(e).format())}\n"
            f"Level was {level}")

    logger.info("run_mario done after %f sec", time.time() - start)

    return result
