"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from functools import partial

from dask.distributed import get_worker

from src.mario.module import MarioConfig, MarioModule
from src.maze.agents.rl_agent import RLAgentConfig, RLAgent
from src.maze.module import MazeConfig, MazeModule


#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)


#
# Maze module
#

MAZE_MOD_ATTR = "maze_module"


def init_maze_module(config: MazeConfig):
    """Initializes this worker's maze module."""
    set_worker_state(MAZE_MOD_ATTR, MazeModule(config))


def get_maze_module() -> MazeModule:
    """Retrieves this worker's maze module."""
    return get_worker_state(MAZE_MOD_ATTR)


#
# Maze RL agent
#

MAZE_RL_AGENT_MOD_ATTR = "maze_rl_agent"


def init_maze_rl_agent_func(config: RLAgentConfig):
    """Initializes this worker's maze module."""
    set_worker_state(MAZE_RL_AGENT_MOD_ATTR, partial(RLAgent, config=config))


def get_maze_rl_agent_func() -> callable:
    """Retrieves this worker's maze rl agent."""
    return get_worker_state(MAZE_RL_AGENT_MOD_ATTR)


#
# Mario module
#

MARIO_MOD_ATTR = "mario_module"


def init_mario_module(config: MarioConfig):
    """Initializes this worker's Mario module."""
    set_worker_state(MARIO_MOD_ATTR, MarioModule(config))


def get_mario_module() -> MarioModule:
    """Retrieves this worker's Mario module."""
    return get_worker_state(MARIO_MOD_ATTR)
