"""Visualize mazes from an experiment.

This script should be run within a Singularity shell.

Usage:
    python -m src.analysis.maze_viz LOGDIR MODE

Examples:
    # Visualize the best policy.
    python -m src.analysis.maze_viz my-logdir/ best

    # Visualize the policy at index [10,10].
    python -m src.analysis.maze_viz my-logdir/ idx --query "[10,10]"

    # Visualize the policy at index [10,10] as a video.
    python -m src.analysis.maze_viz my-logdir/ idx --query "[10,10]" --video
"""
import os
from functools import partial
from pprint import pprint

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from src.analysis.utils import load_experiment, load_metrics
from src.maze.agents.rl_agent import RLAgent, RLAgentConfig
from src.maze.envs.maze import MazeEnv
from src.maze.level import OBJ_TYPES_TO_INT, MazeLevel
from src.maze.module import MazeModule
from src.utils.logging import setup_logging


def main(logdir: str,
         mode: str = "best",
         query: "array-like" = None,
         gen: int = None,
         video: bool = False,
         framerate: int = 6,
         occupancy: bool = False):
    """Reads results from logdir and outputs maze images.

    Images are output in the `maze_viz/` directory of the logging directory.

    Args:
        logdir: Path to a logging directory output by an experiment.
        mode: "best", "random", "all", "idx"
        query: Index to query in the archive (mode must be "idx").
        gen: Generation from which to load archive. By default, the final
            generation is loaded.
        video: Pass this to save a video instead of just a still image.
        framerate: Frame rate of the video.
        occupancy: Pass this to save the occupancy grid heatmap instead of
            visualizing the level.
    """
    setup_logging(on_worker=False)
    logdir = load_experiment(logdir)

    gen = load_metrics(logdir).total_itrs if gen is None else gen
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

    # Helpful for finding good indices.
    #  for idx in df.batch_indices():
    #      if idx[0] == 100:
    #          print(idx)

    def visualize(sol, obj, beh, idx, metadata):
        """Visualizes a single maze."""
        print("===== Solution Info =====\n"
              f"Objective Value: {obj}\n"
              f"Behavior Values: {beh}\n"
              f"Index: {idx}\n"
              f"Metadata:")
        pprint(metadata)

        idx_str = '_'.join(map(str, idx))
        level = sol.reshape((16, 16)).astype(int)
        print(level)

        print("==> Level <==")
        print(MazeLevel(level).to_str())
        print("Bit map:")
        print(level)
        adj = MazeModule._get_adj(level)

        if occupancy:
            occ_grid = metadata["maze_metadata"]["aug_level"]
            # Use this to tweak the grid so that the trajectory cells show up
            # more brightly.
            #  plt.imshow((occ_grid + 1)**0.001)
            plt.imshow(occ_grid)
            # Hide axes and borders so only the grid is shown.
            plt.axis("off")
            plt.savefig(
                logdir.file(f"maze_viz/{mode}__idx_{idx_str}_occupancy.png"))
            return

        # Find the best distances
        dist, predecessors = csgraph.floyd_warshall(adj,
                                                    return_predecessors=True)
        dist[dist == np.inf] = -np.inf  # For easier argmax to find the diameter

        print(f"Optimal path length: {dist.max()}")
        # Label the start and the end point
        endpoints = np.unravel_index(dist.argmax(), dist.shape)
        start_cell, end_cell = zip(*np.unravel_index(endpoints, level.shape))

        endpoint_level = level.copy()
        endpoint_level[start_cell] = OBJ_TYPES_TO_INT["S"]
        endpoint_level[end_cell] = OBJ_TYPES_TO_INT["G"]

        # Offset start, goal to account for the added outer walls
        start_pos = (start_cell[1] + 1, start_cell[0] + 1)
        goal_pos = (end_cell[1] + 1, end_cell[0] + 1)
        print(f"Start: {start_pos}; End: {goal_pos}")
        env_func = partial(MazeEnv,
                           size=level.shape[0] + 2,
                           bit_map=level,
                           start_pos=start_pos,
                           goal_pos=goal_pos)
        rl_agent_conf = RLAgentConfig(recurrent_hidden_size=256,
                                      model_path="accel_seed_1/model_20000.tar",
                                      n_envs=1)
        rl_agent = RLAgent(env_func, n_evals=1, config=rl_agent_conf)

        if video:
            frame = 0
            video_dir = logdir.pdir(f"maze_viz/{mode}__idx_{idx_str}/",
                                    touch=True)

            def render_func(vec_env):
                nonlocal frame
                img = vec_env.render(mode="rgb_array")
                name = video_dir / f"frame-{frame:08d}.png"
                cv2.imwrite(str(name), img)
                print(f"Saved frame {frame}")
                frame += 1

            rl_agent.eval_and_track(level_shape=level.shape,
                                    render_func=render_func)

            print("Assembling video with ffmpeg")
            os.system(f"""\
ffmpeg -an -r {framerate} -i "{video_dir / 'frame-%*.png'}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    {logdir.file(f'maze_viz/{mode}__idx_{idx_str}.mp4')} \
    -y \
""")
        else:
            rl_agent.vec_env.reset()
            img = rl_agent.vec_env.render(mode="rgb_array")
            name = logdir.file(f"maze_viz/{mode}__idx_{idx_str}.png")
            cv2.imwrite(name, img)
            print(f"Saved to {name}")

    # Select mazes to visualize.
    if mode == "all":
        for elite in df.iterelites():
            visualize(*elite)
    else:
        if mode == "best":
            rollout_idx = np.argmax(df["objective"])
        elif mode == "random":
            rollout_idx = np.random.randint(len(df))
        elif mode == "idx":
            query = tuple(query)
            try:
                rollout_idx = df.batch_indices().index(query)
            except ValueError:
                print(f"Index {query} not available")
                return
        else:
            raise ValueError(f"Unknown mode {mode}")

        visualize(
            df.batch_solutions()[rollout_idx],
            df.batch_objectives()[rollout_idx],
            df.batch_behaviors()[rollout_idx],
            df.batch_indices()[rollout_idx],
            df.batch_metadata()[rollout_idx],
        )


if __name__ == "__main__":
    fire.Fire(main)
