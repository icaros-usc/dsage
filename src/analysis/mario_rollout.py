"""Visualize Mario rollouts from an experiment.

This script should be run within a Singularity shell.

TODO: Video does not work (probably not easy to capture it since it is done in
Java).

By default, this script also generates videos of the rollout in the
videos/ directory of the logdir. Each video will be saved in a sub-directory
named MODE__idx_X_X_X../eval_X/, which contains all the frames as well as
video.mp4.  The video directory can be overridden with --video-output.

Usage:
    python -m src.analysis.mario_rollout LOGDIR MODE

Examples:
    # Run the best policy and record video in
    # my-logdir/videos/best__idx_X_X/eval_0/video.mp4
    python -m src.analysis.mario_rollout my-logdir/ best

    # Run the best policy and show the rendering but don't record video.
    python -m src.analysis.mario_rollout my-logdir/ best --novideo

    # Run 5 evals of the policy at index [10,10] without rendering anything.
    python -m src.analysis.mario_rollout my-logdir/ idx --query "[10,10]" \
        --norender --n-evals 5
"""
import os
import shutil
from pathlib import Path
from pprint import pprint

import fire
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from pyvirtualdisplay.smartdisplay import SmartDisplay

from src.analysis.utils import load_experiment, load_metrics
from src.mario.level import MarioLevel
from src.mario.module import MarioConfig, MarioModule
from src.mario.visualize import visualize_level
from src.utils.logging import setup_logging


def main(logdir: str,
         mode: str = "best",
         n_evals: int = 1,
         render: bool = True,
         render_delay: float = 0.01,
         video: bool = False,
         framerate: int = 60,
         delete_frames: bool = False,
         video_output: str = None,
         query: "array-like" = None,
         motion_blur: bool = False,
         still: bool = False,
         gen: int = None):
    """Reads results from logdir and rolls out policies from the archive.

    Args:
        logdir: Path to a logging directory output by an experiment.
        mode: "best", "random", "all", "idx"
        n_evals: Number of rollouts to perform.
        render: Whether to render the rollout.
        render_delay: Time to delay after each frame is rendered.
        video: Whether to record a video of the rollout(s) in the `videos/`
            directory of the logdir. We also record a small description of
            the video in `info.md` in the same directory as the video.
            render_delay is set to 0 when this is True.
        framerate: Frame rate (FPS) for video.
        delete_frames: Pass this to delete video frames after the video is
            created. This is useful because video frames may occupy lots of
            space.
        video_output: Pass this to override the output directory for videos.
            Videos will be saved in {video_output}/eval_X/video.mp4
        query: Index to query in the archive (mode must be "idx").
        motion_blur: Pass this flag to output the motion blur instead of
            outputting a video. Motion blurs are saved in the `motion_blur/`
            directory in the logdir.
        still: Pass this flag to output a still image of the environment instead
            of outputting a video. Still images are saved in the `still/`
            directory in the logdir.
        gen: Generation from which to load archive. By default, the final
            generation is loaded.
    """
    if video:
        raise RuntimeError("video is currently not implemented")

    if mode == "all" and video and video_output is not None:
        # Necessary because video_output only specifies one directory, and "all"
        # mode will have many outputs.
        raise ValueError("video_output must be None in 'all' mode")

    setup_logging(on_worker=False)
    logdir = load_experiment(logdir)

    if video:
        # Set up virtual display for capturing video.
        disp = SmartDisplay()  # Use default sizes.
        disp.start()

        # Avoid delays since rendering already takes so long.
        render_delay = 0.0

    mario_module = MarioModule(MarioConfig())

    gen = load_metrics(logdir).total_itrs if gen is None else gen
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

    def rollout(sol, obj, beh, idx, metadata):
        """Rolls out a single policy."""
        print("===== Solution Info =====\n"
              f"Objective Value: {obj}\n"
              f"Behavior Values: {beh}\n"
              f"Index: {idx}\n"
              f"Metadata:")
        pprint(metadata)
        print("==> Level <==")
        print(MarioLevel(metadata["level"]).to_str())

        idx_str = '_'.join(map(str, idx))
        if motion_blur:
            visualize_level(
                MarioLevel(metadata["level"]).to_str(),
                logdir.file(f"motion_blur/{mode}__idx_{idx_str}.png"),
            )
            return
        elif still:
            visualize_level(
                MarioLevel(metadata["level"]).to_str(),
                logdir.file(f"still/{mode}__idx_{idx_str}.png"),
                show_mario=False,
            )
            return

        print("Evaluation Results:")

        if video:
            video_dir = None
            frame_i = 0

            def render_callable():
                nonlocal frame_i, disp
                img = disp.waitgrab()
                img.save(video_dir / f"frame-{frame_i:05d}.png")
                frame_i += 1

        if video:
            main_dir = (logdir.pdir(f"videos/{mode}__idx_{idx_str}",
                                    touch=False)
                        if video_output is None else Path(video_output))
            if main_dir.exists():  # Clear existing video directory.
                shutil.rmtree(main_dir)
            main_dir.mkdir()

        with alive_bar(n_evals, "Evals") as progress:
            for i in range(n_evals):
                if video:
                    video_dir = main_dir / f"eval_{i}"
                    video_dir.mkdir()

                res = mario_module.evaluate(
                    level=metadata["level"],
                    n_evals=1,
                    render=render,
                )

                print("-------------")
                print(res)

                if video:
                    print("Assembling video with ffmpeg")
                    os.system(f"""\
ffmpeg -an -r {framerate} -i "{video_dir / 'frame-%*.png'}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    "{video_dir / 'video.mp4'}" \
    -y \
""")

                    if delete_frames:
                        print("Deleting video frames")
                        for file in video_dir.glob("frame-*.png"):
                            file.unlink()

                    with (video_info :=
                          video_dir / "info.md").open("w") as file:
                        file.write(str(res))
                    print(f"Saved info to {video_info}")

                    print(f"Saved video to {video_dir/'video.mp4'}")

                progress()  # pylint: disable = not-callable

    # Select policies to rollout.
    if mode == "all":
        for elite in df.iterelites():
            rollout(*elite)
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

        rollout(
            df.batch_solutions()[rollout_idx],
            df.batch_objectives()[rollout_idx],
            df.batch_behaviors()[rollout_idx],
            df.batch_indices()[rollout_idx],
            df.batch_metadata()[rollout_idx],
        )

    # Cleanup.
    if video:
        disp.stop()


if __name__ == "__main__":
    fire.Fire(main)
