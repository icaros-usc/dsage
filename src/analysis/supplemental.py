"""Generates additional figures and videos.

Usage:
    python -m src.analysis.supplemental COMMAND
"""
import functools
import os
import shutil
from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import slugify
from loguru import logger

from src.analysis.figures import load_manifest
from src.analysis.heatmap import plot_generation_on_axis
from src.analysis.utils import load_archive_gen, load_experiment, load_metrics
from src.mpl_styles.utils import mpl_style_file


def visualize_env_archives(mode,
                           root_dir,
                           viz_data,
                           env,
                           cur_subfig,
                           custom_gen=None):
    """Plots visualizations for one environment in viz_data.

    cur_subfig may be either a subfigure or a regular figure.

    Pass custom_gen to plot generations that are not the last generation. In
    this case, custom_gen must be a dict mapping from algorithm name to the
    generation for that algorithm.
    """
    ncols = len(viz_data[env]["Algorithms"])
    ax = cur_subfig.subplots(
        1,
        ncols,
        gridspec_kw={"wspace": {
            "heatmap": 0.03,
            "histogram": 0.3,
        }[mode]},
    )

    for col, (cur_ax, algo) in enumerate(zip(ax, viz_data[env]["Algorithms"])):
        logdir = load_experiment(root_dir /
                                 viz_data[env]["Algorithms"][algo]["logdir"])
        gen = (load_metrics(logdir).total_itrs
               if custom_gen is None else custom_gen[algo])
        # Allow for setting custom names.
        algo = {
            "DSAGE-Only Anc": "DSAGE-Only\nAnc",
            "DSAGE-Only Down": "DSAGE-Only\nDown",
        }.get(algo, algo)

        cur_ax.set_title(algo)

        if mode == "heatmap":
            # See heatmap.py for these settings.
            plot_generation_on_axis(
                ax=cur_ax,
                mode="single",
                logdir=logdir,
                gen=gen,
                plot_kwargs={
                    "square": True,
                    "cmap": "viridis",
                    "pcm_kwargs": {
                        "rasterized": True,
                    },
                    "vmin": 0,
                    "vmax": 1,
                },
            )

            if env == "Mario":
                cur_ax.set_xticks([0, 150])

            if col == 0:
                # Set y label on left plot.
                cur_ax.set_ylabel(viz_data[env]["ylabel"])
            else:
                # Clear y ticks on non-left plots.
                cur_ax.yaxis.set_ticklabels([])

            if col == ncols // 2:
                # Set x label on middle plot.
                cur_ax.set_xlabel(viz_data[env]["xlabel"])
            if col == ncols - 1:
                # Add colorbar when on last plot in column.

                # Remove all current colorbars.
                for a in cur_subfig.axes:
                    if a.get_label() == "<colorbar>":
                        a.remove()

                # Retrieve the heatmap mesh.
                artist = None
                for child in cur_ax.get_children():
                    if isinstance(child, mpl.collections.QuadMesh):
                        artist = child

                # Add axes for the colorbar. Solutions for this are complicated:
                # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                ratio = cur_ax.get_position().width / 0.129638671875
                # The axis width is expressed as a fraction of the figure width.
                # We first took the width when there were 5 figures, which was
                # 0.129638671875, and now we express width as a ratio of axis
                # width to that width.
                cax = cur_subfig.add_axes([
                    cur_ax.get_position().x1 + 0.02 * ratio,
                    cur_ax.get_position().y0,
                    0.02 * ratio,
                    cur_ax.get_position().height,
                ])

                # Create colorbar.
                cur_subfig.colorbar(artist, cax=cax)
        elif mode == "histogram":
            archive = load_archive_gen(logdir, gen)
            min_score, max_score = 0, 1

            # We cut off the histogram at the min score because the min score is
            # known, but we increase the max score a bit to show solutions which
            # exceed the reward threshold.
            bin_counts, bins, patches = cur_ax.hist(  # pylint: disable = unused-variable
                archive.as_pandas().batch_objectives(),
                range=(min_score, max_score + 400),
                bins=100,
            )

            # Rough estimate of max items in a bin.
            cur_ax.set_ylim(top=150)

            # Force ticks, as there are no ticks when the plot is empty.
            cur_ax.set_yticks([0, 50, 100, 150])

            # Alternative - logarithmic scale (harder to understand).
            #  cur_ax.set_yscale("log")
            #  cur_ax.set_ylim(top=1000)

            # Set axis grid to show up behind the histogram.
            # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
            cur_ax.set_axisbelow(True)

            # Set up grid lines on y-axis. Style copied from simple.mplstyle.
            cur_ax.grid(color="0.9", linestyle="-", linewidth=0.3, axis="y")

            # Color the histogram with viridis.
            # https://stackoverflow.com/questions/51347451/how-to-fill-histogram-with-color-gradient-where-a-fixed-point-represents-the-mid
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            maxi = np.abs(bin_centers).max()
            norm = plt.Normalize(-maxi, maxi)
            cm = plt.cm.get_cmap("viridis")
            for c, p in zip(bin_centers, patches):
                # Also rasterize so we do not see "white lines" between bars.
                p.set(facecolor=cm(norm(c)), rasterized=True)

            # Hide spines.
            for pos in ['right', 'top', 'left']:
                cur_ax.spines[pos].set_visible(False)

            if col == ncols // 2:
                # Set x label on middle plot.
                cur_ax.set_xlabel("Objective")

        # Adjust position of plots -- somehow, we need to adjust it on every
        # iteration rather than just adjusting it once at the end.
        left = 0.1
        right = 0.945
        if mode == "heatmap":
            cur_subfig.subplots_adjust(left=left, right=right)
        elif mode == "histogram":
            cur_subfig.subplots_adjust(left=left,
                                       bottom=0.25,
                                       right=right,
                                       top=0.7)

    # Add suptitle at center of plots.
    #  center_x = (ax[0].get_position().x0 + ax[-1].get_position().x1) / 2
    #  cur_subfig.suptitle(env, x=center_x, y=0.95)


def visualize_archives(manifest: str,
                       output: str = None,
                       custom_gen=None,
                       mode: str = "heatmap",
                       video: bool = False):
    """Generates archive visualizations for appendix.

    Requires a manifest which looks like this:

        Archive Visualization:
          Environment 1:
            heatmap: True/False  # Whether to plot heatmap in this environment
                                 # (not all environments support heatmaps).
            xlabel: "XXX"  # Label for x-axis.
            ylabel: "XXX"  # Label for y-axis.
            # Algorithms are assumed to be the same across all environments.
            Algorithms:
              Name 1:
                logdir: XXXX
              Name 2:
                logdir: XXXX
              ...
          Environment 2:
            heatmap: ...
            xlabel: ...
            ylabel: ...
            Algorithms:
              ...
          ...

    Note this manifest can be included in the same document as the one for
    figures.py and for `agent_videos`.

    Args:
        manifest: Path to manifest file.
        output: Output file for figure. Defaults to `{mode}_figure.pdf` or
            `{mode}_figure_sans.pdf` (depending on whether sans=True).
        custom_gen: See `visualize_env_archives`.
        mode: Either "heatmap" or "histogram".
        video: Whether this function is being called as part of video
            generation. This induces a few special settings.
    """
    assert mode in ["heatmap", "histogram"], \
        f"Invalid mode {mode}"

    output = Path(f"{mode}_figure/" if output is None else output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    for sans in [False, True]:

        logger.info("Loading manifest")
        viz_data, root_dir = load_manifest(manifest, "Archive Visualization")
        if mode == "heatmap":
            # Only keep environments that support heatmap.
            viz_data = {
                key: val
                for key, val in viz_data.items()
                if val.get("heatmap", False)
            }

        logger.info("Plotting visualizations")
        with mpl_style_file("archive_visualization_sans.mplstyle"
                            if sans else "archive_visualization.mplstyle") as f:
            with plt.style.context(f):
                if video:
                    logger.info("Using video mode")

                for env in viz_data:
                    fig = plt.figure(figsize=(
                        {
                            # Width.
                            "Maze": 5.0,
                            "Mario": 5.5,
                        }[env],
                        {
                            # Height.
                            "Maze": 2.5,
                            "Mario": 1.4,
                        }[env]))
                    visualize_env_archives(
                        mode,
                        root_dir,
                        viz_data,
                        env,
                        fig,
                        custom_gen,
                    )

                    # The colorbar is giving trouble, so tight_layout does not work,
                    # and bbox_inches="tight" does not work either as it seems to cut
                    # things off despite this:
                    # https://stackoverflow.com/questions/35393863/matplotlib-error-figure-includes-axes-that-are-not-compatible-with-tight-layou
                    # Hence, we manually rearrange everything e.g. with subplots_adjust.
                    name = f"{slugify.slugify(env)}" + ("-sans" if sans else "")
                    for ext in ["pdf", "svg", "png"]:
                        fig.savefig(output / f"{name}.{ext}",
                                    dpi=600 if video else 300)
                    plt.close(fig)

    logger.info("Done")


heatmap_figure = functools.partial(visualize_archives,
                                   mode="heatmap",
                                   video=False)
histogram_figure = functools.partial(visualize_archives,
                                     mode="histogram",
                                     video=False)


def visualize_archives_video(manifest: str,
                             output: str = None,
                             frames: int = 250,
                             framerate: int = 20,
                             delete_frames: bool = False,
                             skip_frames: bool = False,
                             mode: str = "heatmap"):
    """Generates videos of archive visualizations.

    Uses same manifest as `visualize_archives`.

    Note: This method is inefficient because it calls `visualize_archives` for
    each frame, so it repeatedly reloads the manifest, logging directories, and
    archive histories.

    Args:
        manifest: Path to manifest file.
        output: Output name. The video is saved to "{output}.mp4", and frames
            are saved to "{output}/". Defaults to "{mode}_video".
        frames: Number of frames to include in video.
        framerate: FPS for video.
        delete_frames: Whether to delete frames once video is assembled.
        skip_plot: Skip plotting the frames and just make the video.
    """
    output = f"{mode}_video" if output is None else output
    frame_dir = Path(f"{output}/")

    if not skip_frames:
        logger.info("Removing existing frames")
        shutil.rmtree(output, ignore_errors=True)
        frame_dir.mkdir()

        logger.info("Determining frequency for each algorithm")
        freq = {}
        viz_data, root_dir = load_manifest(manifest, "Archive Visualization")
        for env in viz_data:
            algo_data = viz_data[env]["Algorithms"]
            for algo in algo_data:
                logdir = load_experiment(root_dir / algo_data[algo]["logdir"])
                total_itrs = load_metrics(logdir).total_itrs
                if total_itrs % frames != 0:
                    raise RuntimeError(
                        f"Number of generations ({total_itrs}) "
                        f"for {algo} "
                        f"must be divisible by `frames` ({frames})")
                freq[algo] = total_itrs // frames

        logger.info("Plotting frames")
        for i in range(frames + 1):
            logger.info("Frame {}", i)
            visualize_archives(
                manifest,
                frame_dir / f"{i:08d}.png",
                custom_gen={algo: i * f for algo, f in freq.items()},
                mode=mode,
                video=True,
            )

    logger.info("Assembling video with ffmpeg")
    video_output = Path(f"{output}.mp4")
    os.system(f"""\
ffmpeg -an -r {framerate} -i "{frame_dir / '%*.png'}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    "{video_output}" \
    -y \
""")

    if delete_frames:
        logger.info("Deleting frames")
        shutil.rmtree(frame_dir)

    logger.info("Done")


heatmap_video = functools.partial(visualize_archives_video, mode="heatmap")
histogram_video = functools.partial(visualize_archives_video, mode="histogram")

if __name__ == "__main__":
    fire.Fire({
        "heatmap_figure": heatmap_figure,
        "heatmap_video": heatmap_video,
        "histogram_figure": histogram_figure,
        "histogram_video": histogram_video,
    })
