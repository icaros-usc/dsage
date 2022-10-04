"""Utilities for other postprocessing scripts.

Note that most of these functions require that you first call `load_experiment`
so that gin configurations are loaded properly.
"""
import pickle as pkl
from pathlib import Path

import gin
import numpy as np
from logdir import LogDir

# Including this makes gin config work because main imports (pretty much)
# everything.
import src.main  # pylint: disable = unused-import
from src.archives import GridArchive
from src.utils.deprecation import DEPRECATED_OBJECTS
from src.utils.metric_logger import MetricLogger


def load_experiment(logdir: str) -> LogDir:
    """Loads gin configuration and logdir for an experiment.

    Intended to be called at the beginning of an analysis script.

    Args:
        logdir: Path to the experiment's logging directory.
    Returns:
        LogDir object for the directory.
    """
    gin.clear_config()  # Erase all previous param settings.
    gin.parse_config_file(Path(logdir) / "config.gin",
                          skip_unknown=DEPRECATED_OBJECTS)
    logdir = LogDir(gin.query_parameter("experiment.name"), custom_dir=logdir)
    return logdir


def load_metrics(logdir) -> MetricLogger:
    return MetricLogger.from_json(logdir.file("metrics.json"))


def load_archive_from_history(logdir: LogDir, individual=False) -> GridArchive:
    """Generator that produces archives loaded from archive_history.pkl.

    Note that these archives will only contain objectives and BCs.

    Pass `individual` to indicate that the archive should be yielded after each
    solution is inserted into the archive, rather than only at the end of each
    iteration / generation.

    WARNING: Be careful that the history only recorded solutions that were
    inserted into the archive successfully, so many solutions are excluded.
    """
    archive_type = str(gin.query_parameter("Manager.archive_type"))
    if archive_type == "@GridArchive":
        # Same construction as in Manager.
        # pylint: disable = no-value-for-parameter
        archive = GridArchive(seed=42, dtype=np.float32)
    else:
        raise TypeError(f"Cannot handle archive type {archive_type}")
    archive.initialize(0)  # No solutions.

    with logdir.pfile("archive_history.pkl").open("rb") as file:
        archive_history = pkl.load(file)

    yield archive  # Start with empty archive.
    for gen_history in archive_history:
        archive.new_history_gen()
        for obj, bcs in gen_history:
            archive.add([], obj, bcs, None)  # No solutions, no metadata.
            if individual:
                yield archive
        if not individual:
            yield archive


def load_archive_gen(logdir: LogDir, gen: int) -> GridArchive:
    """Loads the archive at a given generation; works for ME-ES too."""
    itr = iter(load_archive_from_history(logdir))
    for _ in range(gen + 1):
        archive = next(itr)
    return archive
