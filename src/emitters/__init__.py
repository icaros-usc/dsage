"""pyribs-compliant emitters."""
import gin
import ribs

from src.emitters.map_elites_baseline_emitter import MapElitesBaselineEmitter

__all__ = [
    "GaussianEmitter",
    "ImprovementEmitter",
    "MapElitesBaselineEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""


@gin.configurable
class ImprovementEmitter(ribs.emitters.ImprovementEmitter):
    """gin-configurable version of pyribs ImprovementEmitter."""
