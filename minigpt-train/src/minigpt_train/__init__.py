"""MiniGPT's reproducible training and operations package."""

from .config import ResolvedConfig, load_config
from .data import ShardSplit, load_shards_manifest

__all__ = ["ResolvedConfig", "ShardSplit", "load_config", "load_shards_manifest"]
__version__ = "0.1.0"
