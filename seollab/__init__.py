"""SeolLab DreamerV2/V3 evaluation helpers for notebook import."""

__version__ = '0.1.0'

from .bootstrap import ensure_package
from .paths import PathConfig, default_paths
from .env_setup import setup_jax, clone_dreamerv3, ensure_xvfb
from . import inference_demo
