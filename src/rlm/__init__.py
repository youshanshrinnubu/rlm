"""rlm - Reinforcement Learning with Language Models.

A library for training and evaluating language models using
reinforcement learning techniques.

Personal fork: using this to experiment with RLHF on smaller models.
See my notes in /docs/experiments for details.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlm")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "alexzhang13"
__license__ = "MIT"

from rlm.agent import RLMAgent
from rlm.config import RLMConfig
from rlm.trainer import RLMTrainer

__all__ = [
    "RLMAgent",
    "RLMConfig",
    "RLMTrainer",
    "__version__",
]
