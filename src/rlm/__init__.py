"""rlm - Reinforcement Learning with Language Models.

A library for training and evaluating language models using
reinforcement learning techniques.

Personal fork: using this to experiment with RLHF on smaller models.
See my notes in /docs/experiments for details.

Note: Added RLMEvaluator to the public API since I use it frequently
in my experiment scripts and was tired of importing it directly.
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
from rlm.evaluator import RLMEvaluator
from rlm.trainer import RLMTrainer

__all__ = [
    "RLMAgent",
    "RLMConfig",
    "RLMEvaluator",
    "RLMTrainer",
    "__version__",
]
