"""Core module for rlm — Reinforcement Learning with Language Models.

This module provides the foundational classes and utilities for integrating
reinforcement learning algorithms with language model backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for an RLM agent.

    Attributes:
        model_name: Name or path of the language model to use.
        max_tokens: Maximum number of tokens to generate per step.
        temperature: Sampling temperature for the language model.
        reward_scale: Scaling factor applied to raw rewards.
        discount_factor: Discount factor (gamma) for future rewards.
        max_steps: Maximum number of steps per episode.
        device: Device to run the model on ('cpu', 'cuda', etc.).
    """

    model_name: str = "gpt2"
    max_tokens: int = 256
    temperature: float = 1.0
    reward_scale: float = 1.0
    discount_factor: float = 0.99
    max_steps: int = 100
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.temperature:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                f"discount_factor must be in [0, 1], got {self.discount_factor}"
            )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")


@dataclass
class Transition:
    """A single environment transition.

    Attributes:
        observation: The observation (prompt/context) seen by the agent.
        action: The action (generated text) taken by the agent.
        reward: The scalar reward received after the action.
        next_observation: The observation following the action.
        done: Whether the episode ended after this transition.
        info: Optional auxiliary information from the environment.
    """

    observation: str
    action: str
    reward: float
    next_observation: str
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all RLM agents.

    Subclasses must implement :meth:`act` and :meth:`update`.
    """

    def __init__(self, config: RLMConfig) -> None:
        self.config = config
        self._step: int = 0
        logger.info("Initialised %s with config: %s", self.__class__.__name__, config)

    @abstractmethod
    def act(self, observation: str) -> str:
        """Select an action given the current observation.

        Args:
            observation: The current textual observation / prompt.

        Returns:
            The generated text action.
        """

    @abstractmethod
    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        """Update the agent's parameters from a batch of transitions.

        Args:
            transitions: A list of :class:`Transition` objects.

        Returns:
            A dictionary of scalar training metrics (e.g. loss values).
        """

    def reset(self) -> None:
        """Reset any per-episode state."""
        self._step = 0

    def compute_returns(
        self, rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Compute discounted returns from a sequence of rewards.

        Args:
            rewards: List of scalar rewards.
            dones: List of episode-termination flags.

        Returns:
            List of discounted cumulative returns.
        """
        gamma = self.config.discount_factor
        returns: List[float] = []
        running_return = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            running_return = reward + gamma * running_return * (1.0 - float(done))
            returns.insert(0, running_return)
        return returns

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(model={self.config.model_name!r})"
