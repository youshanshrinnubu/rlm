"""Concrete agent implementations for reinforcement learning with language models."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from rlm.core import BaseAgent, RLMConfig, Transition


class RandomAgent(BaseAgent):
    """An agent that selects actions uniformly at random.

    Useful as a baseline or for environment sanity checks.
    """

    def __init__(self, config: RLMConfig, action_space: list[Any]) -> None:
        """Initialize the random agent.

        Args:
            config: Configuration object for the agent.
            action_space: List of possible actions the agent can take.
        """
        super().__init__(config)
        self.action_space = action_space

    def select_action(self, state: Any) -> Any:
        """Select a random action from the action space.

        Args:
            state: The current environment state (ignored).

        Returns:
            A randomly selected action.
        """
        return random.choice(self.action_space)

    def update(self, transition: Transition) -> dict[str, float]:
        """No-op update for the random agent.

        Args:
            transition: The observed transition (ignored).

        Returns:
            Empty metrics dictionary.
        """
        return {}


class EpsilonGreedyAgent(BaseAgent):
    """An epsilon-greedy agent that balances exploration and exploitation.

    Maintains a Q-table for discrete state-action pairs and updates
    using the standard Q-learning update rule.
    """

    def __init__(
        self,
        config: RLMConfig,
        action_space: list[Any],
        epsilon: float = 0.1,
        alpha: float = 0.01,
    ) -> None:
        """Initialize the epsilon-greedy agent.

        Args:
            config: Configuration object for the agent.
            action_space: List of possible actions the agent can take.
            epsilon: Probability of selecting a random action (exploration rate).
            alpha: Learning rate for Q-value updates.
        """
        super().__init__(config)
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table: dict[Any, dict[Any, float]] = {}

    def _get_q_value(self, state: Any, action: Any) -> float:
        """Retrieve Q-value for a state-action pair, defaulting to 0.0."""
        state_key = str(state)
        action_key = str(action)
        return self.q_table.get(state_key, {}).get(action_key, 0.0)

    def _set_q_value(self, state: Any, action: Any, value: float) -> None:
        """Set the Q-value for a state-action pair."""
        state_key = str(state)
        action_key = str(action)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = value

    def select_action(self, state: Any) -> Any:
        """Select an action using the epsilon-greedy policy.

        Args:
            state: The current environment state.

        Returns:
            The selected action.
        """
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        q_values = [self._get_q_value(state, a) for a in self.action_space]
        best_idx = int(np.argmax(q_values))
        return self.action_space[best_idx]

    def update(self, transition: Transition) -> dict[str, float]:
        """Update Q-values using the Q-learning update rule.

        Args:
            transition: The observed (state, action, reward, next_state, done) transition.

        Returns:
            Dictionary containing the TD error for logging.
        """
        state, action, reward, next_state, done = (
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        )

        current_q = self._get_q_value(state, action)

        if done:
            target_q = reward
        else:
            next_q_values = [self._get_q_value(next_state, a) for a in self.action_space]
            target_q = reward + self.config.gamma * max(next_q_values)

        td_error = target_q - current_q
        new_q = current_q + self.alpha * td_error
        self._set_q_value(state, action, new_q)

        return {"td_error": float(td_error)}
