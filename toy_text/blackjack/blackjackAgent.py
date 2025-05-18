from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    """
    @class BlackjackAgent
    @brief Implements a tabular Q-learning agent for the Blackjack environment.

    This agent uses a dictionary to store Q-values for each discrete state-action pair
    and applies the epsilon-greedy strategy for action selection.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        existing_q: dict = None
    ):
        """
        @brief Constructor for the BlackjackAgent.

        @param env The Blackjack environment from OpenAI Gym.
        @param learning_rate The learning rate (alpha) for updating Q-values.
        @param initial_epsilon Initial exploration rate for the epsilon-greedy policy.
        @param epsilon_decay Amount by which epsilon decays after each episode.
        @param final_epsilon Minimum value that epsilon can decay to.
        @param discount_factor Discount factor (gamma) for future rewards.
        @param existing_q Optional pre-trained Q-table to resume training from.
        """
        self.env = env
        if existing_q:
            self.q_values = existing_q
        else:
            self._initialize_q_values()
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        @brief Selects an action using an epsilon-greedy policy.

        @param obs The current observation (state) represented as a tuple of (player sum, dealer card, usable ace).
        @return The selected action (integer).
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        @brief Updates the Q-value of a state-action pair using the Bellman equation.

        @param obs The current state.
        @param action The action taken in the current state.
        @param reward The reward received after the action.
        @param terminated Whether the episode has terminated.
        @param next_obs The next state reached after the action.
        """

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        @brief Decays the epsilon value by a fixed amount until the final_epsilon is reached.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def _initialize_q_values(self):
        """
        @brief Initializes the Q-value table with all possible state-action pairs for Blackjack.

        The state is represented as a tuple of:
        - player’s sum (0 to 31)
        - dealer’s visible card (0 to 10)
        - whether the player has a usable ace (True/False)

        Each state maps to an array of action values initialized to zero.
        """
        self.q_values = {
            (d1, d2, d3): np.zeros(self.env.action_space.n)
            for d1 in range(32)
            for d2 in range(11)
            for d3 in range(2)
        }
