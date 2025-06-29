import gymnasium as gym
import numpy as np


class MountainCarAgent:
    """
    @class MountainCarAgent
    @brief Implements a Q-learning agent for the MountainCar-v0 environment from OpenAI Gym.

    This agent uses a Q-table to learn an optimal policy for navigating the environment,
    balancing exploration and exploitation using an epsilon-greedy strategy.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.9,
        existing_q: np.ndarray = None,
        position_bins: int = 20,
        velocity_bins: int = 20,
    ):
        """
        @brief Constructor for the MountainCarAgent.

        @param env The environment in which the agent will operate.
        @param learning_rate Learning rate (alpha) for Q-value updates.
        @param initial_epsilon Initial value of epsilon for the epsilon-greedy policy.
        @param epsilon_decay Amount to decay epsilon after each episode.
        @param final_epsilon Minimum value that epsilon can decay to.
        @param discount_factor Discount factor (gamma) for future rewards.
        @param existing_q Optional existing Q-table to continue training from.
        @param position_bins Number of bins for discretizing the position state.
        @param velocity_bins Number of bins for discretizing the velocity state.
        """
        self.env = env
        if existing_q is None:
            self.q_values = np.zeros((position_bins, velocity_bins, env.action_space.n))
        else:
            self.q_values = existing_q

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        # Discretize position and velocity
        self.pos_bins = np.linspace(-1.2, 0.6, position_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, velocity_bins)

    def get_action(self, obs: int) -> int:
        """
        @brief Selects an action using an epsilon-greedy policy.

        @param obs The current observation (state) from the environment.
        @return The action to take (integer).
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            state_p, state_v = self._get_pos_vel_state(obs)
            return np.argmax(self.q_values[state_p, state_v, :])

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """
        @brief Updates the Q-value table based on the agent's experience.

        @param obs The current observation (state).
        @param action The action taken.
        @param reward The reward received.
        @param terminated Whether the episode has terminated.
        @param next_obs The next observation (state) after taking the action.
        """
        state_p, state_v = self._get_pos_vel_state(obs)
        next_state_p, next_state_v = self._get_pos_vel_state(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_state_p, next_state_v, :])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state_p, state_v, action]
        )

        self.q_values[state_p, state_v, action] = (
            self.q_values[state_p, state_v, action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        @brief Decays the epsilon value towards the final_epsilon.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def _get_pos_vel_state(self, obs: int) -> tuple:
        """
        @brief Converts the observation (obs) into a discrete state based on position and velocity.

        @param obs The observation from the environment.
        @return A tuple of (position_state, velocity_state).
        """

        position_state = np.digitize(obs[0], self.pos_bins)
        if position_state == len(self.pos_bins):
            position_state -= 1
        velocity_state = np.digitize(obs[1], self.vel_bins)
        if velocity_state == len(self.vel_bins):
            velocity_state -= 1
        return position_state, velocity_state