import gymnasium as gym
import numpy as np
from common.loggerConfig import logger

class MountainCarContAgent:
    """
    @class MountainCarContAgent
    @brief Implements a Q-learning agent for the MountainCarContinuous-v0 environment from OpenAI Gym.

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
        actions_bins: int = 25,
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
            self.q_values = np.zeros((position_bins, velocity_bins, actions_bins))
        else:
            self.q_values = existing_q

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        # Discretize position and velocity
        self.num_position_bins = position_bins
        self.num_velocity_bins = velocity_bins
        self.num_actions_bins = actions_bins
        self.pos_bins = np.linspace(-1.2, 0.6, position_bins)
        self.vel_bins = np.linspace(-0.07, 0.07, velocity_bins)
        self.act_bins = np.linspace(-1.0, 1.0, actions_bins)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
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
            act_idx = np.argmax(self.q_values[state_p, state_v, :])
            return np.array([self.act_bins[act_idx]])

    def update(
        self,
        obs: float,
        action: float,
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
        action_index = self._get_action_index(action)
        next_state_p, next_state_v = self._get_pos_vel_state(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_state_p, next_state_v, :])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state_p, state_v, action_index]
        )

        self.q_values[state_p, state_v, action_index] = (
            self.q_values[state_p, state_v, action_index] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference.item() if isinstance(temporal_difference, np.ndarray) else temporal_difference)

    def rewind_episode(self, episode_history: list, last_reward_influence: float = 0.98):
        last_item = episode_history[-1]
        last_reward = last_item[2]
        for obs, action, reward, terminated, next_obs in reversed(episode_history):
            reward = max(reward, last_reward)
            self.update(obs, action, reward, terminated, next_obs)
            last_reward = last_reward * last_reward_influence

    def get_meta_data(self) -> dict:
        """
        @brief Returns metadata about the agent.

        @return A dictionary containing the agent's metadata.
        """
        return {
            'num_position_bins': self.num_position_bins,
            'num_velocity_bins': self.num_velocity_bins,
            'num_actions_bins': self.num_actions_bins,
        }

    def decay_epsilon(self):
        """
        @brief Decays the epsilon value towards the final_epsilon.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def _get_pos_vel_state(self, obs: np.ndarray) -> tuple:
        """
        @brief Converts the observation (obs) into a discrete state based on position and velocity.

        @param obs The observation from the environment.
        @return A tuple of (position_state, velocity_state).
        """

        position_state = np.digitize(obs[0], self.pos_bins)
        velocity_state = np.digitize(obs[1], self.vel_bins)
        return position_state, velocity_state

    def _get_action_index(self, action: np.ndarray) -> int:
        """
        @brief Converts the continuous action into an index based on the action bins.

        @param action The continuous action from the environment.
        @return The index of the action in the action bins.
        """
        action_index = np.digitize(action, self.act_bins)

        if action_index >= self.act_bins.shape[0]:
            action_index = self.act_bins.shape[0] - 1
        return action_index