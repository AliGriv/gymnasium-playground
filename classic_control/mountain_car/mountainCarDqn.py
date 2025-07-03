import gymnasium as gym
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from common.loggerConfig import logger
from common.dqn import DQN
from common.replayMemory import ReplayMemory
from copy import deepcopy
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from common.utils import get_moving_avgs


class MountainCarDQNAgent:

    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float = None,
                 existing_dqn_path: Path = None,
                 save_dqn_path: Path = None,
                 network_sync_rate: int = 500,
                 replay_memory_size: int = 30000,
                 discount_factor: float = 0.9,
                 optimizer: str = "adam",
                 log_directory: str = "logs/mountainCarDqn",
                 batch_size: int = 32,
                 enable_dueling: bool = False,
                 enable_double: bool = False,
                 hidden_layer_dims: List[int] = [12, 4],
                 max_episode_steps: int = 500,
                 clip_grad_norm: float = 1.0,
                 save_interval: int = 500):

        self.env = env

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = 0.0

        self.network_sync_rate = network_sync_rate
        self.batch_size = batch_size
        self.enable_dqn_double = enable_double
        self.enable_dqn_dueling = enable_dueling
        self.save_interval = save_interval
        self.clip_grad_norm = clip_grad_norm
        self.max_episode_steps = max_episode_steps


        self.graph_update_rate_seconds = 20 # Update graph every 20 seconds

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.save_path = save_dqn_path
        if save_dqn_path and not isinstance(save_dqn_path, Path):
            self.save_path =  Path(save_dqn_path)

            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing model if provided
        if isinstance(existing_dqn_path, str):
            existing_dqn_path = Path(existing_dqn_path)
        if existing_dqn_path and existing_dqn_path.exists():
            try:
                q_net = DQN.load_model(existing_dqn_path, self.device) #TODO: Remove this device
                self.q_net = q_net
            except Exception as e:
                logger.warning(f"Faield to load the model from {existing_dqn_path}: {e}")
                logger.info(f"Switching to default model.")
                self.q_net.load_state_dict(torch.load(existing_dqn_path))
                logger.info(f"Loaded existing DQN from {existing_dqn_path}")
            self.target_net = deepcopy(self.q_net)
            self.target_net.to(self.device)

        else:
            # DQN and target network
            self.q_net = DQN(state_dim=self.state_dim,
                             action_dim=self.action_dim,
                             hidden_dims=hidden_layer_dims,
                             enable_dueling_dqn=self.enable_dqn_dueling,
                             tanh_output=False)

            self.target_net = DQN(state_dim=self.state_dim,
                                  action_dim=self.action_dim,
                                  hidden_dims=hidden_layer_dims,
                                  enable_dueling_dqn=self.enable_dqn_dueling,
                                  tanh_output=False)

            self.q_net.to(self.device)
            self.target_net.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer, self.q_net.parameters(), lr=self.lr)
        # Loss Function
        self.loss_fn = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)


        self.graph_file = Path(log_directory) / 'mountainCarDQN.png'
        self.graph_file.parent.mkdir(parents=True, exist_ok=True)

    def get_optimizer(self, optimizer: str, model_parameters, lr: float):
        # TODO: Add doxygen
        optimizer = optimizer.lower()
        if optimizer == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Supported optimizers are 'adam' and 'sgd'.")

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        @brief Selects an action using an epsilon-greedy policy.

        @param obs The current observation (state) from the environment.
        @return The action to take (np.ndarray).
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            with torch.no_grad():
                state_t, = self._as_tensor(self._normalize_state(state))
                q_value = self.q_net(state_t)
                return q_value.cpu().numpy()

    def decay_epsilon(self):
        # TODO: Add doxygen
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def sync_target_network(self):
        # TODO: Add doxygen
        logger.debug(f"  [Sync] – target network updated")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def optimize(self,
                mini_batch: Tuple[List[Tensor], List[int], List[Tensor], List[float], List[bool]],
                policy_dqn: Module,
                target_dqn: Module) -> None: # TODO: Step count may be removed
        """
        @brief Performs a single optimization step on the policy DQN using a mini-batch of experiences.

        Computes the loss between the predicted Q-values from the policy network and the target Q-values
        estimated using the Bellman equation. Supports both standard DQN and Double DQN update strategies.

        @param mini_batch A tuple containing five elements:
                        - states: List of tensors representing current states.
                        - actions: List of integers representing actions taken.
                        - new_states: List of tensors representing resulting states.
                        - rewards: List of floats representing rewards received.
                        - terminations: List of booleans indicating if the episode terminated.

        @param policy_dqn The DQN model currently being trained (policy network).
        @param target_dqn The target DQN used for estimating future Q-values.

        @return None
        """

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = mini_batch

        # Stack tensors to create batch tensors

        states = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=self.device) for s in states]).to(self.device)
        actions = torch.as_tensor([action for action in actions], dtype=torch.float32, device=self.device)
        new_states = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=self.device) for s in new_states]).to(self.device)
        rewards = torch.as_tensor([reward for reward in rewards], dtype=torch.float32, device=self.device).unsqueeze(1)
        terminations = torch.as_tensor([t for t in terminations], dtype=torch.float32, device=self.device).unsqueeze(1)


        with torch.no_grad():
            if self.enable_dqn_double:
                best_actions_from_policy = policy_dqn(new_states)

                target_q = rewards + (1-terminations) * self.discount_factor * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states)



        current_q = policy_dqn(states)

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        # every N steps or batches
        logger.debug(f"  [Optimize] loss={loss.item():.4f}, ε={self.epsilon:.3f}")

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_grad_norm) # TODO: Do I really need this?
        self.optimizer.step()       # Update network parameters i.e. weights and biases

    def train(self,
              num_episodes: int):

        logger.info(f"Beginning training for {num_episodes} episodes.")
        if self.epsilon_decay is None:
            self.epsilon_decay = (self.epsilon - self.final_epsilon) / (num_episodes * 0.95)

        logger.info(f"Epsilon decay set to {self.epsilon_decay} per episode.")

        start_time = datetime.now()
        last_graph_update_time = start_time

        logger.info(f"[Train Start]– "
            f"{num_episodes or '∞'} episodes, lr={self.lr}, γ={self.discount_factor}, "
            f"dueling={self.enable_dqn_dueling}, double={self.enable_dqn_double}")

        rewards_per_episode = []
        epsilon_history = []
        best_reward = float("-inf")
        self.sync_target_network()
        sync_count = 0
        save_count = 0
        for episode in range(num_episodes):
            state, info = self.env.reset()
            next_state = state
            terminated, truncated = False, False
            episode_reward, reward = 0.0, 0.0
            done = False
            step_count = 0
            episode_memory = ReplayMemory(self.max_episode_steps)
            while not done:
                step_count += 1
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.memory.push(*self._as_tensor(self._normalize_state(state),
                                                  action,
                                                  self._normalize_state(next_state),
                                                  self._normalize_reward(reward),
                                                  terminated))
                episode_memory.push(*self._as_tensor(self._normalize_state(state),
                                                     action, self._normalize_state(next_state),
                                                     self._normalize_reward(reward), terminated))

                episode_reward += reward
                state = next_state
                done = terminated or step_count >= self.max_episode_steps

            rewards_per_episode.append(episode_reward)
            epsilon_history.append(self.epsilon)
            logger.debug(f"  [Episode] {episode + 1}: Reward = {episode_reward}, Epsilon = {self.epsilon:.3f}, Steps = {step_count} {', Terminated' if terminated else ''}")
            if terminated:
                self.memory.push_success(episode_memory.states,
                                         episode_memory.actions,
                                         episode_memory.next_states,
                                         self._propagate_success(episode_memory.rewards),
                                         episode_memory.terminated)
                logger.debug(f"    [Episode] {episode + 1}: Terminated after {step_count} steps, with reward {episode_reward}")

            episode_memory.clear()

            if len(self.memory) > self.batch_size:
                mini_batch = self.memory.sample(self.batch_size)
                self.optimize(mini_batch, self.q_net, self.target_net)
                sync_count += 1

                if sync_count > self.network_sync_rate:
                    self.sync_target_network()
                    sync_count=0

            # Update graph every x seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=self.graph_update_rate_seconds):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

            if episode_reward > best_reward:
                best_reward = episode_reward
                logger.info(f"New best reward: {best_reward} at episode {episode + 1}")
                self.q_net.save_model(self.save_path)
            self.decay_epsilon()
            if save_count >= self.save_interval:
                save_count = 1
                self.q_net.save_model(self.save_path)
            else:
                save_count += 1


        self.q_net.save_model(self.save_path)
        logger.info(f"Training complete. Model saved to {self.save_path}")


    def _propagate_success(self, rewards, ratio: float = 0.97):
        propagated_value = rewards[-1]
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i] = max(rewards[i], propagated_value)
            propagated_value = propagated_value*ratio

        return rewards


    def _as_tensor(self, *args) -> Tuple[Tensor, ...]:
        """
        Converts a tuple of arguments into a tuple of tensors.
        """
        return tuple(torch.as_tensor(arg, dtype=torch.float32, device=self.device) for arg in args)


    def save_graph(self, rewards_per_episode: List[float], epsilon_history: List[float]) -> None:
        """
        @brief Saves a plot showing the training progress over episodes.

        This function generates and saves two side-by-side line plots:
        - The moving average of rewards per episode (window size 100).
        - The epsilon value (exploration rate) over time.

        Both plots are saved as a single image file to `self.graph_file`.

        @param rewards_per_episode List of total rewards obtained in each episode.
        @param epsilon_history List of epsilon values over training steps or episodes.
        """
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = get_moving_avgs(rewards_per_episode, 100, "same")

        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.graph_file)
        plt.close(fig)

    def evaluate(self, num_episodes: int):
        logger.info(f"Beginning evaluation: {num_episodes} episodes")
        self.q_net.eval()
        logger.debug(f"Evaluation mode enabled.")
        self.epsilon = 0.0
        rewards_per_episode = []
        terminated_per_episode = []
        for episode in tqdm(range(num_episodes), desc="Episodes", leave=False):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            terminated = False
            while not done:
                step_count += 1
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward


                done = terminated or step_count >= self.max_episode_steps
                state = next_state

            rewards_per_episode.append(episode_reward)
            terminated_per_episode.append(terminated)

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}: Reward = {rewards_per_episode[episode]}, Terminated = {terminated_per_episode[episode]}")
        mean_reward = np.mean(rewards_per_episode)
        logger.info(f"Evaluation complete. Mean reward: {mean_reward:.2f} over {num_episodes} episodes.")


    def _normalize_state(self, state: np.ndarray) -> np.ndarray:

        _min = self.env.observation_space.low
        _max = self.env.observation_space.high

        return (state - _min) / (_max - _min)

    def _normalize_reward(self, reward: float) -> float:
        _min = -0.1*self.max_episode_steps
        _max = 100.0
        return (reward - _min) / (_max - _min)