import gymnasium as gym
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from common.loggerConfig import logger
from common.dnnActorCritic import DnnActorCritic
from common.replayMemory import ReplayMemory
from copy import deepcopy
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from common.utils import get_moving_avgs
from common.ornsteinUhlenbeck import OrnsteinUhlenbeck

class MountainCarDNNAgent:

    def __init__(self,
                 env: gym.Env,
                 policy_learning_rate: float,
                 quality_learning_rate: float,
                 existing_model_path: Path = None,
                 save_path: Path = None,
                 update_rate: int = 5,
                 replay_memory_size: int = 30000,
                 discount_factor: float = 0.999,
                 optimizer: str = "adam",
                 log_directory: str = "logs/mountainCarDqn",
                 batch_size: int = 64,
                 hidden_layer_dims: List[int] = [16, 16],
                 max_episode_steps: int = 500,
                 save_interval: int = 500,
                 polyak_coefficient: float = 0.9,
                 pure_explore_steps: int = 2000):

        self.env = env

        self.pi_lr = policy_learning_rate
        self.q_lr = quality_learning_rate
        self.discount_factor = discount_factor
        self.polyak =  polyak_coefficient

        self.pure_explore_steps = pure_explore_steps

        self.update_rate = update_rate
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.max_episode_steps = max_episode_steps
        self.ou_noise = OrnsteinUhlenbeck()
        self.action_damping = 0.5 # TODO: Make this a parameter


        self.graph_update_rate_seconds = 20 # Update graph every 20 seconds

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        self.save_path = save_path
        if save_path and not isinstance(save_path, Path):
            self.save_path =  Path(save_path)

            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO
        # Load existing model if provided
        if isinstance(existing_model_path, str):
            existing_model_path = Path(existing_model_path)
        if existing_model_path and existing_model_path.exists():
            try:
                ac_net = DnnActorCritic.load_model(existing_model_path, self.device) #TODO: Remove this device
                self.ac_net = ac_net
            except Exception as e:
                logger.warning(f"Failed to load the model from {existing_model_path}: {e}")
                logger.info(f"Switching to default model.")
                self.ac_net.load_state_dict(torch.load(existing_model_path))
                logger.info(f"Loaded existing DQN from {existing_model_path}")
            self.target_net = deepcopy(self.ac_net)
            self.target_net.to(self.device)

        else:
            # DQN and target network
            self.ac_net = DnnActorCritic(state_dim=self.state_dim,
                                         action_dim=self.action_dim,
                                         hidden_dims=hidden_layer_dims,
                                         action_limit=self.action_limit,
                                         device=self.device)

            self.target_net = deepcopy(self.ac_net)

            self.ac_net.to(self.device)
            self.target_net.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer for policy and Q function
        self.pi_optimizer = self.get_optimizer(optimizer, self.ac_net.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = self.get_optimizer(optimizer, self.ac_net.q.parameters(), lr=self.q_lr)
        # Loss Function
        self.loss_fn = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)


        self.graph_file = Path(log_directory) / 'mountainCarDQN.png'
        self.graph_file.parent.mkdir(parents=True, exist_ok=True)

    def compute_quality_loss(self,
                             states: torch.Tensor,
                             actions: torch.Tensor,
                             rewards: torch.Tensor,
                             next_states: torch.Tensor,
                             terminated: torch.Tensor) -> torch.Tensor:
        """
        Method for computing DDPG Q-loss
        """
        # TODO: Add doxygen
        q_value = self.ac_net.q(states, actions)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.target_net.q(next_states, self.target_net.pi(next_states))
            q_pi_targ = q_pi_targ.unsqueeze(-1)
            backup = rewards + self.discount_factor * (1 - terminated) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q_value - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q_value.detach().cpu().numpy())

        return loss_q, loss_info

    def compute_policy_loss(self,
                            states: torch.Tensor) -> torch.Tensor:
        """
        Method for computing DDPG policy loss
        """

        # TODO: Add doxygen

        return -self.ac_net.q(states, self.ac_net.pi(states)).mean()

    def get_optimizer(self, optimizer: str, model_parameters, lr: float):
        # TODO: Add doxygen
        optimizer = optimizer.lower()
        if optimizer == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Supported optimizers are 'adam' and 'sgd'.")

    def get_action(self,
                   state: np.ndarray,
                   random: bool = False,
                   induce_noise: bool = False) -> np.ndarray:
        """
        @brief Selects an action to take in the environment based on the current state.

        @param state The current observation (state) from the environment.
        @param random If True, selects a random action for exploration; otherwise, selects an action based on the policy.
        @param induce_noise Optional noise to add to the action for exploration purposes.
        @return The action to take (np.ndarray).
        """

        if random:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_t, = self._as_tensor(self._normalize_state(state))
                a = self.ac_net.act(state_t)
                if induce_noise:
                    a = a * self.action_damping + self.ou_noise.step()
                # logger.debug(f"  [Uncapped Action] {a}")
                return np.clip(a, -self.action_limit, self.action_limit)

    def optimize(self,
                 mini_batch: Tuple[List[Tensor], List[Tensor], List[Tensor], List[float], List[bool]]) -> None:
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

        # Conditionally unsqueeze actions if it's 1D
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)   # now [32, 1]

        # logger.debug(
        #     f"shapes → states: {states.shape}, "
        #     f"actions: {actions.shape}, "
        #     f"new_states: {new_states.shape}, "
        #     f"rewards: {rewards.shape}, "
        #     f"terminations: {terminations.shape}"
        # )

        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_quality_loss(states, actions, rewards, new_states, terminations)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.ac_net.q.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_policy_loss(states)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.ac_net.q.parameters():
            param.requires_grad = True

        logger.debug(f"   [Optimize] - Loss Q {loss_q.item():.4f}, Loss Pi {loss_pi.item():.4f}") # , Loss Info: {loss_info}

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for ac_param, target_param in zip(self.ac_net.parameters(), self.target_net.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * ac_param.data)

    def train(self,
              num_episodes: int):

        logger.info(f"Beginning training for {num_episodes} episodes.")
        if self.pure_explore_steps >= num_episodes:
            self.pure_explore_steps = int(0.2 * num_episodes)
            logger.warning(f"Pure exploration steps ({self.pure_explore_steps}) is greater than or equal to total episodes ({num_episodes}). Setting pure exploration steps to 20% of total episodes.")

        start_time = datetime.now()
        last_graph_update_time = start_time

        logger.info(f"[Train Start]– "
            f"{num_episodes or '∞'} episodes, policy lr={self.pi_lr}, quality lr={self.q_lr} γ={self.discount_factor}")

        rewards_per_episode = []

        best_reward = float("-inf")
        update_count = 0
        save_count = 0
        for episode in range(num_episodes):
            self.ou_noise.reset()
            state, info = self.env.reset()
            update_count += 1
            next_state = state
            terminated, truncated = False, False
            episode_reward, reward = 0.0, 0.0
            done = False
            step_count = 0
            episode_memory = ReplayMemory(self.max_episode_steps)

            explore = True if episode < self.pure_explore_steps else False

            while not done:
                step_count += 1
                action = self.get_action(state, explore, induce_noise=True)
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

            logger.debug(f"  [Episode] {episode + 1}: Reward = {episode_reward}, Steps = {step_count}, Explore = {explore} {', Terminated' if terminated else ''}")
            if terminated:
                self.memory.push_success(episode_memory.states,
                                         episode_memory.actions,
                                         episode_memory.next_states,
                                         self._propagate_success(deepcopy(episode_memory.rewards)),
                                         episode_memory.terminated)
                logger.debug(f"    [Episode] {episode + 1}: Terminated after {step_count} steps, with reward {episode_reward}")

            episode_memory.clear()

            if len(self.memory) > self.batch_size and update_count % self.update_rate == 0:

                for _ in range(self.update_rate):
                    mini_batch = self.memory.sample(self.batch_size)
                    self.optimize(mini_batch)

            # Update graph every x seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=self.graph_update_rate_seconds):
                self.save_graph(rewards_per_episode)
                last_graph_update_time = current_time

            if episode_reward > best_reward:
                best_reward = episode_reward
                logger.info(f"New best reward: {best_reward} at episode {episode + 1}")
                self.ac_net.save_model(self.save_path)


            if save_count >= self.save_interval:
                save_count = 0
                self.ac_net.save_model(self.save_path)
            save_count += 1


        self.ac_net.save_model(self.save_path)
        logger.info(f"Training complete. Model saved to {self.save_path}")


    def _propagate_success(self, rewards, ratio=0.97):
        propagated = rewards[-1]
        # go from last index −1 down to 0 inclusive
        for i in range(len(rewards)-1, -1, -1):
            # take whatever is larger: your original reward or the decayed later reward
            rewards[i] = max(rewards[i], propagated)
            propagated *= ratio
        return rewards


    def _as_tensor(self, *args) -> Tuple[Tensor, ...]:
        """
        Converts a tuple of arguments into a tuple of tensors.
        """
        return tuple(torch.as_tensor(arg, dtype=torch.float32, device=self.device) for arg in args)


    def save_graph(self, rewards_per_episode: List[float]) -> None:
        """
        @brief Saves a plot showing the training progress over episodes.

        This function generates and saves two side-by-side line plots:
        - The moving average of rewards per episode (window size 100).


        Both plots are saved as a single image file to `self.graph_file`.

        @param rewards_per_episode List of total rewards obtained in each episode.

        """
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = get_moving_avgs(rewards_per_episode, 100, "same")

        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.graph_file)
        plt.close(fig)

    def evaluate(self, num_episodes: int):
        logger.info(f"Beginning evaluation: {num_episodes} episodes")
        self.ac_net.eval()
        logger.debug(f"Evaluation mode enabled.")
        rewards_per_episode = []
        terminated_per_episode = []
        for episode in tqdm(range(num_episodes), desc="Episodes", leave=False):
            self.ou_noise.reset()
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            terminated = False
            while not done:
                step_count += 1
                action = self.get_action(state, random=False, induce_noise=False)
                logger.debug(f"  [Episode {episode + 1}] Step {step_count}: Action = {action}, State = {state}")
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
        """
        Mapping the values to the range [-1, 1]
        """
        low, high = self.env.observation_space.low, self.env.observation_space.high
        norm01 = (state - low) / (high - low)  # [0..1]
        return norm01 * 2.0 - 1.0               # now in [-1..+1]

    def _normalize_reward(self, reward: float) -> float:
        """
        Mapping the rewards to the range [0, 1]
        """
        _min = -0.1*self.max_episode_steps
        _max = 100.0
        return (reward - _min) / (_max - _min)