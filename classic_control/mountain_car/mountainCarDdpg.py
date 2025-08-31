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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from itertools import islice

def deque_slice(dq, start):
    # returns a list slice from start to end
    return list(islice(dq, start, None))

class MountainCarDNNAgent:

    def __init__(self,
                 env: gym.Env,
                 policy_learning_rate: float,
                 quality_learning_rate: float,
                 existing_model_path: Path = None,
                 save_path: Path = None,
                 update_rate: int = 2,
                 replay_memory_size: int = 30000,
                 discount_factor: float = 0.99,
                 optimizer: str = "adam",
                 log_directory: str = "logs/mountainCarDqn",
                 batch_size: int = 64,
                 hidden_layer_dims: List[int] = [16, 16],
                 max_episode_steps: int = 500,
                 save_interval: int = 200,
                 polyak_coefficient: float = 0.995,
                 pure_explore_steps: int = 50,
                 action_damping_factor: float = 1.0):

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
        self.action_damping = self.action_damping_factor


        self.graph_update_rate_seconds = 20 # Update graph every 20 seconds

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        self.save_path = save_path
        if save_path and not isinstance(save_path, Path):
            self.save_path =  Path(save_path)

            self.save_path.parent.mkdir(parents=True, exist_ok=True)

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
        @brief Compute the critic (Q-function) loss for DDPG.

        This method calculates the Mean Squared Error (MSE) between the predicted Q-values
        from the critic network and the Bellman backup target. The target is computed using
        the target actor-critic networks and the observed rewards.

        @param states
            Batch of current environment states. Shape: [B, state_dim].
        @param actions
            Batch of actions taken in each state. Shape: [B, action_dim].
        @param rewards
            Batch of rewards received after taking the actions. Shape: [B, 1] or [B].
        @param next_states
            Batch of next environment states observed after taking the actions. Shape: [B, state_dim].
        @param terminated
            Batch of done flags indicating whether the episode ended after the transition.
            Shape: [B, 1] or [B].

        @return
            A tuple (loss_q, loss_info):
            - loss_q (torch.Tensor): Scalar critic loss value (MSE).
            - loss_info (dict): Dictionary with auxiliary logging information,
              including "QVals" (numpy array of detached Q-values).
        """
        q_value = self.ac_net.q(states, actions)

        # Bellman backup for Q function
        # Compute y_i = r_i + \gamma Q_{target}(s_{i+1}, \pi_target(s_{i+1}))
        with torch.no_grad():
            q_pi_targ = self.target_net.q(next_states, self.target_net.pi(next_states))  # [B]
            rewards_1d      = rewards.view(-1)         # or rewards.squeeze(-1)
            terminated_1d   = terminated.view(-1)      # or terminated.squeeze(-1)
            backup = rewards_1d + self.discount_factor * (1 - terminated_1d) * q_pi_targ  # [B]


        # MSE loss against Bellman backup
        loss_q = F.mse_loss(q_value, backup)

        # Useful info for logging
        loss_info = dict(QVals=q_value.detach().cpu().numpy())

        return loss_q, loss_info

    def compute_policy_loss(self,
                            states: torch.Tensor) -> torch.Tensor:
        """
        @brief Compute the actor (policy) loss for DDPG.

        This method implements the deterministic policy gradient objective for the actor.
        The actor seeks to maximize the expected Q-value of its actions:
            J(θ) = E[ Q(s, πθ(s)) ].
        Since optimizers perform gradient descent, the loss is defined as:
            L_actor(θ) = - E[ Q(s, πθ(s)) ],
        so that minimizing the loss corresponds to maximizing the expected Q-value.

        @param states
            Batch of environment states used to evaluate the actor policy. Shape: [B, state_dim].

        @return
            Scalar tensor containing the actor loss (torch.Tensor).
            Minimization of this loss via backpropagation updates the policy parameters
            in the direction of the deterministic policy gradient.
        """

        return -self.ac_net.q(states, self.ac_net.pi(states)).mean()

    def get_optimizer(self, optimizer: str, model_parameters, lr: float):
        """
        @brief Create and return a PyTorch optimizer.

        This method initializes a PyTorch optimizer for the given model parameters
        based on the optimizer type specified. Currently supports Adam and SGD.

        @param optimizer
            Name of the optimizer to use. Supported values: "adam", "sgd" (case-insensitive).
        @param model_parameters
            Iterable of model parameters to be optimized (e.g., model.parameters()).
        @param lr
            Learning rate for the optimizer.

        @return
            A torch.optim.Optimizer instance corresponding to the requested optimizer.

        @throws ValueError
            If an unsupported optimizer name is provided.
        """
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
        @brief Perform a single optimization step for the actor-critic networks (DDPG).

        This method updates both the critic (Q-network) and the actor (policy network)
        using a mini-batch of experience transitions. It first computes the critic loss
        via the Bellman backup and performs a gradient descent step on the Q-network.
        Then it updates the actor by maximizing the Q-value of its actions through
        the deterministic policy gradient. Finally, the target networks are updated
        using Polyak averaging.

        @param mini_batch
            A tuple containing five lists corresponding to sampled transitions:
            - states: List[Tensor], current environment states.
            - actions: List[Tensor] or List[int], actions taken in each state.
            - new_states: List[Tensor], next states resulting from actions.
            - rewards: List[float], rewards received after taking actions.
            - terminations: List[bool], flags indicating if an episode ended.

        @return
            None. Updates are applied in-place to the actor, critic, and target networks.

        @details
            The optimization proceeds in three stages:
            1. **Critic update**: Minimize the MSE between predicted Q-values and Bellman targets.
            2. **Actor update**: Maximize the critic’s Q-value estimate for the actor’s actions
               by minimizing the negative Q-value loss.
            3. **Target network update**: Slowly update target actor-critic networks using Polyak
               averaging with factor `self.polyak`.

            Gradient clipping is applied to both networks for stability.
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


        # Page 5 of https://arxiv.org/pdf/1509.02971 for a good reference
        # First run one gradient descent step for Q.
        # Updates the critic by minimizing the loss
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_quality_loss(states, actions, rewards, new_states, terminations)
        loss_q.backward()
        clip_grad_norm_(self.ac_net.q.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.ac_net.q.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_policy_loss(states)
        loss_pi.backward()
        clip_grad_norm_(self.ac_net.pi.parameters(), max_norm=0.9)
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
        """
        @brief Train the actor-critic agent using the DDPG algorithm.

        Executes training for a specified number of episodes by interacting with the
        environment, storing transitions in replay memory, and updating both actor and
        critic networks. Supports an initial pure exploration phase before learning begins.

        @param num_episodes
            Number of training episodes to run.

        @return
            None. Training updates are applied in-place, and the learned model is saved
            periodically as well as at the end of training.

        @details
            The training loop proceeds as follows:
            1. **Pure exploration**:
               - For the first `self.pure_explore_steps` episodes, the agent acts randomly
                 (with added Ornstein-Uhlenbeck noise) to encourage exploration.
               - If `pure_explore_steps >= num_episodes`, it is reset to 20% of `num_episodes`.
            2. **Episode rollout**:
               - Reset environment and noise process at the beginning of each episode.
               - At each step, select an action with optional exploration noise.
               - Execute action in the environment, collect reward, next state, and termination flag.
               - Store both step transitions and episode transitions into replay memory.
               - If the episode ends in termination, push the last 20% of transitions (or all if < 200 steps)
                 into a separate success buffer.
            3. **Optimization**:
               - Once replay memory contains more samples than `self.batch_size` and pure exploration
                 has ended, sample minibatches and perform gradient updates via `self.optimize()`.
               - Both critic and actor are updated, and target networks are Polyak-averaged.
            4. **Logging and checkpointing**:
               - Log per-episode rewards and termination events.
               - Update training graphs periodically every `self.graph_update_rate_seconds` seconds.
               - Save the model whenever a new best reward is achieved, at regular intervals
                 (`self.save_interval`), and at the end of training.
            5. **Noise scheduling**:
               - Decay Ornstein-Uhlenbeck noise sigma after each episode.

            The method tracks cumulative rewards per episode and saves the final trained
            model to `self.save_path`.
        """
        
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
                                                  reward,
                                                  terminated))
                episode_memory.push(*self._as_tensor(self._normalize_state(state),
                                                     action, self._normalize_state(next_state),
                                                     reward, terminated))

                episode_reward += reward
                state = next_state
                done = terminated or step_count >= self.max_episode_steps

            rewards_per_episode.append(episode_reward)

            logger.debug(f"  [Episode] {episode + 1}: Reward = {episode_reward}, Steps = {step_count}, Explore = {explore} {', Terminated' if terminated else ''}")
            if terminated:
                # Only push the last 20% of the episode, or all if less than 200 steps
                n = len(episode_memory)
                if n < 200:
                    idx_start = 0
                else:
                    idx_start = int(n * 0.8)
                self.memory.push_success(
                    deque_slice(episode_memory.states, idx_start),
                    deque_slice(episode_memory.actions, idx_start),
                    deque_slice(episode_memory.next_states, idx_start),
                    deepcopy(deque_slice(episode_memory.rewards, idx_start)),
                    deque_slice(episode_memory.terminated, idx_start)
                )
                logger.debug(f"    [Episode] {episode + 1}: Terminated after {step_count} steps, with reward {episode_reward}")

            episode_memory.clear()

            if len(self.memory) > self.batch_size and update_count > self.pure_explore_steps:

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
            self.ou_noise.decay_sigma()


        self.ac_net.save_model(self.save_path)
        logger.info(f"Training complete. Model saved to {self.save_path}")


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
        """
        @brief Evaluate the current actor policy over multiple episodes.

        Runs the actor network in evaluation mode for a specified number of episodes
        without exploration noise. For each episode, the environment is reset, and the
        policy selects actions deterministically until termination or until the maximum
        episode length is reached. Tracks cumulative rewards and termination flags.

        @param num_episodes
            Number of evaluation episodes to run.

        @return
            None. Logs per-episode rewards and termination status, and reports the mean
            reward across all evaluation episodes.

        @details
            - The actor network is switched to evaluation mode (`self.ac_net.eval()`).
            - Ornstein-Uhlenbeck noise is reset at the start of each episode but not applied
              to actions during evaluation (`induce_noise=False`).
            - For each step, the actor selects an action deterministically via
              `self.get_action(state, random=False, induce_noise=False)`.
            - Episodes terminate either when the environment signals termination or when the
              maximum episode length (`self.max_episode_steps`) is reached.
            - Rewards are accumulated per episode, and both rewards and termination flags are
              stored for logging.
            - After evaluation, mean reward across episodes is computed and logged.
        """

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
                logger.debug(f"  [Episode {episode + 1}] Step {step_count}: Action = {action}, State = {state}, Normalized State = {self._normalize_state(state)}")
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
        @brief Normalize an observation state to the range [-1, 1].

        Maps the raw environment observation values from their original range
        defined by the environment’s observation space `[low, high]` into a
        normalized range of [-1, 1].

        @param state
            Raw state observation as a NumPy array. Shape matches the environment’s
            observation space.

        @return
            Normalized state as a NumPy array with values scaled to the range [-1, 1].

        @details
            The normalization is performed in two steps:
            1. Shift and scale to map `[low, high]` → [0, 1].
            2. Linearly transform [0, 1] → [-1, 1].
        """

        low, high = self.env.observation_space.low, self.env.observation_space.high
        norm01 = (state - low) / (high - low)  # [0..1]
        return norm01 * 2.0 - 1.0               # now in [-1..+1]
