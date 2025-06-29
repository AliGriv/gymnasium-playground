import itertools
import random
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn import Module

from common.dqn import DQN
from common.replayMemory import ReplayMemory
from common.utils import get_moving_avgs
from common.loggerConfig import logger


def set_seed(seed: int = 2025, use_cuda: bool = False):
    """
    @brief Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function ensures deterministic behavior by setting the same seed for Python's
    `random` module, NumPy, and PyTorch. If CUDA is used, it also sets the CUDA-specific
    seeds and enforces deterministic behavior in cuDNN.

    @param seed The seed value to use for all random number generators. Default is 2025.
    @param use_cuda Boolean flag indicating whether to set CUDA-specific seeds and
                    enforce deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: # TODO
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


class FrozenLakeAgent:
    # TODO: Add doxygen
    def __init__(self,
                 maps: Dict[str, List[str]],
                 learning_rate: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 network_sync_rate: int = 500,
                 replay_memory_size: int = 10000,
                 batch_size: int = 32,
                 enable_dqn_dueling: bool = False,
                 enable_dqn_double: bool = False,
                 hidden_layer_dims: List[int] = [64, 16],
                 stop_on_reward: Optional[int] = 10000,
                 discount_factor: float = 0.95,
                 existing_dqn_path: Optional[Path] = None,
                 optimizer: str = "adam",
                 log_directory: str = "logs/frozenLake",
                 is_slippery: bool = False,
                 save_dqn_path: Optional[Path] = Path("models/toy_text/frozenLakeDaqn.pt"),
                 save_interval: int = 500,
                 clip_grad_norm: float = 1.0,
                 max_episode_steps: int = 500):

        self.env = None
        self.maps = maps
        self.size = len(next(iter(self.maps.values())))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.is_slippery = is_slippery

        self.epsilon = initial_epsilon
        self.epsilon_decay = None
        self.final_epsilon = final_epsilon

        self.network_sync_rate = network_sync_rate
        self.batch_size = batch_size
        self.stop_on_reward = stop_on_reward
        self.enable_dqn_double = enable_dqn_double
        self.enable_dqn_dueling = enable_dqn_dueling
        self.save_interval = save_interval
        self.clip_grad_norm = clip_grad_norm
        self.max_episode_steps = max_episode_steps
        self.episode_history = set()
        self.state_action_history = {}
        self.q_net = None
        self.target_net = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(2025, use_cuda=torch.cuda.is_available())
        # Dimensions
        state_dim = 2*self.size**2 # TODO: Extend this
        action_dim = 4


        self.save_path = save_dqn_path

        if save_dqn_path:
            if not isinstance(save_dqn_path, Path):
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
            self.q_net = DQN(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dims=hidden_layer_dims,
                             enable_dueling_dqn=enable_dqn_dueling)

            self.target_net = DQN(state_dim=state_dim,
                                 action_dim=action_dim,
                                 hidden_dims=hidden_layer_dims,
                                 enable_dueling_dqn=enable_dqn_dueling)

            self.q_net.to(self.device)
            self.target_net.to(self.device)

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer, self.q_net.parameters(), lr=self.lr)
        # Loss Function
        self.loss_fn = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)

        self.log_file = Path(log_directory) / 'frozenLake.log'
        self.graph_file = Path(log_directory) / 'frozenLake.png'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


    def get_optimizer(self, optimizer: str, model_parameters, lr: float):
        # TODO: Add doxygen
        optimizer = optimizer.lower()
        if optimizer == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Supported optimizers are 'adam' and 'sgd'.")

    def get_action(self, state: torch.tensor) -> int:
        # TODO: Add doxygen
        if np.random.random() < self.epsilon:
            action =  self.env.action_space.sample()
            return action
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()


    def decay_epsilon(self):
        # TODO: Add doxygen
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def sync_target_network(self):
        # TODO: Add doxygen
        logger.debug(f"  [Sync] – target network updated")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self,
              num_trains: int = 20,
              num_subtrains: int = 10,
              map_batch_count: int = 5,
              num_episodes: Optional[int] = None,
              render: bool = False):
        """Public API for training the agent."""
        logger.info(f"Beginning training: up to {num_trains or '∞'} episodes")
        # set up epsilon decay and target network
        self._init_epsilon_decay(num_trains)
        self.sync_target_network()
        # delegate to internal loop
        self._train_loop(num_trains, num_subtrains, map_batch_count, num_episodes, render)

    def evaluate(self,
                 num_episodes: Optional[int] = None,
                 render: bool = False) -> List[float]:
        """Public API for evaluating the agent."""
        logger.info(f"Beginning evaluation: {num_episodes or 'all'} episodes")
        return self._eval_loop(num_episodes, render)

    def _init_epsilon_decay(self, num_episodes: Optional[int]):
        span = num_episodes * 0.95 if num_episodes else 1000  # TODO: Avoid magic number
        self.epsilon_decay = (self.epsilon - self.final_epsilon) / span
        logger.debug(f"  [Init] ε={self.epsilon:.3f}, ε_decay={self.epsilon_decay:.6f}, final ε={self.final_epsilon:.3f}")

    def _get_map_order(self, shuffle: bool) -> List[List[str]]:
        keys = list(self.maps.keys())
        return [self.maps[i] for i in (random.sample(keys, len(keys)) if shuffle else keys)]

    def _make_env(self, desc: List[str], render: bool) -> gym.Env:
        self.update_current_map(desc)
        return gym.make(
            'FrozenLake-v1',
            desc=desc,
            map_name="8x8" if self.size == 8 else "4x4",
            is_slippery=self.is_slippery,
            render_mode='human' if render else None,
            max_episode_steps=self.max_episode_steps
        )

    def _train_loop(self,
                    num_trains: int,
                    num_subtrains: int,
                    map_batch_count: int,
                    num_episodes: Optional[int],
                    render: bool):
        # shared preamble
        logger.info(f"Training starting for up to {num_episodes or '∞'} episodes")
        self.sync_target_network()
        rewards_per_episode = []
        epsilon_history = []
        best_reward = float("-inf")
        step_count = super_step_count = 0
        last_graph_time = datetime.now()

        map_order = self._get_map_order(shuffle=True)
        for train_round in range(num_trains):
            for map_count, desc in enumerate(map_order, start=1):
                self.env = self._make_env(desc, render)
                # gather experience on this map
                last_reward = self._run_episodes(self.env, desc, num_episodes, rewards_per_episode, is_training=True)
                # maybe save best model
                if last_reward > best_reward:
                    self._save_best_model(last_reward, best_reward)
                    best_reward = last_reward
                # periodically train on replay memory
                if len(self.memory) >= num_subtrains * self.batch_size:
                    batches = self.memory.sample(self.batch_size, num_subtrains)
                    for mb in batches:
                        self.optimize(mb, self.q_net, self.target_net, super_step_count)
                        step_count += 1
                        super_step_count += 1
                        if step_count > self.network_sync_rate:
                            self.sync_target_network()
                            step_count = 0
                # maybe update graph
                if datetime.now() - last_graph_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    epsilon_history.append(self.epsilon)
                    last_graph_time = datetime.now()
            self.decay_epsilon()
            logger.info(f"Completed train round {train_round+1}/{num_trains}")
            self.q_net.save_model(self.save_path)
        # final save
        self.q_net.save_model(self.save_path)
        logger.info(f"Training complete. Model saved to {self.save_path}")

    def _eval_loop(self,
                   num_episodes: Optional[int],
                   render: bool):
        logger.info("Evaluation mode")
        self.q_net.eval()
        rewards = []
        map_order = self._get_map_order(shuffle=False)
        for desc in map_order:
            self.env = self._make_env(desc, render)
            last_reward = self._run_episodes(self.env, desc, num_episodes, rewards, is_training=False)
        return rewards

    def _run_episodes(self,
                      env: gym.Env,
                      desc: List[str],
                      max_episodes: Optional[int],
                      rewards_acc: List[float],
                      is_training: bool) -> float:
        """
        Run episodes on a single env/map. Returns last episode's reward.
        Appends each episode_reward to rewards_acc.
        """
        episode = 0
        last_reward = 0.0
        while True:
            if max_episodes is not None and episode >= max_episodes:
                break
            obs, _ = env.reset()
            state = self.get_state(obs, desc)
            episode_memory = ReplayMemory(1000)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                action = self.get_action(state)
                new_obs, reward, terminated, truncated, _ = env.step(action)
                reward = self.override_reward(obs, new_obs, reward, terminated)
                next_state = self.get_state(new_obs, desc)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if is_training:
                    # store in both main memory and this episode
                    self.memory.push(state, action, next_state, torch.tensor(reward, device=self.device), terminated)
                    episode_memory.push(state, action, next_state, torch.tensor(reward, device=self.device), terminated)

                state, obs = next_state, new_obs
                episode_reward += reward

            rewards_acc.append(float(episode_reward))
            # if we reached goal, stash successful trajectory
            if is_training and new_obs == (self.size**2 - 1):
                self.memory.push_success(
                    episode_memory.states,
                    episode_memory.actions,
                    episode_memory.next_states,
                    episode_memory.rewards,
                    episode_memory.terminated
                )
            episode += 1
            last_reward = episode_reward
        return last_reward

    def _save_best_model(self, new_reward: float, old_reward: float):
        pct = (new_reward - old_reward)/old_reward*100 if old_reward != 0 else float("inf")
        msg = f"New best reward {new_reward:.1f} ({pct:+.1f}%)—saving model."
        logger.debug(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
        self.q_net.save_model(self.save_path)

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



    def optimize(self,
                 mini_batch: Tuple[List[Tensor], List[int], List[Tensor], List[float], List[bool]],
                 policy_dqn: Module,
                 target_dqn: Module,
                 step_count: int) -> None: # TODO: Step count may be removed
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
        @param step_count The current step count (used for logging or conditional updates).

        @return None
        """

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = mini_batch

        # Stack tensors to create batch tensors

        states = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=self.device) for s in states]).to(self.device)
        actions = torch.as_tensor([action for action in actions], dtype=torch.long, device=self.device)
        new_states = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=self.device) for s in new_states]).to(self.device)
        rewards = torch.as_tensor([reward for reward in rewards], dtype=torch.float32, device=self.device).unsqueeze(1)
        terminations = torch.as_tensor([t for t in terminations], dtype=torch.float32, device=self.device).unsqueeze(1)




        with torch.no_grad():
            if self.enable_dqn_double:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1, keepdim=True)[0]



        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        # every N steps or batches
        logger.debug(f"  [Optimize] step={step_count:6d}, loss={loss.item():.4f}, ε={self.epsilon:.3f}")

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_grad_norm) # TODO: Do I really need this?
        self.optimizer.step()       # Update network parameters i.e. weights and biases


    def update_current_map(self, map: List[str]):

        self.current_map =  map
        self.current_map_state = torch.zeros(2*self.size**2, dtype=torch.float32, device=self.device)
        for i in range(self.size):
            for j in range(self.size):
                if not map[i][j] == 'H':
                    self.current_map_state[self.size**2 + i*self.size + j] = 1.0


    def get_state(self, obs: int, map: List[str]):
        """
        Convert the observation to a state tensor.
        """
        state = self.current_map_state.clone()
        state[obs] = 1.0
        return state


    def _get_distance_reward(self, obs: int):
        i = obs // self.size
        j = obs % self.size
        dist = (np.sqrt((i - self.size + 1)**2 + (j - self.size + 1)**2) / (np.sqrt(2) * (self.size - 1))).astype(np.float32)
        return dist

    def override_reward(self, previous_obs: int, obs: int, reward: float, terminated: bool) -> float:
        """
        @brief Overrides the environment reward to encourage progress toward the goal.

        This function modifies the given reward based on the agent's current and previous
        observations. It encourages forward progress and penalizes stagnation or revisiting
        past states.

        @param previous_obs The previous observation (state index) before taking the action.
        @param obs The current observation (state index) after taking the action.
        @param reward The original reward returned by the environment.
        @param terminated A boolean flag indicating whether the episode has ended.

        @return The adjusted reward based on movement and goal proximity:
                - Adds a large bonus if reaching the goal state (obs == 63).
                - Penalizes revisiting states or staying in place.
                - Encourages movement that reduces distance to the goal.
        """
        if obs in self.episode_history or terminated:
            return reward + 5.0*int(obs == 63)
        if previous_obs == obs:
            return reward - 1.0
        prev_dist = self._get_distance_reward(previous_obs)
        dist = self._get_distance_reward(obs)
        if dist > prev_dist:
            return reward
        return reward + 1.0 - dist

