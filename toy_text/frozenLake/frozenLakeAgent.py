from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.optim as optim
import gymnasium as gym
import itertools
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt

from common.dqn import DQN
from common.replayMemory import ReplayMemory   
from toy_text.utils import get_moving_avgs

# For printing date and time
DATE_FORMAT = "%y-%m-%d %H:%M:%S"

class FrozenLakeAgent:

    def __init__(self,
                 maps: Dict[str, List[str]],
                 learning_rate: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 network_sync_rate: int = 100,
                 replay_memory_size: int = 10000,
                 batch_size: int = 32,
                 enable_dqn_dueling: bool = False,
                 enable_dqn_double: bool = False,
                 hidden_layer_dims: List[int] = [128,128],
                 stop_on_reward: Optional[int] = 10000,
                 discount_factor: float = 0.95,
                 existing_dqn_path: Optional[Path] = None,
                 optimizer: str = "adam",
                 log_directory: str = "logs/frozenLake",
                 is_slippery: bool = False,
                 save_dqn_path: Optional[Path] = Path("models/toy_text/frozenLakeDaqn.pt")):

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

        # TODO: Add support for CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Dimensions
        state_dim = 2*self.size**2
        action_dim = 4

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


        self.save_path = save_dqn_path

        if not isinstance(save_dqn_path, Path):
            self.save_path =  Path(save_dqn_path)
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer, self.q_net.parameters(), lr=self.lr)

        # Loss Function
        self.loss_fn = torch.nn.MSELoss()     

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)

        # Load existing model if provided
        if isinstance(existing_dqn_path, str):
            existing_dqn_path = Path(existing_dqn_path)
        if existing_dqn_path and existing_dqn_path.exists():
            self.q_net.load_state_dict(torch.load(existing_dqn_path))
            print(f"Loaded existing DQN from {existing_dqn_path}")
        
        self.log_file = Path(log_directory) / 'frozenLake.log'
        self.graph_file = Path(log_directory) / 'frozenLake.png'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def get_optimizer(self, optimizer: str, model_parameters, lr: float):
        optimizer = optimizer.lower()
        if optimizer == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Supported optimizers are 'adam' and 'sgd'.")

    def get_action(self, state: torch.tensor) -> torch.tensor:

        if np.random.random() < self.epsilon:
            action =  self.env.action_space.sample()
            return action
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return torch.argmax(q_values).item()

    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def run(self, num_episodes: Optional[int]=None, is_training: bool = True, render: bool = False):

        if num_episodes:
            self.epsilon_decay = (self.epsilon - self.final_epsilon) / (num_episodes / 2)
        else:
            self.epsilon_decay = (self.epsilon - self.final_epsilon) / 1000 # TODO: Avoid magic number

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.log_file, 'w') as file:
                file.write(log_message + '\n')


        desc = list(self.maps.values())[0] # TODO
        self.env = gym.make('FrozenLake-v1', 
                       desc=desc,
                       map_name="8x8" if self.size == 8 else "4x4",
                       is_slippery=self.is_slippery,
                       render_mode='human' if render else None)


        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        if is_training:
            
        
            self.target_net.load_state_dict(self.q_net.state_dict())
            

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            if self.save_path.exists():
                self.q_net.load_state_dict(torch.load(self.save_path))
                print(f"Loaded existing DQN from {self.save_path}")
            else:
                self.q_net.load_state_dict(torch.load(self.existing_dqn_path))
                print(f"Loaded existing DQN from {self.existing_dqn_path}")
            # switch model to evaluation mode
            self.q_net.eval()

        # if num_upisodes is None, train INDEFINITELY, 
        # manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():
            if num_episodes and episode >= num_episodes:
                break
            obs, info = self.env.reset()  
            state = self.get_state(obs, desc)

            terminated = False      
            episode_reward = 0.0    
            # Perform actions until episode terminates or reaches max rewards
            while(not terminated and episode_reward < self.stop_on_reward):

                action = self.get_action(state)
                if torch.is_tensor(action):
                    action = action.item()
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                next_state = self.get_state(new_obs, desc)
                action_state = self.get_action_state(action)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if is_training:
                    # Append to memory
                    self.memory.push(state, action_state, next_state, reward, terminated)
                    
                    step_count+=1

                state = next_state

            rewards_per_episode.append(episode_reward)
            print(f"episode: {episode}, episode_reward: {episode_reward}")

            # Save model when new best reward is obtained.
            if is_training:
                if best_reward and episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.log_file, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(self.q_net.state_dict(), self.save_path)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(self.memory) > self.batch_size:
                    mini_batch = self.memory.sample(self.batch_size)
                    self.optimize(mini_batch, self.q_net, self.target_net) # TODO
                    self.decay_epsilon()
                    epsilon_history.append(self.epsilon)
                    if step_count > self.network_sync_rate:
                        self.target_net.load_state_dict(self.q_net.state_dict())
                        step_count=0

    def save_graph(self, rewards_per_episode, epsilon_history):
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



    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = mini_batch

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
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
                # Calculate target Q values (expected returns)
                # print(f"rewards: {rewards.shape}")
                # print(f"terminations: {terminations.shape}")
                # print(f"new_states: {new_states.shape}")
                # print(f"target_dqn(new_states): {target_dqn(new_states).shape}")
                # print(f"target_dqn(new_states).max(dim=1): {target_dqn(new_states).max(dim=1).shape}")
                # print(f"target_dqn(new_states): {target_dqn(new_states).shape}")
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1, keepdim=True)[0]


        # Calcuate Q values from current policy
        # print(f"actions: {actions.shape}")
        # print(actions)
        # print(f"policy_dqn(states): {policy_dqn(states).shape}")
        # print(f"target_q: {target_q.shape}")
        current_q = policy_dqn(states).gather(dim=1, index=actions)
        # print(f"current_q: {current_q.shape}")

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases


    def get_state(self, obs: int, map: List[str]):
        """
        Convert the observation to a state tensor.
        """
        state = np.zeros(2*self.size**2, dtype=np.float32)
        state[obs] = 1.0
        for i in range(self.size):
            for j in range(self.size):
                if not map[i][j] == 'H':
                    state[self.size**2 + i*self.size + j] = 1.0
        return torch.from_numpy(state).to(self.device)
    
    def get_action_state(self, action: int):

        action_state = np.zeros(4, dtype=np.int64)
        action_state[action] = 1
        return action_state