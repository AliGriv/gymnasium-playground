import torch
import json
from pathlib import Path
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Sequence, Type, Union
from common.loggerConfig import logger
from datetime import datetime
"""
Disclaimer: Heavily inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/core.py
"""

class Dnn(nn.Module):
    def __init__(self,
                 sizes: Sequence[int],
                 activation: Type[nn.Module],
                 output_activation: Type[nn.Module] = nn.Identity,
                 device: str = None):
        """
        @brief Initializes a Deep Neural Network (DNN) with specified layer sizes and activation functions

        @param sizes List of integers representing the sizes of each layer.
        @param activation Activation function for hidden layers.
        @param output_activation Activation function for the output layer.
        @param device Device to run the model on (e.g., 'cpu', 'cuda
        ').
        """
        super().__init__()

        self.sizes = sizes
        self.activation = activation
        self.output_activation = output_activation

        available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device if device else available_device

        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

        self.network = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.network(x)


class DnnActor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 activation: Type[nn.Module] = nn.ReLU,
                 action_limit: float = 1.0,
                 device: str = None):
        """
        @brief Initializes the actor network for policy approximation.

        @param state_dim Dimension of the input state.
        @param action_dim Dimension of the output action.
        @param hidden_dims List of hidden layer sizes.
        @param activation Activation function for hidden layers.
        @param action_limit Action limit/scale for the output.
        @param device Device to run the model on (e.g., 'cpu', 'cuda').
        """
        super().__init__()
        self.action_limit = action_limit

        # Policy Network
        self.pi = Dnn(
            sizes = [state_dim] + list(hidden_dims) + [action_dim],
            activation = activation,
            device = device
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Policy output but scaled to match action limits
        return self.action_limit * self.pi(state)


class DnnQFunction(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int],
                 activation: Type[nn.Module] = nn.ReLU,
                 device: str = None):
        """
        @brief Initializes the Q-function network for value approximation.

        @param state_dim Dimension of the input state.
        @param action_dim Dimension of the input action.
        @param hidden_dims List of hidden layer sizes.
        @param activation Activation function for hidden layers.
        @param device Device to run the model on (e.g., 'cpu', 'cuda').
        """
        super().__init__()

        # Q-function (Quality) Network which maps state, action to a Q-value
        self.q = Dnn(
            sizes = [state_dim + action_dim] + list(hidden_dims) + [1],
            activation = activation,
            output_activation = nn.Identity,
            device = device
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate state and action for Q-value computation
        x = torch.cat([state, action], dim=-1)
        return torch.squeeze(self.q(x), -1)

class DnnActorCritic(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_limit: float,
                 hidden_dims: Sequence[int],
                 activation: Type[nn.Module] = nn.ReLU,
                 device: str = None):
        super().__init__()

        """
        Two main accessors for the DnnActorCritic class:
        - self.pi: The policy network (actor) that outputs actions given states.
        - self.q: The quality network (critic) that estimates Q-values for state-action pairs
        based on the current policy.

        The update logic for both networks is typically handled in the training loop.
        """

        # Policy Network - Actor
        self.pi = DnnActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            action_limit=action_limit,
            device = device
        )
        # Quality Network - Critic
        self.q = DnnQFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            device = device
        )
        self.meta_data = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'action_limit': float(action_limit),
            'hidden_dims': hidden_dims,
            'activation': activation.__name__,
            'device': device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        }

    def act(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.pi(state).cpu().numpy()

    def save_model(self, path: Path, notes: str = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), path.with_suffix(".pt"))
        logger.info(f"Model weights saved to {path.with_suffix('.pt')}")

        self.meta_data['timestamp'] = datetime.now().isoformat()
        # Save metadata
        if notes:
            self.meta_data['notes'] = notes

        logger.info(f"Model metadata: {self.meta_data}")
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(self.meta_data, f, indent=4)
        logger.info(f"Model metadata saved to {path.with_suffix('.json')}")

    @classmethod
    def load_model(cls, path: Union[str, Path], device: str) -> "DnnActorCritic":
        """
        @brief Load a new DnnActorCritic instance from saved weights and metadata.

        @param path Path to the model weights (.pt) and metadata (.json).
        @param device Device to run the model on (e.g., 'cpu', 'cuda').
        @return A new DnnActorCritic instance with loaded weights and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path {path} does not exist.")

        # Load metadata
        with open(path.with_suffix(".json"), "r") as f:
            meta_data = json.load(f)

        # Create a new instance with the loaded metadata
        model = cls(
            state_dim=meta_data['state_dim'],
            action_dim=meta_data['action_dim'],
            action_limit=meta_data['action_limit'],
            hidden_dims=meta_data['hidden_dims'],
            activation=getattr(nn, meta_data['activation']),
            device=device
        )

        # Load the state dict
        model.load_state_dict(torch.load(path.with_suffix(".pt")))
        model.to(device)

        return model