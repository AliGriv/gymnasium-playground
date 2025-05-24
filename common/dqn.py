import torch
import json
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime
from common.loggerConfig import logger

"""
Disclaimer: This code is heavily inspired by https://github.com/johnnycode8/dqn_pytorch/blob/main/dqn.py
Thanks JognnyCode8!
"""

class DQN(nn.Module):
    """
    @class DQN
    @brief Deep Q-Network with optional Dueling DQN architecture.

    Constructs a feedforward neural network with configurable hidden layers and
    supports both standard and dueling DQN structures.

    @param state_dim       Dimension of input state vector.
    @param action_dim      Dimension of action space.
    @param hidden_dims     List of hidden layer sizes. Default is [256].
    @param enable_dueling_dqn Whether to use dueling DQN architecture.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256],
                 enable_dueling_dqn: bool = True):
        super().__init__()
        self.enable_dueling_dqn=enable_dueling_dqn
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if not hidden_dims:
            raise RuntimeError("hidden_dims must be a list of integers or a single integer")
        self.hidden_dims = hidden_dims
        self.fc1 = nn.Linear(state_dim, self.hidden_dims[0])

        if self.enable_dueling_dqn:
            # Value stream
            self.value_layers = self._get_hidden_layers(self.hidden_dims)
            self.value = nn.Linear(self.hidden_dims[-1], 1)

            # Advantages stream
            self.adv_layers = self._get_hidden_layers(self.hidden_dims)
            self.advantages = nn.Linear(self.hidden_dims[-1], action_dim)

        else:
            self.hidden_layers = self._get_hidden_layers(self.hidden_dims)
            self.output = nn.Linear(self.hidden_dims[-1], action_dim)

        # Apply He init to **every** Linear layer in this module
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # Kaiming uniform is one common "He" init; adjust nonlinearity if you change activation
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

    def _get_hidden_layers(self, hidden_dims: List[int]) -> List[nn.Linear]:
        """
        @brief Creates a list of fully connected layers for the given hidden dimensions.

        @param hidden_dims List of layer sizes.
        @return List of nn.Linear layers.
        """
        if not hidden_dims:
            return []
        return [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)]

    def forward(self, x):
        """
        @brief Performs a forward pass through the network.

        If dueling DQN is enabled, splits into value and advantage streams and recombines
        into Q-values. Otherwise, standard feedforward output.

        @param x Input tensor of shape (batch_size, state_dim)
        @return Q-values tensor of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = x
            for value_layer in self.value_layers:
                v = F.relu(value_layer(v))
            V = self.value(v)

            # Advantages calc
            a = x
            for adv_layer in self.adv_layers:
                a = F.relu(adv_layer(a))
            A = self.advantages(a)

            # Calc Q
            if A.dim() == 1:
                A = A.unsqueeze(0)
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            Q = self.output(x)
        return Q

    def __repr__(self) -> str:
        """
        @brief Returns a formatted string summarizing the network configuration.
        """
        info = f"DQN(\n"
        info += f"  input_dim={self.fc1},\n"
        info += f"  hidden_dims={self.hidden_dims},\n"
        info += f"  dueling_dqn={self.enable_dueling_dqn},\n"

        if self.enable_dueling_dqn:
            info += f"  Value stream: {self.value_layers}\n"
            info += f"  Value Hidden layers: {self.value_layers}\n"
            info += f"  Advantage stream: {self.adv_layers}\n"
            info += f"  Advantage Hidden layers: {self.adv_layers}\n"
        else:
            info += f"  Hidden layers: {self.hidden_layers}\n"
            info += f"  output={self.output},\n"
        info += ")"
        return info

    def save_model(self, path: Union[str, Path], notes: Optional[str] = None):
        """
        @brief Save model weights and architecture metadata.

        Saves both the model weights (.pt file) and a human-readable JSON metadata file
        that contains the model architecture and other optional notes.

        @param path Destination path (without extension) to save the files.
        @param notes Optional comments or notes to include in the metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), path.with_suffix(".pt"))
        logger.info(f"Model weights saved to {path.with_suffix('.pt')}")

        # Save metadata
        metadata = {
            "state_dim": self.fc1.in_features,
            "action_dim": self.output.out_features if not self.enable_dueling_dqn else self.advantages.out_features,
            "hidden_dims": self.hidden_dims,
            "enable_dueling_dqn": self.enable_dueling_dqn,
            "timestamp": datetime.now().isoformat()
        }
        if notes:
            metadata["notes"] = notes

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {path.with_suffix('.json')}")

    @classmethod
    def load_model(cls, path: Union[str, Path], device: str) -> "DQN":
        """
        @brief Load a new DQN instance from saved weights and metadata.

        Loads model architecture from a JSON metadata file and reconstructs the model.
        Then loads the saved weights from a corresponding .pt file. If the metadata file
        is missing or malformed, an error is raised. This method creates a new instance
        of the DQN model using the loaded architecture.

        @param path Base path (without extension) from which to load the model.
                    Expected files are <path>.json for metadata and <path>.pt for weights.
        @param device Device to load the model onto (e.g., 'cpu' or 'cuda').

        @return DQN instance with architecture and weights loaded.
        """
        path = Path(path)
        weights_path = path.with_suffix(".pt")
        metadata_path = path.with_suffix(".json")
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Optional logging or verification of compatibility
                logger.info(f"Loaded architecture metadata: {metadata}")
            except:
                logger.exception(f"Failed to load the metadata from {metadata_path}.")
        else:
            raise RuntimeError(f"Metadata file not found at {metadata_path}. "
                          f"Using current architecture to load weights.")

        if not metadata:
            raise RuntimeError(f"No metadata/architecture was defined for the model.")
        else:
            model = cls(**{
                "state_dim": metadata["state_dim"],
                "action_dim": metadata["action_dim"],
                "hidden_dims": metadata["hidden_dims"],
                "enable_dueling_dqn": metadata["enable_dueling_dqn"]
            })
        logger.debug(f"Loading the model weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))
        model.to(device)
        model.eval()
        # for name, param in model.named_parameters():
        #     logger.debug(f"{name}: {param.data}")
        return model

if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    hidden_dims = [20, 5, 5]
    net = DQN(state_dim, action_dim, hidden_dims=hidden_dims, enable_dueling_dqn=True)
    state = torch.randn(10, state_dim)


    print(f"DQN Model: {net}")
    print(f"Input: {state}")
    output = net(state)
    print(f"Input Shape: {state.shape}, Output Shape: {output.shape}")
    print(f"Output: {output}")

