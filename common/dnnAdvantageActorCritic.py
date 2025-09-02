import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Sequence, Type, Union, Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

from common.loggerConfig import logger
from common.dnnActorCritic import Dnn


class A2CActor(nn.Module):
    """
    Stochastic actor for continuous actions using a tanh-squashed Gaussian policy.
    - Mean (mu) is produced by Dnn
    - log_std is a learned, state-independent parameter (robust and simple for MountainCarContinuous).
    - Actions are in [-action_limit, action_limit] (e.g., 1.0 for MountainCarContinuous-v0).
    - forward() returns (action, log_prob, entropy_approx).
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Sequence[int] = (128, 128),
                 activation: Type[nn.Module] = nn.ReLU,
                 action_limit: float = 1.0,
                 log_std_init: float = 0.01,
                 log_std_bounds = (-5.0, 2.0),
                 device: str = None):
        super().__init__()
        self.action_limit = float(action_limit)
        self.log_std_bounds = log_std_bounds

        # Mean network
        self.mu_net = Dnn(
            sizes=[state_dim, *hidden_dims, action_dim],
            activation=activation,
            output_activation=nn.Identity,   # raw mean; tanh happens in the transform
            device=device
        )
        # State-independent log-std parameter (one per action dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        # Keep a cached device string consistent with Dnn
        self.device = self.mu_net.device

    def _make_dist(self, state: torch.Tensor) -> TransformedDistribution:
        state = state.to(self.device)
        mu = self.mu_net(state)  # [B, A]

        # Clamp log_std for numerical stability, then broadcast to batch
        log_std = torch.clamp(self.log_std, *self.log_std_bounds)
        std = torch.exp(log_std).expand_as(mu)  # [B, A]

        base = Normal(mu, std)  # unbounded
        # Tanh to (-1,1), then scale to [-action_limit, action_limit]
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=0.0, scale=self.action_limit)]
        return TransformedDistribution(base, transforms), mu, log_std.expand_as(mu)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """Sample action for environment interaction (no grads)."""
        dist, _, _ = self._make_dist(state)
        return dist.sample()

    @torch.no_grad()
    def act_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Greedy/eval action: tanh(mu) scaled to action_limit.
        (Useful for evaluation without exploration noise.)
        """
        state = state.to(self.device)
        mu = self.mu_net(state)
        return torch.tanh(mu) * self.action_limit

    def forward(self, state: torch.Tensor):
        """
        Returns:
            action:      reparameterized sample a ~ pi(.|state)       [B, A]
            log_prob:    log pi(a|state), summed over action dims     [B, 1]
            entropy_appx: entropy of base Normal (approximation)      [B, 1]
        """
        dist, mu, log_std = self._make_dist(state)

        # Reparameterized sample so grads flow through mu/log_std
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Approximate entropy with base Normal (simple & common)
        base_entropy = (0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=self.device)) + log_std).sum(-1, keepdim=True)

        return action, log_prob, base_entropy

    def distribution(self, state: torch.Tensor) -> TransformedDistribution:
        """Expose the action distribution if you need it."""
        dist, _, _ = self._make_dist(state)
        return dist

class A2CCritic(nn.Module):
    r"""
    @brief State-value function approximator V(s) for A2C.

    Uses `Dnn` to map a state vector to a single scalar value estimate.
    Designed to pair with the tanh-Gaussian `A2CActor`.

    Training tip:
      In vanilla A2C, the critic is trained to minimize 0.5 * MSE(V(s_t), R_t),
      where R_t is the n-step/bootstrap return (or GAE target), and the factor
      0.5 matches common practice.

    @param state_dim      Dimension of the environment state vector.
    @param hidden_dims    Hidden layer sizes for the value network (e.g., (128, 128)).
    @param activation     Activation class for hidden layers (e.g., nn.ReLU).
    @param device         Device string ('cpu' or 'cuda'); defaults to Dnn's choice.

    @note Initialization:
      `Dnn` already applies Xavier to hidden layers and a small uniform init
      to the final layer. That’s suitable for a critic too (keeps initial values
      near zero to avoid large early TD errors).
    """
    def __init__(self,
                 state_dim: int,
                 hidden_dims: Sequence[int] = (128, 128),
                 activation: Type[nn.Module] = nn.ReLU,
                 device: str = None):
        super().__init__()

        # Value head: outputs a single scalar per state
        self.v_net = Dnn(
            sizes=[state_dim, *hidden_dims, 1],
            activation=activation,
            output_activation=nn.Identity,
            device=device
        )
        self.device = self.v_net.device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        @brief Compute the state value V(s).

        @param state  Tensor of shape [B, state_dim] or [state_dim] (auto-batched).
        @return       Tensor of shape [B, 1]: estimated value(s) V(s).
        """
        # `Dnn` handles device transfer internally, but we keep this explicit for clarity.
        state = state.to(self.device)
        v = self.v_net(state)            # [B, 1]
        return v

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        @brief Value prediction without gradient tracking (inference/eval).

        @param state  Tensor [B, state_dim] or [state_dim].
        @return       Tensor [B, 1] with V(s).
        """
        return self.forward(state)



@dataclass
class A2CHparams:
    r"""
    @brief Hyperparameters for the A2C trainer wrapper.

    @param ent_coef       Entropy bonus coefficient (encourages exploration).
    @param vf_coef        Value-function loss coefficient.
    @param max_grad_norm  Global gradient clip (L2 norm).
    @param actor_lr       Learning rate for the actor.
    @param critic_lr      Learning rate for the critic.
    """
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


class A2C(nn.Module):
    """
    High-level A2C wrapper combining actor and critic with utilities:
    - Acting (stochastic / deterministic)
    - Training (update step)
    - Saving and loading with metadata
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_limit: float,
                 hidden_dims=(16, 16),
                 activation=nn.ReLU,
                 hparams: Optional[A2CHparams] = None,
                 device: Optional[str] = None):
        super().__init__()
        self.hparams = hparams or A2CHparams()

        self.actor = A2CActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            action_limit=action_limit,
            device=device,
        )
        self.critic = A2CCritic(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            device=device,
        )

        # cache dimensions for metadata saving
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.hidden_dims = list(hidden_dims)
        self.activation = activation.__name__

        # Device
        self.device = self.actor.device
        self.to(self.device)


    @torch.no_grad()
    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.act(state)

    @torch.no_grad()
    def act_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.act_deterministic(state)

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic.predict(state)

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        r"""
        @brief Save a checkpoint containing models, optimizers, and config.

        @param path   Filesystem path for torch.save (e.g., 'a2c_mccont.pt').
        @param extra  Optional dict for run metadata (step, seed, env_id, etc.).
        """
        path = Path(path)

        # Save weights
        torch.save(self.state_dict(), path.with_suffix(".pt"))
        logger.info(f"A2C model weights saved to {path.with_suffix('.pt')}")

        # Save metadata
        meta = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_limit": self.action_limit,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "hparams": asdict(self.hparams),
        }
        if extra:
            meta["extra"] = extra

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"A2C model metadata saved to {path.with_suffix('.json')}")

    @classmethod
    def load_model(cls, path: Union[str, Path], device: str = "cpu") -> "A2C":
        """
        Load a new A2C instance from saved weights (.pt) and metadata (.json).
        """
        path = Path(path)
        if not path.with_suffix(".pt").exists():
            raise FileNotFoundError(f"Checkpoint not found: {path.with_suffix('.pt')}")
        if not path.with_suffix(".json").exists():
            raise FileNotFoundError(f"Metadata not found: {path.with_suffix('.json')}")

        # Load metadata
        with open(path.with_suffix(".json"), "r") as f:
            meta = json.load(f)

        logger.info(f"Loading A2C model from {path} on device {device}")
        logger.info(f"Model metadata: {meta}")

        # Rebuild hparams
        hparams = A2CHparams(**meta.get("hparams", {}))

        # Create new instance
        model = cls(
            state_dim=meta["state_dim"],
            action_dim=meta["action_dim"],
            action_limit=meta["action_limit"],
            hidden_dims=meta["hidden_dims"],
            activation=getattr(nn, meta["activation"]),
            hparams=hparams,
            device=device,
        )

        # Load weights
        state_dict = torch.load(path.with_suffix(".pt"), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        return model