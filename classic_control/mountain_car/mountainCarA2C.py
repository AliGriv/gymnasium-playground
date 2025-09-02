import gymnasium as gym
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Type, Dict, Any
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch import nn
import time
from common.loggerConfig import logger
from common.dnnAdvantageActorCritic import A2C, A2CHparams
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

class MountainCarA2CAgent:
    def __init__(self,
                 env: gym.Env,
                 save_path: Path,
                 existing_model_path: Optional[Path] = None,
                 *,
                 n_steps: int = 32,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 hidden_layers: Sequence[int] = (16, 16),
                 activation: Type[nn.Module] = nn.ReLU,
                 ent_coef: float = 1e-3,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 optimizer: str = "adam",
                 device: Optional[str] = None,
                 seed: Optional[int] = None,
                 normalize_advantages: bool = True,
                 max_steps_per_episode: int = 1000):
        r"""
        @brief Agent wrapper for training A2C on MountainCarContinuous-v0.

        This class manages the environment interaction, rollout buffer,
        and the A2C model (actor + critic).

        @param env                     Gym/Gymnasium environment (e.g., MountainCarContinuous-v0).
        @param save_path               Path to store model checkpoints and metadata.
        @param existing_model_path     Path to load an existing saved model (if provided).
        @param n_steps                 Number of steps per on-policy rollout before an update.
        @param gamma                   Discount factor.
        @param lam                     GAE lambda for advantage estimation.
        @param hidden_layers           List of hidden layer sizes for actor and critic.
        @param activation              Activation function class for hidden layers.
        @param ent_coef                Entropy bonus coefficient.
        @param vf_coef                 Value-function loss coefficient.
        @param max_grad_norm           Maximum global gradient norm for clipping.
        @param actor_lr                Learning rate for actor optimizer.
        @param critic_lr               Learning rate for critic optimizer.
        @param optimizer               Optimizer type ("adam", "sgd", "rmsprop", etc.).
        @param device                  Device string ("cpu" or "cuda").
        @param seed                    Random seed for reproducibility.
        @param normalize_advantages    Whether to normalize advantages before updates.
        @param max_steps_per_episode   Maximum allowed steps per training episode.
        """
        # --- Environment setup ---
        self.env = env
        self.save_path = Path(save_path)
        self.max_steps_per_episode = int(max_steps_per_episode)

        if seed is not None:
            try:
                self.env.reset(seed=seed)
            except TypeError:
                self.env.seed(seed)
            torch.manual_seed(seed)

        # --- Device selection ---
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Environment dimensions & action limits ---
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        assert len(obs_space.shape) == 1, "Expected 1D observation space."
        assert len(act_space.shape) == 1, "Expected 1D action space."

        self.state_dim = int(obs_space.shape[0])
        self.action_dim = int(act_space.shape[0])
        self.action_limit = float(act_space.high[0])

        # --- A2C model setup ---
        hparams = A2CHparams(
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )

        if existing_model_path:
            # Load from saved checkpoint and metadata
            self.a2c = A2C.load_model(existing_model_path, device=self.device)
        else:
            # Create new A2C model instance
            self.a2c = A2C(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                action_limit=self.action_limit,
                hidden_dims=hidden_layers,
                activation=activation,
                hparams=hparams,
                device=self.device,
            )

        self.optimizer_type = optimizer.lower()

        self.actor_optimizer = self.get_optimizer(self.optimizer_type, self.a2c.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = self.get_optimizer(self.optimizer_type, self.a2c.critic.parameters(), lr=critic_lr)

        # --- Rollout configuration ---
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.normalize_advantages = bool(normalize_advantages)

        # --- Rollout buffers ---
        self.states     = torch.zeros(self.n_steps, self.state_dim, dtype=torch.float32, device=self.device)
        self.actions    = torch.zeros(self.n_steps, self.action_dim, dtype=torch.float32, device=self.device)
        self.rewards    = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)
        self.dones      = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)
        self.values     = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)
        self.logps      = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)
        self.returns    = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(self.n_steps, 1, dtype=torch.float32, device=self.device)

        # Buffer index
        self.step_idx = 0

        # --- Episode tracking ---
        self.ep_return = 0.0
        self.ep_length = 0
        self.episode_count = 0

        # --- Initial reset ---
        state, info = self.env.reset()
        self.last_obs = torch.tensor(state, dtype=torch.float32, device=self.device)


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
        elif optimizer == "adamw":
            return  optim.AdamW(model_parameters, lr=lr)
        elif optimizer == "sgd":
            return optim.SGD(model_parameters, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Supported optimizers are 'adam' and 'sgd'.")

    def compute_values(self, states: Tensor) -> Tensor:
        r"""
        @brief Compute state-value estimates V(s) using the critic.

        @param states  Tensor [B, state_dim]
        @return        Tensor [B, 1] of V(s)
        """
        return self.a2c.critic(states.to(self.device))

    def compute_returns(self,
                        rewards: Tensor,
                        dones: Tensor,
                        last_value: Tensor,
                        gamma: Optional[float] = None) -> Tensor:
        r"""
        @brief Compute bootstrapped returns R_t for a single on-policy segment.

        Recurrence (backward over t = T-1..0):
            R_T = last_value
            R_t = r_t + γ * (1 - done_t) * R_{t+1}

        @param rewards     Tensor [T, 1]
        @param dones       Tensor [T, 1] with {0.0, 1.0}
        @param last_value  Tensor [1, 1] or [1]
        @param gamma       Discount factor (defaults to self.gamma)
        @return            Tensor [T, 1] of returns
        """
        gamma = self.gamma if gamma is None else float(gamma)
        T = rewards.size(0)
        returns = torch.zeros_like(rewards, device=self.device)

        next_return = last_value.squeeze(-1)  # shape [1]
        for t in reversed(range(T)):
            next_return = rewards[t] + gamma * (1.0 - dones[t]) * next_return
            returns[t] = next_return
        return returns

    def compute_advantages(self,
                           rewards: Tensor,
                           values: Tensor,
                           dones: Tensor,
                           last_value: Tensor,
                           gamma: Optional[float] = None,
                           lam: Optional[float] = None) -> Tensor:
        r"""
        @brief Compute GAE(λ) advantages for a single on-policy segment.

        Temporal-difference residual:
            δ_t = r_t + γ (1 - done_t) V(s_{t+1}) - V(s_t)

        Backward recursion:
            A_t = δ_t + γ λ (1 - done_t) A_{t+1}

        @param rewards     Tensor [T, 1]
        @param values      Tensor [T, 1] (V(s_t))
        @param dones       Tensor [T, 1] with {0.0, 1.0}
        @param last_value  Tensor [1, 1] (V(s_T) for bootstrap)
        @param gamma       Discount factor (defaults to self.gamma)
        @param lam         GAE lambda (defaults to self.lam)
        @return            Tensor [T, 1] advantages
        """
        gamma = self.gamma if gamma is None else float(gamma)
        lam = self.lam if lam is None else float(lam)

        T = rewards.size(0)
        advantages = torch.zeros_like(rewards, device=self.device)

        next_value = last_value
        next_adv = torch.zeros(1, device=self.device)
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * (1.0 - dones[t]) * next_value - values[t]
            next_adv = delta + gamma * lam * (1.0 - dones[t]) * next_adv
            advantages[t] = next_adv
            next_value = values[t]
        return advantages

    def compute_policy_terms(self,
                             states: Tensor,
                             actions: Tensor,
                             advantages: Tensor) -> Dict[str, Tensor]:
        r"""
        @brief Compute policy loss components for A2C using provided actions.

        Evaluates log π(a|s) under the current policy (no re-sampling) and
        uses the actor's entropy approximation for the entropy bonus.

        Notes:
        - Inputs are moved to the actor's device and cast to float.
        - Actions are slightly shrunk toward the interior of the support to
            avoid -inf log-prob at exact ±action_limit.
        @param states       Tensor [B, state_dim]
        @param actions      Tensor [B, action_dim]
        @param advantages   Tensor [B, 1]
        @return             Dict with 'policy_loss', 'entropy'
        """
        dev = self.a2c.actor.device
        states = states.to(dev).float()
        actions = actions.to(dev).float()
        advantages = advantages.to(dev).float()

        # Build the current policy distribution
        dist = self.a2c.actor.distribution(states)

        # Keep actions strictly inside (-limit, limit) to avoid -inf log-probs
        eps = 1e-6
        a_lim = float(self.a2c.actor.action_limit)
        safe_actions = torch.clamp(actions, -a_lim, a_lim) * (1.0 - eps)

        # Log π(a|s) (sum over action dims)
        logp = dist.log_prob(safe_actions).sum(dim=-1, keepdim=True)  # [B,1]

        # Entropy approximation from actor (returns per-sample entropy tensor)
        # This keeps gradients through log_std as desired.
        _, _, entropy_per_sample = self.a2c.actor(states)             # [B,1]

        # Policy gradient term (advantages treated as constants)
        policy_loss = -(logp * advantages.detach()).mean()
        entropy = entropy_per_sample.mean()

        return {"policy_loss": policy_loss, "entropy": entropy}


    def compute_value_loss(self, values: Tensor, returns: Tensor) -> Tensor:
        r"""
        @brief Compute the critic regression loss (0.5 * MSE).

        @param values   Tensor [B, 1] predicted V(s)
        @param returns  Tensor [B, 1] targets R
        @return         Scalar tensor loss
        """
        return 0.5 * F.mse_loss(values, returns)

    def compute_losses(self,
                       states: Tensor,
                       actions: Tensor,
                       returns: Tensor,
                       advantages: Tensor) -> Dict[str, Tensor]:
        r"""
        @brief Aggregate A2C losses: policy, value, entropy, and total.

        @param states       Tensor [B, state_dim]
        @param actions      Tensor [B, action_dim]
        @param returns      Tensor [B, 1]
        @param advantages   Tensor [B, 1]
        @return             Dict with 'policy_loss', 'value_loss', 'entropy', 'total_loss'
        """
        # Critic forward (with grads)
        values = self.a2c.critic(states.to(self.device))  # [B,1]

        policy_terms = self.compute_policy_terms(states, actions, advantages)
        policy_loss = policy_terms["policy_loss"]
        entropy = policy_terms["entropy"]

        value_loss = self.compute_value_loss(values, returns)

        # Coefficients: from agent if available, else from model hparams, else defaults
        ent_coef = getattr(self, "ent_coef", None)
        vf_coef = getattr(self, "vf_coef", None)
        if ent_coef is None or vf_coef is None:
            hp = getattr(self.a2c, "hparams", None)
            if hp is not None:
                ent_coef = hp.ent_coef
                vf_coef = hp.vf_coef
            else:
                ent_coef = 1e-3
                vf_coef = 0.5

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

    # ===================================================================== #
    #                               Optimize                                #
    # ===================================================================== #

    def optimize(
        self,
        mini_batch: Tuple[
            List[Tensor],  # states
            List[Tensor],  # actions
            List[Tensor],  # next_states
            List[float],   # rewards
            List[bool],    # dones
        ],
        *,
        normalize_advantages: Optional[bool] = None,
    ) -> Dict[str, float]:
        r"""
        @brief Perform one A2C optimization step on a contiguous on-policy segment.

        Expected mini_batch layout:
          - states      : List[Tensor] [B, state_dim] elements
          - actions     : List[Tensor] [B, action_dim] elements
          - next_states : List[Tensor] [B, state_dim] elements
          - rewards     : List[float]  length B
          - dones       : List[bool]   length B

        Steps:
          1) Collate lists into tensors on the configured device.
          2) Compute critic values and bootstrap value for the segment end.
          3) Compute bootstrapped returns and GAE(λ) advantages.
          4) Optionally normalize advantages.
          5) Compute losses locally and apply optimizer steps with gradient clipping.

        @param mini_batch             Tuple of lists as specified above.
        @param normalize_advantages   Override for advantage normalization; if None,
                                      the agent's flag is used.
        @return                       Dict[str, float] with scalar stats for logging.
        """


        norm_adv = self.normalize_advantages if normalize_advantages is None else bool(normalize_advantages)

        # ---- 1) Collate ----
        states = torch.stack([s.to(self.device).float() for s in mini_batch[0]], dim=0)       # [B, S]
        # Robust action collation (handles tensors or numeric types)
        if isinstance(mini_batch[1][0], torch.Tensor):
            actions = torch.stack([a.to(self.device).float() for a in mini_batch[1]], dim=0)  # [B, A]
        else:
            actions = torch.tensor(mini_batch[1], dtype=torch.float32, device=self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
        next_states = torch.stack([s.to(self.device).float() for s in mini_batch[2]], dim=0)  # [B, S]
        rewards = torch.tensor(mini_batch[3], dtype=torch.float32, device=self.device).unsqueeze(-1)  # [B,1]
        terminations = torch.tensor(mini_batch[4], dtype=torch.float32, device=self.device).unsqueeze(-1)    # [B,1]

        B = states.size(0)

        # ---- 2) Values & bootstrap ----
        values = self.compute_values(states)                                  # [B,1]
        with torch.no_grad():
            last_value = self.compute_values(next_states[-1:].contiguous())   # [1,1]

        # ---- 3) Returns & advantages ----
        with torch.no_grad():
            returns = self.compute_returns(rewards, terminations, last_value)        # [B,1]
            advantages = self.compute_advantages(rewards, values, terminations, last_value)  # [B,1]
            if norm_adv and B > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
                advantages = (advantages - adv_mean) / adv_std

        # ---- 4) Losses ----
        losses = self.compute_losses(states, actions, returns, advantages)
        total_loss = losses["total_loss"]

        # Resolve max grad norm
        max_grad_norm = getattr(self, "max_grad_norm", None)
        if max_grad_norm is None:
            hp = getattr(self.a2c, "hparams", None)
            max_grad_norm = hp.max_grad_norm if hp is not None else 0.5

        # ---- 5) Backprop & step ----
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.a2c.actor.parameters()) + list(self.a2c.critic.parameters()),
                max_norm=max_grad_norm
            )

        self.critic_optimizer.step()
        self.critic_optimizer.step()

        # Return Python floats
        return {
            "policy_loss": losses["policy_loss"].item(),
            "value_loss": losses["value_loss"].item(),
            "entropy": losses["entropy"].item(),
            "total_loss": total_loss.item(),
        }

    # ------------------------------------------------------------------ #
    #                           Save helper                               #
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, base_path: Optional[Path] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        r"""
        @brief Save model weights and metadata to disk.

        Uses the A2C wrapper's save API. If `base_path` is not provided,
        the agent's `save_path` from the constructor is used.

        @param base_path  Base path (no extension needed); .pt and .json will be created.
        @param extra      Optional metadata dict to be stored alongside the model.
        """
        base = Path(base_path) if base_path is not None else self.save_path
        # Prefer `save_model` if present; fallback to `save`.
        if hasattr(self.a2c, "save_model"):
            self.a2c.save_model(base, extra=extra)
        else:
            # Backward-compat: some versions expose `save` with the same behavior.
            self.a2c.save(base, extra=extra)

    # ------------------------------------------------------------------ #
    #                            Evaluation                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        r"""
        @brief Evaluate the current policy over multiple episodes using deterministic actions.

        @param num_episodes  Number of episodes to run.
        @return              Dictionary with average return and length statistics.
        """
        self.a2c.actor.eval()
        self.a2c.critic.eval()

        returns = []
        lengths = []

        for ep in range(int(num_episodes)):
            reset_out = self.env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            ep_ret = 0.0
            ep_len = 0

            for _ in range(self.max_steps_per_episode):
                action_t = self.a2c.act_deterministic(self._normalize_state(obs_t).unsqueeze(0)).squeeze(0)  # [action_dim]
                action = action_t.detach().cpu().numpy()

                step_out = self.env.step(action)
                next_obs, reward, terminated, truncated, _ = step_out
                done = terminated or ep_len >= self.max_steps_per_episode

                ep_ret += float(reward)
                ep_len += 1

                if done or ep_len >= self.max_steps_per_episode:
                    break

                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

            returns.append(ep_ret)
            lengths.append(ep_len)

        avg_ret = float(sum(returns) / max(1, len(returns)))
        avg_len = float(sum(lengths) / max(1, len(lengths)))

        logger.info(f"[Eval] Episodes={num_episodes} | AvgReturn={avg_ret:.3f} | AvgLen={avg_len:.1f}")
        return {"avg_return": avg_ret, "avg_length": avg_len}

    # ------------------------------------------------------------------ #
    #                              Training                               #
    # ------------------------------------------------------------------ #
    def train(self,
              max_episodes: int,
              *,
              eval_every: int = 25,
              save_every: int = 50,
              log_every: int = 1) -> None:
        r"""
        @brief Train the A2C agent for a maximum number of episodes.

        The loop:
          - runs episodes up to `max_episodes`;
          - collects on-policy transitions and performs updates every `n_steps`
            or at episode end via `optimize(...)`;
          - logs progress at the specified interval;
          - periodically evaluates and saves checkpoints; and
          - saves a final checkpoint at the end.

        @param max_episodes  Maximum number of training episodes.
        @param eval_every    Run evaluation every N episodes (0 disables periodic eval).
        @param save_every    Save a checkpoint every N episodes (0 disables periodic save).
        @param log_every     Log training stats every N episodes.
        """
        self.a2c.actor.train()
        self.a2c.critic.train()

        episode = 0
        global_step = 0
        start_time = time.time()
        best_reward = -float('inf')
        new_best_reward = False

        while episode < int(max_episodes):
            # Reset environment

            obs, _ = self.env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            ep_ret = 0.0
            ep_len = 0

            # Clear minibatch lists for this episode
            mb_states, mb_actions, mb_next_states, mb_rewards, mb_dones = [], [], [], [], []

            for t in range(self.max_steps_per_episode):
                # Stochastic action during training
                action_t = self.a2c.act(self._normalize_state(obs_t).unsqueeze(0)).squeeze(0)  # [action_dim]
                # logger.debug(f"\t \t Step {t} | Action: {action_t}")
                action = action_t.detach().cpu().numpy()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated)
                ep_ret += float(reward)
                ep_len += 1
                global_step += 1

                # Store transition
                mb_states.append(self._normalize_state(obs_t))
                mb_actions.append(action_t)
                mb_next_states.append(self._normalize_state(torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)))
                mb_rewards.append(float(reward))
                mb_dones.append(bool(done))

                # If we have enough steps or episode ended, optimize on this segment
                if len(mb_states) >= self.n_steps or done:
                    stats = self.optimize((mb_states, mb_actions, mb_next_states, mb_rewards, mb_dones))
                    # Reset the buffers for the next segment within the same episode
                    mb_states.clear()
                    mb_actions.clear()
                    mb_next_states.clear()
                    mb_rewards.clear()
                    mb_dones.clear()

                if done or ep_len >= self.max_steps_per_episode:
                    break

                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

            episode += 1

            if ep_ret > best_reward:
                logger.info(f"[Train] New best return {ep_ret:.3f} (old {best_reward:.3f}) at episode {episode}")
                best_reward = ep_ret
                new_best_reward = True
            # Logging
            if log_every and (episode % log_every == 0):
                elapsed = time.time() - start_time
                sps = global_step / max(1e-6, elapsed)
                logger.info(f"[Train] Ep={episode:05d} | Ret={ep_ret:.3f} | Len={ep_len:04d} | Steps/s={sps:.1f}")

            # Periodic evaluation
            if eval_every and (episode % eval_every == 0):
                self.evaluate(num_episodes=5)

            # Periodic save
            if new_best_reward or (save_every and (episode % save_every == 0)):
                extra = {"episode": episode, "global_step": global_step, "timestamp": time.time()}
                self._save_checkpoint(extra=extra)
                logger.info(f"[Save] Checkpoint saved at episode {episode}")
                new_best_reward = False

        # Final save at the end of training
        extra = {"episode": episode, "global_step": global_step, "timestamp": time.time(), "final": True}
        self._save_checkpoint(extra=extra)
        logger.info(f"[Save] Final model saved after {episode} episodes.")

    def _normalize_state(self, state: Tensor) -> Tensor:
        r"""
        @brief Normalize an observation state to the range [-1, 1].

        Maps raw environment observation values from their original range
        defined by the environment’s observation space `[low, high]` into
        a normalized range of [-1, 1].

        @param state
            Raw state observation as a torch.Tensor. Shape matches the environment’s
            observation space. Can be [state_dim] or [B, state_dim].

        @return
            Normalized state as a torch.Tensor with values scaled to [-1, 1].

        @details
            The normalization is performed in two steps:
            1. Shift and scale to map `[low, high]` → [0, 1].
            2. Linearly transform [0, 1] → [-1, 1].
        """
        low = torch.as_tensor(self.env.observation_space.low, dtype=torch.float32, device=state.device)
        high = torch.as_tensor(self.env.observation_space.high, dtype=torch.float32, device=state.device)

        norm01 = (state - low) / (high - low + 1e-8)  # avoid div by 0
        return norm01 * 2.0 - 1.0
