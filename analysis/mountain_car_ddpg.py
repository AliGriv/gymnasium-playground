import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from common.loggerConfig import logger
from pathlib import Path
from common.dnnActorCritic import DnnActorCritic

def run_mountain_car_ddpg_analysis(model_path: Path):

    # -------------------------
    # Load your trained model
    # -------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")

    ac_net = DnnActorCritic.load_model(model_path, device=device)
    ac_net.eval()

    # -------------------------
    # Create the environment
    # -------------------------
    env = gym.make('MountainCarContinuous-v0')
    low, high = env.observation_space.low, env.observation_space.high

    def normalize_state(state: np.ndarray) -> np.ndarray:
        """Map raw state to [-1, 1] range."""
        norm01 = (state - low) / (high - low)
        return norm01 * 2.0 - 1.0

    # -------------------------
    # Sample a grid of states
    # -------------------------
    pos = np.linspace(low[0], high[0], 200)
    vel = np.linspace(low[1], high[1], 200)
    P, V = np.meshgrid(pos, vel)

    actions = np.zeros_like(P)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            state = np.array([P[i, j], V[i, j]], dtype=np.float32)
            state_norm = normalize_state(state)
            state_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(device)
            a = ac_net.act(state_t)[0, 0]
            actions[i, j] = a

    # -------------------------
    # Compute statistics
    # -------------------------
    a_min, a_max = actions.min(), actions.max()
    a_mean, a_std = actions.mean(), actions.std()
    min_idx = np.unravel_index(np.argmin(actions), actions.shape)
    max_idx = np.unravel_index(np.argmax(actions), actions.shape)
    min_state = (P[min_idx], V[min_idx])
    max_state = (P[max_idx], V[max_idx])

    logger.info("==== Actor Output Statistics ====")
    logger.info(f"Action min:  {a_min:.3f} at state {min_state}")
    logger.info(f"Action max:  {a_max:.3f} at state {max_state}")
    logger.info(f"Action mean: {a_mean:.3f}")
    logger.info(f"Action std:  {a_std:.3f}")

    # -------------------------
    # Generate 1D slices
    # -------------------------
    # Fix velocity = 0
    vel0_idx = np.abs(vel - 0).argmin()
    actions_vs_pos = actions[vel0_idx, :]

    # Fix position = 0 (roughly middle)
    pos0_idx = np.abs(pos - 0).argmin()
    actions_vs_vel = actions[:, pos0_idx]

    # -------------------------
    # Plotting
    # -------------------------
    plt.figure(figsize=(16, 10))

    # (a) Heatmap
    plt.subplot(2, 2, 1)
    heatmap = plt.pcolormesh(P, V, actions, shading='auto', cmap='coolwarm')
    plt.colorbar(heatmap, label="Action value")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Actor Output Heatmap")

    # (b) 1D slice: Action vs Position (vel=0)
    plt.subplot(2, 2, 2)
    plt.plot(pos, actions_vs_pos)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Position (vel=0)")
    plt.ylabel("Action")
    plt.title("Action vs Position (v=0)")

    # (c) 1D slice: Action vs Velocity (pos~0)
    plt.subplot(2, 2, 3)
    plt.plot(vel, actions_vs_vel)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Velocity (pos≈0)")
    plt.ylabel("Action")
    plt.title("Action vs Velocity (x≈0)")

    # (d) Histogram of actions
    plt.subplot(2, 2, 4)
    plt.hist(actions.flatten(), bins=50, color='gray', edgecolor='black')
    plt.xlabel("Action value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actor Outputs")

    plt.tight_layout()
    plt.show()