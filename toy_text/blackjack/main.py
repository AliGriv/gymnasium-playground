from tqdm import tqdm
from toy_text.blackjack.blackjackAgent import BlackjackAgent
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import gymnasium as gym
import toy_text.utils as utils
from common.loggerConfig import logger

MINIMUM_EPSILON = 0.05


def run(
    train: bool,
    test: bool,
    episodes: int,
    render: bool,
    learning_rate: float,
    start_epsilon: float,
    epsilon_decay: float,
    epsilon_min: float = MINIMUM_EPSILON,
    model_save_path: str = None,
    model_load_path: str = None,
    plot: bool = True
):
    """
    @brief Runs the Blackjack training and/or testing experiment.

    Initializes the environment and agent, loads or saves the Q-table,
    runs training/testing episodes, and optionally plots the results.

    @param train Whether to train the agent.
    @param test Whether to test the agent.
    @param episodes Number of episodes to run.
    @param render Whether to render the environment during testing.
    @param learning_rate Learning rate (alpha) for Q-learning updates.
    @param start_epsilon Initial epsilon value for the epsilon-greedy strategy.
    @param epsilon_decay Amount by which epsilon decays after each episode.
    @param epsilon_min Minimum value for epsilon after decay.
    @param model_save_path Optional path to save the Q-table after training.
    @param model_load_path Optional path to load an existing Q-table before training/testing.
    @param plot Whether to plot training statistics (rewards, errors).
    """

    env = gym.make('Blackjack-v1', sab=False, render_mode='human' if render else None)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)


    if model_load_path:
        model_load_path = Path(model_load_path)
    existing_model = utils.load_existing_model(model_load_path)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=epsilon_min,
        existing_q=existing_model
    )

    def run_episodes(num_episodes: int, is_train: bool = True):
        rewards_per_episode = np.zeros(num_episodes)
        terminated_per_episode = []
        truncated_per_episode = []

        if not is_train:
            agent.epsilon = epsilon_min

        for episode in tqdm(range(num_episodes), desc="Episodes", leave=False):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if is_train:
                    agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs
                total_reward += reward

            if is_train:
                agent.decay_epsilon()

            rewards_per_episode[episode] = total_reward
            terminated_per_episode.append(terminated)
            truncated_per_episode.append(truncated)

        stats = {
            'rewards': rewards_per_episode,
            'terminated': terminated_per_episode,
            'truncated': truncated_per_episode
        }
        return stats

    # --- Training
    if train:
        train_stats = run_episodes(num_episodes=episodes, is_train=True)

    # --- Testing
    if test:
        num_test_episodes = 1 if train else episodes
        test_stats = run_episodes(num_episodes=num_test_episodes, is_train=False)

        r = test_stats['rewards']
        te = test_stats['terminated']
        tr = test_stats['truncated']

        for episode in range(num_test_episodes):
            logger.info(f"Episode #{episode}: Reward {r[episode]}, Terminated {te[episode]}, Truncated {tr[episode]}")

    env.close()

    # --- Model saving (optional)
    if model_save_path:
        model_save_path = Path(model_save_path).resolve()
        utils.save_trained_model(model_save_path, agent.q_values)

    # --- Plotting (only for training)
    if plot and train:
        rolling_length = 500

        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

        axs[0].set_title("Episode rewards")
        reward_moving_average = utils.get_moving_avgs(env.return_queue, rolling_length, "valid")
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

        axs[1].set_title("Episode lengths")
        length_moving_average = utils.get_moving_avgs(env.length_queue, rolling_length, "valid")
        axs[1].plot(range(len(length_moving_average)), length_moving_average)

        axs[2].set_title("Training Error")
        training_error_moving_average = utils.get_moving_avgs(agent.training_error, rolling_length, "same")
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

        plt.tight_layout()
        plt.show()
