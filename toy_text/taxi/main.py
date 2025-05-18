from tqdm import tqdm
from toy_text.taxi.taxiAgent import *
from matplotlib import pyplot as plt
from pathlib import Path
import toy_text.utils as utils
from common.loggerConfig import logger



def run(
    train: bool,
    test: bool,
    episodes: int,
    render: bool,
    learning_rate: float,
    start_epsilon: float,
    epsilon_decay: float,
    model_save_path: str,
    model_load_path: str = None,
    plot: bool = True
):
    """
    @brief Entry point for training and testing a Q-learning agent on the Taxi-v3 environment.

    Initializes the environment and agent, handles model loading/saving, and executes training
    and/or testing episodes. Optionally renders the environment and plots results.

    @param train Flag to indicate if the agent should be trained.
    @param test Flag to indicate if the agent should be tested.
    @param episodes Number of episodes to run during training or testing.
    @param render Flag to render the environment (True for visualization).
    @param learning_rate Learning rate (alpha) for Q-learning updates.
    @param start_epsilon Initial exploration rate for the epsilon-greedy policy.
    @param epsilon_decay Rate at which epsilon is decayed after each episode.
    @param model_save_path Path to save the trained Q-table (as a model).
    @param model_load_path Optional path to load a pre-trained Q-table.
    @param plot Whether to plot training statistics (rewards and errors).
    """

    env = gym.make('Taxi-v3', render_mode='human' if render else None)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)

    if model_load_path:
        model_load_path = Path(model_load_path)
    existing_model = utils.load_existing_model(model_load_path)

    agent = TaxiAgent(env,
                      learning_rate,
                      start_epsilon,
                      epsilon_decay,
                      final_epsilon=0.0,
                      existing_q=existing_model)


    def run_episodes(num_episodes, is_train=True):
        rewards_per_episode = np.zeros(episodes)
        terminated_per_episode = []
        truncated_per_episode = []
        reward = 0
        if not is_train:
            agent.epsilon = 0.0
        for episode in tqdm(range(num_episodes), desc="Episodes", leave=False):
            obs, info = env.reset()
            done = False
            rewards = 0
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs
                rewards += reward

            if is_train:
                agent.decay_epsilon()
            if(agent.epsilon==0):
                agent.lr = 0.0001

            rewards_per_episode[episode] = rewards
            terminated_per_episode.append(terminated)
            truncated_per_episode.append(truncated)

        stats = {
            'rewards': rewards_per_episode,
            'terminated': terminated_per_episode,
            'truncated': truncated_per_episode
            }
        return stats




    if train:
        train_stats = run_episodes(num_episodes=episodes)

    if test:
        num_episodes = 1 if train else episodes
        test_stats = run_episodes(num_episodes=num_episodes, is_train=False)
        r = test_stats['rewards']
        te = test_stats['terminated']
        tr = test_stats['truncated']
        for episode in range(episodes):
            logger.info(f"Episode #{episode}: Reward {r[episode]}, Terminated {te[episode]}, Truncated {tr[episode]}")

    env.close()

    if model_save_path:
        model_save_path = Path(model_save_path).resolve()
        utils.save_trained_model(model_save_path, agent.q_values)

    if plot and train:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(train_stats['rewards'][max(0, t-100):(t+1)])
        plt.figure(1)
        plt.plot(sum_rewards)
        plt.title(f"Rewards Summation for every 100 Episodes")
        plt.grid()
        plt.savefig(model_save_path.parent / 'taxi_v3_5x5_rewards.png')


        plt.figure(2)
        rolling_length = 500
        training_error_moving_average = utils.get_moving_avgs(
            agent.training_error,
            rolling_length,
            "same"
        )
        plt.title(f"Training error for Taxi model")
        plt.plot(training_error_moving_average)
        plt.savefig('taxi_v3_5x5_training_error.png')

