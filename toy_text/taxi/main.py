from tqdm import tqdm
from toy_text.taxi.taxiAgent import *
from matplotlib import pyplot as plt
from pathlib import Path
import toy_text.utils as utils




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
    Run the Taxi experiment.

    Args:
        train (bool): Whether to train the agent.
        test (bool): Whether to test the agent.
        episodes (int): Number of episodes to run.
        render (bool): Whether to render the environment.
        learning_rate (float): Learning rate for the agent.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Rate at which epsilon decays.
        model_save_path (str): Path to save the trained model.
        model_load_path (str, optional): Path to load the model for testing (or additional training).
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
        agent.epsilon = 0.0
        if train:
            num_episodes = 1
        else:
            num_episodes = episodes
        test_stats = run_episodes(num_episodes=num_episodes, is_train=False)
        r = test_stats['rewards']
        te = test_stats['terminated']
        tr = test_stats['truncated']
        for episode in range(episodes):
            print(f"Episode #{episode}: Reward {r[episode]}, Terminated {te[episode]}, Truncated {tr[episode]}")

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

