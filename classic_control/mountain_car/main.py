from tqdm import tqdm
from classic_control.mountain_car.mountainCarAgent import *
from classic_control.mountain_car.mountainCarContAgent import *
from classic_control.mountain_car.mountainCarDdpg import *
from classic_control.mountain_car.mountainCarA2C import *
from matplotlib import pyplot as plt
from pathlib import Path
import common.utils as utils
from common.loggerConfig import logger
from typing import List, Sequence, Optional


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
    plot: bool = True,
    cont_actions: bool = False,
    maximum_episode_steps: int = 1000
    ):
    """
    @brief Entry point for training and testing a Q-learning agent on the MountainCar-v0 environment.

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
    @param cont_actions Flag to indicate if continuous actions should be used (default uses discrete actions space).
    @param maximum_episode_steps Maximum number of steps per episode (default: 1000).
    """



    model_meta_data = {}
    if model_load_path:
        model_load_path = Path(model_load_path)

    existing_model = utils.load_existing_model(model_load_path)
    if existing_model is not None:
        data = existing_model.tolist()
        import json
        with open("array.json", "w") as f:
            json.dump(data, f, indent=2)
    model_meta_data = utils.load_existing_model_metadata(model_load_path)
    if cont_actions and not model_meta_data.get('cont_actions', False) and model_load_path:
        logger.warning("The loaded model has been trained for discrete actions. Switching to discrete actions model.")
        cont_actions = False

    elif not cont_actions and model_meta_data.get('cont_actions', False) and model_load_path:
        logger.warning("The loaded model has been trained for continuous actions. Switching to continuous actions model.")
        cont_actions = True

    if not cont_actions:
        env = gym.make('MountainCar-v0', render_mode="human" if render else None)
        model_meta_data['cont_actions'] = False
        agent = MountainCarAgent(env,
                                 learning_rate,
                                 start_epsilon,
                                 epsilon_decay,
                                 final_epsilon=0.0,
                                 existing_q=existing_model,
                                 position_bins=model_meta_data.get('num_position_bins', 20),
                                 velocity_bins=model_meta_data.get('num_velocity_bins', 20))
        reward_stop_treshold = -1000
    else:
        env = gym.make('MountainCarContinuous-v0', render_mode="human" if render else None)
        model_meta_data['cont_actions'] = True
        agent = MountainCarContAgent(env,
                                     learning_rate,
                                     start_epsilon,
                                     epsilon_decay,
                                     final_epsilon=0.0,
                                     existing_q=existing_model,
                                     position_bins=model_meta_data.get('num_position_bins', 20),
                                     velocity_bins=model_meta_data.get('num_velocity_bins', 20),
                                     actions_bins=model_meta_data.get('num_actions_bins', 25))
        reward_stop_treshold = -5000

    model_meta_data.update(agent.get_meta_data())
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)



    def run_episodes(num_episodes, is_train=True):
        rewards_per_episode = np.zeros(episodes)
        terminated_per_episode = []
        truncated_per_episode = []

        reward = 0
        best_reward = -np.inf
        if not is_train:
            agent.epsilon = 0.0
        for episode in tqdm(range(num_episodes), desc="Episodes", leave=False):
            episode_history = []
            obs, info = env.reset()
            done = False
            rewards = 0
            step = 0
            terminated = False
            while not done:
                step += 1
                action = agent.get_action(obs)
                if not is_train and render:
                    logger.debug(f"Obs: {obs}, Action: {action}, Step: {step}")
                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_history.append((obs, action, reward, terminated, next_obs))
                # update the agent
                agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                obs = next_obs
                rewards += reward
                done = terminated or rewards <= reward_stop_treshold or step >= maximum_episode_steps

            if terminated and cont_actions:
                agent.rewind_episode(episode_history)

            if is_train:
                agent.decay_epsilon()
                if rewards > best_reward:
                    best_reward = rewards
                    logger.debug(f"New best reward: {best_reward} at episode {episode} with epsilon {agent.epsilon}. Terminated: {terminated}, Truncated: {truncated}")

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
        utils.save_trained_model_metadata(model_save_path, model_meta_data)

    if plot and train:
        try:
            # Get base name (without extension) or fallback
            if model_save_path:
                name = model_save_path.stem
                save_dir = model_save_path.parent
            else:
                name = "mountain_car_model"
                save_dir = Path.cwd()

            # --- Rewards plot ---
            sum_rewards = np.zeros(episodes)
            for t in range(episodes):
                window = train_stats["rewards"][max(0, t - 100) : (t + 1)]
                sum_rewards[t] = np.sum(window)

            plt.figure(1)
            plt.plot(sum_rewards)
            plt.title("Rewards Summation for every 100 Episodes")
            plt.grid(True)
            rewards_path = save_dir / f"{name}_rewards.png"
            plt.savefig(rewards_path)

            # --- Training error plot ---
            rolling_length = 500
            training_error_mavg = utils.get_moving_avgs(
                agent.training_error, rolling_length, "same"
            )

            plt.figure(2)
            plt.plot(training_error_mavg)
            plt.title("Training Error for Mountain Car Model")
            plt.grid(True)
            error_path = save_dir / f"{name}_training_error.png"
            plt.savefig(error_path)

        except Exception:
            logger.exception("Failed to generate the plots.")


def run_ddpg(
    train: bool,
    test: bool,
    episodes: int,
    render: bool,
    policy_learning_rate: float,
    quality_learning_rate: float,
    model_save_path: str,
    model_load_path: str = None,
    plot: bool = True,
    hidden_layer_dims: List = [12, 4],
    max_episode_steps: int = 999
    ):

    """
    @brief Run Deep Deterministic Policy Gradient (DDPG) training and/or evaluation
           on the MountainCarContinuous-v0 environment.

    This function sets up the Mountain Car environment, creates a DDPG agent, and either:
    - Trains the agent for a given number of episodes, saving checkpoints and plots, or
    - Evaluates a previously trained model for a given number of episodes.

    @param train
        If true, train the agent for `episodes` episodes.
    @param test
        If true, evaluate a trained agent for `episodes` episodes.
    @param episodes
        Number of episodes to run for training or evaluation.
    @param render
        Whether to render the environment during execution (slows training).
    @param policy_learning_rate
        Learning rate for the actor (policy) network.
    @param quality_learning_rate
        Learning rate for the critic (Q-value) network.
    @param model_save_path
        File path to save the trained model (used only if training).
    @param model_load_path
        Optional file path to load a pre-trained model (for training continuation or testing).
    @param plot
        If true, generate training progress plots (only relevant in training mode).
    @param hidden_layer_dims
        List defining the sizes of hidden layers for both actor and critic networks.
        Default: [12, 4].
    @param max_episode_steps
        Maximum number of steps allowed per episode. Default: 999.

    @return
        None. Side effects include training logs, saved models, evaluation logs,
        and (optionally) saved training progress plots.

    @throws FileNotFoundError
        If `test=True` and no model file exists at the provided `model_load_path`
        or `model_save_path`.
    """

    save_path = Path(model_save_path) if model_save_path else None
    load_path = Path(model_load_path) if model_load_path else None

    # We are targeting the continuous action space only
    env = gym.make('MountainCarContinuous-v0', render_mode="human" if render else None)

    if train:
        logger.info(f"=== Starting training for {episodes} episodes ===")
        agent = MountainCarDNNAgent(
            env=env,
            policy_learning_rate=policy_learning_rate,
            quality_learning_rate=quality_learning_rate,
            existing_model_path=load_path,
            save_path=save_path,
            hidden_layer_dims=hidden_layer_dims,
            max_episode_steps=max_episode_steps
        )
        agent.train(
            num_episodes=episodes
        )
        if plot:
            # the agent’s run method already calls save_graph periodically,
            # and leaves the final plot at agent.graph_file
            logger.info(f"Training complete. Graph saved to {agent.graph_file}")

    if test:
        load_for_test = save_path or load_path
        if load_for_test is None or not load_for_test.exists():
            raise FileNotFoundError(f"No model file found at {load_for_test!r} to load for test run")

        logger.info(f"==== Starting evaluation of Mountain Car DQN for {episodes} episodes ====")
        tester = MountainCarDNNAgent(
            env=env,
            policy_learning_rate=policy_learning_rate,
            quality_learning_rate=quality_learning_rate,
            existing_model_path=load_for_test,
            save_path=None,  # No need to save during testing
            hidden_layer_dims=hidden_layer_dims,
            max_episode_steps=max_episode_steps
        )
        tester.evaluate(
            num_episodes=episodes
        )
        logger.info("Testing complete.")

def run_a2c(
    train: bool,
    test: bool,
    model_save_path: str,
    model_load_path: Optional[str] = None,
    max_episode_steps: int = 1000,
    hidden_layers: Sequence[int] = (16, 16),
    render: bool = False,
    n_steps: int = 32,
    gamma: float = 0.99,
    lam: float = 0.95,
    ent_coef: float = 1e-3,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    optimizer: str = "adam",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    normalize_advantages: bool = True,
    max_episodes: int = 500,
    eval_every: int = 25,
    save_every: int = 50,
    log_every: int = 1,
    num_eval_episodes: int = 5,
):
    r"""
    @brief Run A2C on MountainCarContinuous-v0 in train and/or test mode.

    Creates the environment, constructs a `MountainCarA2CAgent`, and runs either
    training or evaluation based on the input flags. Periodically logs progress
    and saves checkpoints during training; always saves a final checkpoint.

    @param train                 If True, train a model.
    @param test                  If True, evaluate a model (deterministic policy).
    @param model_save_path       Base path (without extension) to save checkpoints (.pt/.json).
    @param model_load_path       Base path (without extension) to load a saved model (.pt/.json).
                                 Required for `test=True` unless a model is produced by the same run with `train=True`
                                 and a valid `model_save_path`.
    @param max_episode_steps     Maximum steps per episode (passed to agent).
    @param hidden_layers         Hidden layer sizes for both actor and critic.
    @param render                If True, render the environment.
    @param n_steps               On-policy rollout horizon per update.
    @param gamma                 Discount factor.
    @param lam                   GAE lambda for advantage estimation.
    @param ent_coef              Entropy bonus coefficient.
    @param vf_coef               Value loss coefficient.
    @param max_grad_norm         Global gradient clipping norm.
    @param actor_lr              Learning rate for actor optimizer.
    @param critic_lr             Learning rate for critic optimizer.
    @param optimizer             Optimizer name ("adam", "adamw", "sgd", "rmsprop").
    @param device                'cuda' or 'cpu'. Auto-selects if None.
    @param seed                  Optional RNG seed.
    @param normalize_advantages  Normalize per-batch advantages during updates.
    @param max_episodes          Maximum number of training episodes.
    @param eval_every            Evaluate every N episodes during training (0 disables).
    @param save_every            Save a checkpoint every N episodes during training (0 disables).
    @param log_every             Log training stats every N episodes.
    @param num_eval_episodes     Number of episodes for each evaluation pass.
    """

    logger.info(f"=== Running Mountain Car A2C with parameters: ===")
    logger.info(f"train={train} | test={test} | \n"
            f"model_save_path={model_save_path} | model_load_path={model_load_path} | \n"
            f"max_episode_steps={max_episode_steps} | hidden_layers={hidden_layers} | \n"
            f"n_steps={n_steps} | gamma={gamma} | lam={lam} | ent_coef={ent_coef} | \n"
            f"vf_coef={vf_coef} | max_grad_norm={max_grad_norm} | actor_lr={actor_lr} | \n"
            f"critic_lr={critic_lr} | optimizer={optimizer} | device={device} | seed={seed} | \n"
            f"normalize_advantages={normalize_advantages} | max_episodes={max_episodes} | \n"
            f"eval_every={eval_every} | save_every={save_every} | log_every={log_every} | \n"
            f"num_eval_episodes={num_eval_episodes} | render={render}")
    # ----- Validate flags/paths -----
    save_base = Path(model_save_path) if model_save_path else None
    load_base = Path(model_load_path) if model_load_path else None

    if not train and not test:
        raise ValueError("At least one of 'train' or 'test' must be True.")

    if test and not train:
        # pure test run requires an explicit load path
        if load_base is None:
            raise ValueError("`model_load_path` is required when test=True and train=False.")
        if not load_base.with_suffix(".pt").exists() or not load_base.with_suffix(".json").exists():
            raise FileNotFoundError(f"Model to test not found at base: {load_base}")

    if train and save_base is None:
        raise ValueError("`model_save_path` is required when train=True.")

    # ----- Env creation -----
    env = gym.make('MountainCarContinuous-v0', render_mode="human" if render else None)

    # Optional seeding (Gym/Gymnasium compatibility)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)

    # ----- Agent creation (load or fresh) -----
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if train:
        # Fresh or resume-from-checkpoint training
        if load_base is not None and load_base.with_suffix(".pt").exists() and load_base.with_suffix(".json").exists():
            logger.info(f"[run_a2c] Loading existing model for training from: {load_base}")
            agent = MountainCarA2CAgent(
                env=env,
                save_path=save_base,
                existing_model_path=load_base,
                n_steps=n_steps,
                gamma=gamma,
                lam=lam,
                hidden_layers=hidden_layers,
                activation=nn.ReLU,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                optimizer=optimizer,
                device=device,
                seed=seed,
                normalize_advantages=normalize_advantages,
                max_steps_per_episode=max_episode_steps,
            )
        else:
            logger.info("[run_a2c] Creating a new model for training.")
            agent = MountainCarA2CAgent(
                env=env,
                save_path=save_base,
                existing_model_path=None,
                n_steps=n_steps,
                gamma=gamma,
                lam=lam,
                hidden_layers=hidden_layers,
                activation=nn.ReLU,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                optimizer=optimizer,
                device=device,
                seed=seed,
                normalize_advantages=normalize_advantages,
                max_steps_per_episode=max_episode_steps,
            )

        # ----- Train -----
        logger.info(f"[run_a2c] Starting training for up to {max_episodes} episodes.")
        agent.train(
            max_episodes=max_episodes,
            eval_every=eval_every,
            save_every=save_every,
            log_every=log_every
        )

        # After training, if test is also requested, reuse the trained agent to evaluate.
        if test:
            logger.info("[run_a2c] Evaluating the newly trained model.")
            agent.evaluate(num_episodes=num_eval_episodes)
        else:
            logger.info("[run_a2c] Training finished.")

    else:
        # ----- Pure evaluation mode -----
        logger.info(f"[run_a2c] Loading model for evaluation from: {load_base}")
        agent = MountainCarA2CAgent(
            env=env,
            save_path=load_base,                 # not used for saving in pure eval, but required by constructor
            existing_model_path=load_base,
            n_steps=n_steps,
            gamma=gamma,
            lam=lam,
            hidden_layers=hidden_layers,
            activation=nn.ReLU,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            device=device,
            seed=seed,
            normalize_advantages=normalize_advantages,
            max_steps_per_episode=max_episode_steps,
        )
        agent.evaluate(num_episodes=num_eval_episodes, render=render)

    # Clean up env if needed
    try:
        env.close()
    except Exception:
        pass
