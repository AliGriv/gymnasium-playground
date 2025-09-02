import click
from common.loggerConfig import logger
from classic_control.mountain_car.main import run as run_mountain_car
from classic_control.mountain_car.main import run_ddpg as run_mountain_car_ddpg
from classic_control.mountain_car.main import run_a2c as run_mountain_car_a2c

@click.group()
def cli():
    """Run experiments from Classic Control projects."""
    pass


@cli.command()
@click.option('--train', is_flag=True, help='Run training mode')
@click.option('--test', is_flag=True, help='Run test mode')
@click.option('--model-save-path', default='models/classic_control/mountain_car.pkl', help='Where to save the pickel model (used in training)')
@click.option('--model-load-path', help='Path to pickel load model from, in test-only mode it is required')
@click.option('--render', is_flag=True, help='Render the environment')
@click.option('--learning-rate', default=0.999, type=float, help='Learning rate')
@click.option('--epsilon', type=float, default=1.0, help='Starting epsilon (will be 0 in test mode)')
@click.option('--epsilon-decay', type=float, help='Epsilon decay rate (default: epsilon / (episodes * 0.8))')
@click.option('--episodes', type=int, required=True, help='Number of episodes to run')
@click.option('--plot', is_flag=True, help='Plot some statistics from training procedure')
@click.option('--cont-actions', is_flag=True, help='Use continuous actions model (defaultuses discrete actions space).')
def mountain_car(train, test, model_save_path, model_load_path, render, learning_rate, epsilon, epsilon_decay, episodes, plot, cont_actions):
    """Run the Taxi experiment"""

    if not train and not test:
        raise click.UsageError("You must specify either --train or --test (or both).")

    if test and not train:
        if not model_load_path:
            raise click.UsageError("--model-load-path is required when using --test.")
        epsilon = 0.0

    if epsilon_decay is None:
        epsilon_decay = epsilon / (episodes * 0.8) # TODO: Avoid magic numbers

    if plot and not train:
        plot = False

    if not train and model_save_path:
        model_save_path = None
    elif train and not model_save_path.endswith('.pkl'):
        model_save_path += '.pkl'


    logger.info("Running Taxi experiment with:")
    logger.info(f"  Mode: {'train' if train else ''} {'test' if test else ''}")
    logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Epsilon: {epsilon}")
    logger.info(f"  Epsilon Decay: {epsilon_decay}")
    logger.info(f"  Model Save Path: {model_save_path}")
    logger.info(f"  Model Load Path: {model_load_path}")
    logger.info(f"  Render: {render}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Plots: {plot}")
    logger.info(f"  Continuous Actions: {cont_actions}")

    # Call your training or testing function here
    run_mountain_car(
        train=train,
        test=test,
        episodes=episodes,
        render=render,
        learning_rate=learning_rate,
        start_epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        plot=plot,
        cont_actions=cont_actions
    )

@cli.command()
@click.option('--train', is_flag=True, help='Run training mode')
@click.option('--test', is_flag=True, help='Run test mode')
@click.option('--model-save-path', default='models/classic_control/mountain_car_ddpg.pt', help='Where to save the model (used in training)')
@click.option('--model-load-path', help='Path to load model from, in test-only mode it is required')
@click.option('--render', is_flag=True, help='Render the environment')
@click.option('--policy-learning-rate', default=0.001, type=float, help='Policy earning rate')
@click.option('--quality-learning-rate', default=0.005, type=float, help='Policy earning rate')
@click.option('--episodes', type=int, required=True, help='Number of episodes to run')
@click.option('--plot', is_flag=True, help='Plot some statistics from training procedure')
@click.option('--hidden-layers', multiple=True, type=int, default=(16,16),
              help="List of integers for number of nodes in each hidden layer.")
@click.option('--max-episode-steps', type=int, default=999,
              help='Maximum number of steps per episode (default: 500).')
def mountain_car_ddpg(train,
                      test,
                      model_save_path,
                      model_load_path,
                      render,
                      policy_learning_rate,
                      quality_learning_rate,
                      episodes,
                      plot,
                      hidden_layers,
                      max_episode_steps):

    if not train and not test:
        raise click.UsageError("Specify either --train or --train to proceed.")

    if test and not train:
        if not model_load_path:
            raise click.UsageError("--model-load-path is required when using --test.")


    if plot and not train:
        plot = False

    if not train and model_save_path:
        model_save_path = None
    elif train and not model_save_path.endswith('.pt'):
        model_save_path += '.pt'

    logger.info("Running Taxi experiment with:")
    logger.info(f"  Mode: {'train' if train else ''} {'test' if test else ''}")
    logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Model Save Path: {model_save_path}")
    logger.info(f"  Model Load Path: {model_load_path}")
    logger.info(f"  Render: {render}")
    logger.info(f"  Policy Learning Rate: {policy_learning_rate}")
    logger.info(f"  Quality Learning Rate: {quality_learning_rate}")
    logger.info(f"  Plots: {plot}")
    logger.info(f"  Hidden Layers: {hidden_layers}")
    logger.info(f"  Max Episode Steps: {max_episode_steps}")

    run_mountain_car_ddpg(
        train=train,
        test=test,
        episodes=episodes,
        render=render,
        policy_learning_rate=policy_learning_rate,
        quality_learning_rate=quality_learning_rate,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        plot=plot,
        hidden_layer_dims=list(hidden_layers),
        max_episode_steps=max_episode_steps
    )

@cli.command()
@click.option('--train', is_flag=True, help='Run training mode')
@click.option('--test', is_flag=True, help='Run test mode')
@click.option('--model-save-path', default='models/classic_control/mountain_car_a2c.pt', help='Where to save the model (base path, no extension)')
@click.option('--model-load-path', help='Path to load model from, required in test-only mode')
@click.option('--render', is_flag=True, help='Render the environment')
@click.option('--episodes', type=int, required=True, help='Maximum number of episodes to run (training)')
@click.option('--hidden-layers', multiple=True, type=int, default=(16, 16),
              help="List of integers for number of nodes in each hidden layer.")
@click.option('--max-episode-steps', type=int, default=500,
              help='Maximum number of steps per episode (default: 1000).')
@click.option('--n-steps', type=int, default=32, help='Number of rollout steps before each update')
@click.option('--gamma', type=float, default=0.99, help='Discount factor')
@click.option('--lam', type=float, default=0.95, help='GAE lambda for advantage estimation')
@click.option('--ent-coef', type=float, default=1e-3, help='Entropy bonus coefficient')
@click.option('--vf-coef', type=float, default=0.5, help='Value function loss coefficient')
@click.option('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm (clipping)')
@click.option('--actor-lr', type=float, default=3e-4, help='Learning rate for actor')
@click.option('--critic-lr', type=float, default=3e-4, help='Learning rate for critic')
@click.option('--optimizer', type=click.Choice(['adam', 'adamw', 'sgd', 'rmsprop']), default='adam', help='Optimizer type')
@click.option('--device', type=str, default=None, help='Device to run on (cpu or cuda)')
@click.option('--seed', type=int, default=None, help='Random seed')
@click.option('--normalize-advantages/--no-normalize-advantages', default=True, help='Enable/disable advantage normalization')
@click.option('--eval-every', type=int, default=25, help='Evaluate every N episodes during training')
@click.option('--save-every', type=int, default=50, help='Save a checkpoint every N episodes during training')
@click.option('--log-every', type=int, default=1, help='Log training stats every N episodes')
@click.option('--num-eval-episodes', type=int, default=5, help='Number of episodes per evaluation pass')
def mountain_car_a2c(train,
                     test,
                     model_save_path,
                     model_load_path,
                     render,
                     episodes,
                     hidden_layers,
                     max_episode_steps,
                     n_steps,
                     gamma,
                     lam,
                     ent_coef,
                     vf_coef,
                     max_grad_norm,
                     actor_lr,
                     critic_lr,
                     optimizer,
                     device,
                     seed,
                     normalize_advantages,
                     eval_every,
                     save_every,
                     log_every,
                     num_eval_episodes):
    """
    Run Advantage Actor-Critic (A2C) on MountainCarContinuous-v0.
    """

    if not train and not test:
        raise click.UsageError("Specify either --train or --train to proceed.")

    if test and not train:
        if not model_load_path:
            raise click.UsageError("--model-load-path is required when using --test.")


    if not train and model_save_path:
        model_save_path = None
    elif train and not model_save_path.endswith('.pt'):
        model_save_path += '.pt'

    run_mountain_car_a2c(
        train=train,
        test=test,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        max_episode_steps=max_episode_steps,
        hidden_layers=hidden_layers,
        render=render,
        n_steps=n_steps,
        gamma=gamma,
        lam=lam,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        optimizer=optimizer,
        device=device,
        seed=seed,
        normalize_advantages=normalize_advantages,
        max_episodes=episodes,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        num_eval_episodes=num_eval_episodes,
    )
