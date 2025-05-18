import click
from toy_text.taxi.main import run as run_taxi
from toy_text.blackjack.main import run as run_blackjack
from toy_text.frozenLake.frozenLakeMaps import FrozenLakeMaps
from toy_text.frozenLake.main import run as run_frozenlake
from common.loggerConfig import logger

@click.group()
def cli():
    """Run experiments from Toy Text projects."""
    pass

@cli.command()
@click.option('--train', is_flag=True, help='Run training mode')
@click.option('--test', is_flag=True, help='Run test mode')
@click.option('--model-save-path', default='models/toy_text/taxi.pkl', help='Where to save the pickel model (used in training)')
@click.option('--model-load-path', help='Path to pickel load model from, in test-only mode it is required')
@click.option('--render', is_flag=True, help='Render the environment')
@click.option('--learning-rate', default=0.9, type=float, help='Learning rate')
@click.option('--epsilon', type=float, default=1.0, help='Starting epsilon (will be 0 in test mode)')
@click.option('--epsilon-decay', type=float, help='Epsilon decay rate (default: epsilon / (episodes / 2))')
@click.option('--episodes', type=int, required=True, help='Number of episodes to run')
@click.option('--plot', is_flag=True, help='Plot some statistics from training procedure')
def taxi(train, test, model_save_path, model_load_path, render, learning_rate, epsilon, epsilon_decay, episodes, plot):
    """Run the Taxi experiment"""

    if not train and not test:
        raise click.UsageError("You must specify either --train or --test (or both).")

    if test and not train:
        if not model_load_path:
            raise click.UsageError("--model-load-path is required when using --test.")
        epsilon = 0.0

    if epsilon_decay is None:
        epsilon_decay = epsilon / (episodes / 2)

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

    # Call your training or testing function here
    run_taxi(
        train=train,
        test=test,
        episodes=episodes,
        render=render,
        learning_rate=learning_rate,
        start_epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        plot=plot
    )

@cli.command()
@click.option('--train', is_flag=True, help='Run training mode')
@click.option('--test', is_flag=True, help='Run test mode')
@click.option('--model-save-path', default='models/toy_text/blackjack.pkl', help='Where to save the pickle model (used in training)')
@click.option('--model-load-path', help='Path to pickle load model from, in test-only mode it is required')
@click.option('--render', is_flag=True, help='Render the environment')
@click.option('--learning-rate', default=0.01, type=float, help='Learning rate')
@click.option('--epsilon', type=float, default=1.0, help='Starting epsilon (will be 0 in test mode)')
@click.option('--epsilon-decay', type=float, help='Epsilon decay rate (default: epsilon / (episodes / 2))')
@click.option('--epsilon-min', type=float, default=0.05, help='Minimum epsilon (default 0.05)')
@click.option('--episodes', type=int, required=True, help='Number of episodes to run')
@click.option('--plot', is_flag=True, help='Plot some statistics from training procedure')
def blackjack(train, test, model_save_path, model_load_path, render, learning_rate, epsilon, epsilon_decay, epsilon_min, episodes, plot):
    """Run the Blackjack experiment"""

    if not train and not test:
        raise click.UsageError("You must specify either --train or --test (or both).")

    if test and not train:
        if not model_load_path:
            raise click.UsageError("--model-load-path is required when using --test.")
        epsilon = 0.0

    if epsilon_decay is None:
        epsilon_decay = epsilon / (episodes / 2)

    if plot and not train:
        plot = False

    if not train and model_save_path:
        model_save_path = None
    elif train and not model_save_path.endswith('.pkl'):
        model_save_path += '.pkl'

    logger.info("Running Blackjack experiment with:")
    logger.info(f"  Mode: {'train' if train else ''} {'test' if test else ''}")
    logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Epsilon: {epsilon}")
    logger.info(f"  Epsilon Decay: {epsilon_decay}")
    logger.info(f"  Epsilon Minimum: {epsilon_min}")
    logger.info(f"  Model Save Path: {model_save_path}")
    logger.info(f"  Model Load Path: {model_load_path}")
    logger.info(f"  Render: {render}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Plots: {plot}")

    # Call your blackjack run function here
    run_blackjack(
        train=train,
        test=test,
        episodes=episodes,
        render=render,
        learning_rate=learning_rate,
        start_epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min = epsilon_min,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        plot=plot
    )


@cli.command()
@click.option('--load-maps-from', type=str, default=None, help='Path to existing maps dataset')
@click.option('--num-maps', default=1000, type=int,
              help='Number of maps to generate')
@click.option('--maps-save-path', default='datasets/toy_text/frozenlake.json', type=str,
              help="Where to store the maps dataset as JSON")
@click.option('--size', default=8, type=int,
              help='Map size (e.g. 8 for 8×8)')
@click.option('--compress', is_flag=True,
              help='Compress the JSON file')
@click.option('--train/--no-train', default=False,
              help='Whether to train on these maps')
@click.option('--test/--no-test', default=False,
              help='Whether to evaluate on held-out maps')
@click.option('--episodes', default=1000, type=int,
              help='Episodes per map')
@click.option('--render', is_flag=True,
              help='Render the environment to screen')
@click.option('--learning-rate', default=1e-3, type=float,
              help='DQN learning rate')
@click.option('--start-epsilon', default=1.0, type=float,
              help='Initial ϵ-greedy value')
@click.option('--final-epsilon', default=0.05, type=float,
              help='Final ϵ after decay')
@click.option('--test-size', default=0.2, type=float,
              help='Fraction of maps held out for testing')
@click.option('--model-save-path', type=str, default=None,
              help='Where to checkpoint trained model')
@click.option('--model-load-path', type=str, default=None,
              help='Path to an existing model to load')
@click.option('--no-plot', 'plot', flag_value=False, default=True,
              help='Disable plotting of training curves')
def frozen_lake(load_maps_from: str,
                num_maps: int,
                maps_save_path: str,
                size: int,
                compress: bool,
                train: bool,
                test: bool,
                episodes: int,
                render: bool,
                learning_rate: float,
                start_epsilon: float,
                final_epsilon: float,
                test_size: float,
                model_save_path: str,
                model_load_path: str,
                plot: bool):

    if load_maps_from:
        dataset = FrozenLakeMaps.load_maps(load_maps_from)
        logger.info(f"Loaded {len(dataset)} maps from {load_maps_from}")

    else:
        """
        Generate FrozenLake maps, then train and/or test a DQN agent.
        """
        dataset = FrozenLakeMaps.generate_dataset(
            num_maps=num_maps,
            size=size,
            filepath=maps_save_path,
            compress=compress
        )
        logger.info(f"Generated {len(dataset)} maps → {maps_save_path}")


    run_frozenlake(
        train=train,
        test=test,
        dataset=dataset,
        episodes=episodes,
        render=render,
        learning_rate=learning_rate,
        start_epsilon=start_epsilon,
        final_epsilon=final_epsilon,
        test_size=test_size,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        plot=plot
    )