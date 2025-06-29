import click
from common.loggerConfig import logger
from classic_control.mountain_car.main import run as run_mountain_car

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
@click.option('--learning-rate', default=0.9, type=float, help='Learning rate')
@click.option('--epsilon', type=float, default=1.0, help='Starting epsilon (will be 0 in test mode)')
@click.option('--epsilon-decay', type=float, help='Epsilon decay rate (default: epsilon / (episodes / 2))')
@click.option('--episodes', type=int, required=True, help='Number of episodes to run')
@click.option('--plot', is_flag=True, help='Plot some statistics from training procedure')
def mountain_car(train, test, model_save_path, model_load_path, render, learning_rate, epsilon, epsilon_decay, episodes, plot):
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
        plot=plot
    )