import click
from analysis.ou_noise_analysis import run_ou_noise_analysis
from analysis.mountain_car_ddpg import run_mountain_car_ddpg_analysis
from common.loggerConfig import logger
from pathlib import Path
@click.group()
def cli():
    """Run some analysis tools for gym-playground."""
    pass


@cli.command()
@click.option("--mu", default=0.0, type=float, show_default=True,
              help="Long-term mean (drift target).")
@click.option("--theta", default=0.05, type=float, show_default=True,
              help="Rate of mean reversion.")
@click.option("--sigma", default=0.3, type=float, show_default=True,
              help="Noise scale.")
@click.option("--dt", default=0.4, type=float, show_default=True,
              help="Time step size.")
@click.option("--x0", default=None, type=float, show_default=True,
              help="Initial value of the process (None defaults to mu).")
@click.option("--min-sigma", "min_sigma", default=0.01, type=float, show_default=True,
              help="Minimum value to which sigma can decay.")
@click.option("--sigma-decay", "sigma_decay", default=0.9995, type=float, show_default=True,
              help="Multiplicative factor for sigma decay per step.")
@click.option("--steps", default=5000, type=int, show_default=True,
              help="Number of steps to simulate.")
def ou_noise(mu, theta, sigma, dt, x0, min_sigma, sigma_decay, steps):
    run_ou_noise_analysis(mu, theta, sigma, dt, x0, min_sigma, sigma_decay, steps)


@cli.command()
@click.option("--model", type=str, required=True, help="Path to the trained DDPG model.")
def mountain_car_ddpg(model):
    """
    Analyze the DDPG model for Mountain Car Continuous environment.

    :param model: Path to the trained DDPG model file.
    """

    model = Path(model)
    if not model.exists():
        raise click.UsageError(f"Model path {model} does not exist.")
    if not model.suffix in ['.pt']:
        raise click.UsageError("Model path must end with '.pt'.")
    if not model.with_suffix('.json').exists():
        raise click.UsageError(f"Model JSON file {model.with_suffix('.json')} does not exist.")
    logger.info(f"Running Mountain Car DDPG analysis on model: {model}")
    run_mountain_car_ddpg_analysis(model)