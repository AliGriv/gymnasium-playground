import click
from toy_text.cli import cli as toy_text_cli
from classic_control.cli import cli as control_classic_cli

@click.group()
def cli():
    """Top-level CLI for gym-playground."""
    pass

cli.add_command(toy_text_cli, name="toy-text")
cli.add_command(control_classic_cli, name="classic-control")