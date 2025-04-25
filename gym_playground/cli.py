import click
from toy_text.cli import cli as toy_text_cli

@click.group()
def cli():
    """Top-level CLI for gym-playground."""
    pass

cli.add_command(toy_text_cli, name="toy-text")