import os
import sys
from pathlib import Path
from typing import List, cast

import click
import mlflow
from mlflow.entities import Run

from .convert import fetch_runs, write_runs


@click.group()
def cli():
    pass


@cli.command()
@click.option("--tracking-uri", prompt=True, default=lambda: os.environ.get("MLFLOW_TRACKING_URI"))
@click.option("--logdir", prompt=True)
@click.option("--search-filter", prompt=True, default="")
@click.option("--start-tb", default=True)
def export_runs(tracking_uri, logdir, search_filter, start_tb):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    click.echo(
        f'Searching "{search_filter}" in \nMLflow Tracking Store: {mlflow.get_tracking_uri()}'
    )
    runs: List[Run] = fetch_runs(search_filter)
    click.echo(f"Found {len(runs)} runs. Exporting...")

    os.makedirs(logdir, exist_ok=True)

    with click.progressbar(runs, label="Runs Exported") as run_list:
        run_list = cast(List[Run], run_list)
        write_runs(run_list, Path(logdir))
    if start_tb:
        os.system(f"{sys.executable} -m tensorboard.main --logdir={logdir}")


@cli.command()
def watch_runs():
    click.echo("Not yet implemented")


if __name__ == "__main__":
    cli()
