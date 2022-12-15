from pathlib import Path
from typing import Dict, List

import mlflow
from mlflow.entities import Metric, Run
from mlflow.tracking import MlflowClient
from tensorboard.summary import Writer


# TODO: convert to generators to pass through pagination
def fetch_runs(filter: str) -> List[Run]:
    # TODO: Add the filter string back in - AML seems to have a search bug. wamartin on point to investigate
    # TODO: switch to MLflow Client instead of fluent
    result = mlflow.search_runs(
        filter_string="", search_all_experiments=True, output_format="list", max_results=100
    )
    assert isinstance(result, List)
    if len(result) == 0:
        raise ValueError("Search syntax returned no results")
    return result


# TODO: Pivot by step to do batched writes? (writer.add_scalars())
def fetch_metrics(client: MlflowClient, run: Run) -> Dict[str, List[Metric]]:
    metric_names: List[str] = run.data.metrics.keys()

    metrics = {}
    for metric_name in metric_names:
        metric_history: List[Metric] = client.get_metric_history(run.info.run_id, metric_name)
        metrics[metric_name] = metric_history

    return metrics


def write_metric(writer: Writer, metric: Metric) -> None:
    writer.add_scalar(metric.key, metric.value, metric.step, wall_time=metric.timestamp)


# logdir/
#   run1/
#       *.event
#   run2/
def write_runs(runs: List[Run], log_dir: Path) -> None:
    client = MlflowClient()
    for run in runs:
        run_dir: Path = log_dir / run.info.run_id
        # TODO: more intelligent sync - for now assume if a run directory is already present, we'll skip it
        if run_dir.exists():
            continue
        run_dir.mkdir()

        metrics = fetch_metrics(client, run)

        run_writer: None | Writer = None
        try:
            run_writer = Writer(str(run_dir))
            for history in metrics.values():
                for metric_val in history:
                    write_metric(run_writer, metric_val)
        finally:
            if run_writer:
                run_writer.close()
