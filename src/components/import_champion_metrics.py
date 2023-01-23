from kfp.v2.dsl import Metrics, Output, component

from components.dependencies import PYTHON38


@component(base_image=PYTHON38)
def import_champion_metrics(models_path: str,
                            champion_metric: Output[Metrics]) -> None:
    import json
    import os

    with open(os.path.join(models_path, 'test_metric.json'),
              'r') as registry_metrics_file:
        champion_metrics_dict = json.load(registry_metrics_file)
    with open(champion_metric.path + '.json', 'w') as pipeline_metrics_file:
        json.dump(champion_metrics_dict, pipeline_metrics_file)
