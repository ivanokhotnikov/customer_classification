from kfp.v2.dsl import Artifact, Input, Metrics, Model, component

from components.dependencies import (GOOGLE_CLOUD_AIPLATFORM, PYTHON38,
                                     SKLEARN, XGBOOST)


@component(
    base_image=PYTHON38,
    packages_to_install=[SKLEARN, GOOGLE_CLOUD_AIPLATFORM, XGBOOST],
)
def upload_model_to_registry(project_id: str, region: str, deploy_image: str,
                             models_path: str, timestamp: str,
                             model: Input[Model], params: Input[Artifact],
                             eval_metric: Input[Metrics],
                             test_metric: Input[Metrics]) -> None:
    """Uploads the successful model and its metrics to the GCS bucket defined in `models_path`.

    Args:
        project_id (str): GCP project id
        region (str): GCP project region
        deploy_image (str): Container image to use in the model endpoint
        models_path (str): Path to the model registry in the GCS bucket
        timestamp (str): Timestamp of training
        model (Input[Model]): Model
        params (Input[Artifact]): Parameters
        eval_metric (Input[Metrics]): Evaluation metric
        test_metric (Input[Metrics]): Test metric
    """
    import json
    import os

    import google.cloud.aiplatform as aip
    import joblib
    model = joblib.load(model.path + '.joblib')
    joblib.dump(model, os.path.join(models_path, 'model.joblib'))

    with open(eval_metric.path + '.json', 'r') as fp:
        eval_metric_dict = json.load(fp)
    with open(os.path.join(models_path, 'eval_metric.json'), 'w') as fp:
        json.dump(eval_metric_dict, fp)

    with open(test_metric.path + '.json', 'r') as fp:
        test_metric_dict = json.load(fp)
    with open(os.path.join(models_path, 'test_metric.json'), 'w') as fp:
        json.dump(test_metric_dict, fp)

    with open(params.path + '.json', 'r') as fp:
        params = json.load(fp)
    with open(os.path.join(models_path, 'params.json'), 'w') as fp:
        json.dump(params, fp)

    aip.init(project=project_id, location=region)
    joblib.dump(model, os.path.join(models_path, 'registry', 'model.joblib'))
    model = aip.Model.upload(project=project_id,
                             location=region,
                             display_name=timestamp,
                             artifact_uri=os.path.join(models_path,
                                                       'registry'),
                             serving_container_image_uri=deploy_image,
                             is_default_version=True)
