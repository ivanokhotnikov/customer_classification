from kfp.v2.dsl import (Artifact, Dataset, Input, Metrics, Model, Output,
                        component)

from components.dependencies import (GOOGLE_CLOUD_AIPLATFORM, NUMPY, PANDAS,
                                     PYTHON38, SKLEARN, XGBOOST)


@component(base_image=PYTHON38,
           packages_to_install=[
               PANDAS, NUMPY, SKLEARN, XGBOOST, GOOGLE_CLOUD_AIPLATFORM
           ])
def evaluate(project_id: str, region: str, exp_name: str, threshold: float,
             timestamp: str, validation: Input[Dataset],
             features: Input[Artifact], model: Input[Model],
             eval_metric: Output[Metrics]) -> None:
    """Evaluates the trained model, logs the eval_metric in the pipeline metadata storage

    Args:
        project_id (str): GCP project id
        region (str): GCP project region
        timestamp (str): Timestamp of training
        exp_name (str): Vertex experiment name
        threshold (float): Classifier threshold (prob_prediction > threshold -> 1)
        validation (Input[Dataset]): Validation dataset
        features (Input[Artifact]): Features artifact
        model (Input[Model]): Model file
        eval_metric (Output[Metrics]): Evaluation metric
    """
    import json
    import logging

    import google.cloud.aiplatform as aip
    import joblib
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import f1_score

    with open(features.path, 'r') as features_file:
        features_list = json.load(features_file)
    validation_df = pd.read_csv(validation.path + '.csv')
    X_validation, y_validation = validation_df[features_list], validation_df[
        'created_account']
    dvalidation = xgb.DMatrix(X_validation,
                              y_validation,
                              feature_names=features_list)
    xgb_model = joblib.load(model.path + '.joblib')
    ### Make predictions and scoring
    logging.info('Make predictions on validation set')
    prediction = xgb_model.predict(dvalidation)
    score = f1_score(y_validation,
                     np.where(prediction > threshold, 1, 0),
                     pos_label=0)
    logging.info('Using threshold = ' + str(threshold))
    logging.info('F1 score (0) for validation set is ' + str(score))
    with open(eval_metric.path + '.json', 'w') as metric_file:
        json.dump({'f1_score': score}, metric_file)
    aip.init(experiment=exp_name, project=project_id, location=region)
    aip.start_run(run=timestamp, resume=True)
    aip.log_metrics({'eval_f1_score_0': score})
    aip.log_params({'val_shape': str(validation_df.shape)})
