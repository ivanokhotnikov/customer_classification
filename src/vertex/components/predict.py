from kfp.v2.dsl import (Artifact, Dataset, Input, Metrics, Model, Output,
                        component)

from components.dependencies import (GOOGLE_CLOUD_AIPLATFORM, NUMPY, PANDAS,
                                     PYTHON38, SKLEARN, XGBOOST)


@component(base_image=PYTHON38,
           packages_to_install=[
               PANDAS, NUMPY, SKLEARN, XGBOOST, GOOGLE_CLOUD_AIPLATFORM
           ])
def predict(project_id: str, region: str, exp_name: str, threshold: float,
            timestamp: str, test: Input[Dataset], features: Input[Artifact],
            model: Input[Model], test_metric: Output[Metrics],
            pred: Output[Artifact]) -> None:
    """Run prediction on test dataset

    Args:
        project_id (str): GCP project id
        region (str): GCP project region
        exp_name (str): Vertex experiment name
        threshold (float): Classifier threshold (prob_prediction > threshold -> 1)
        timestamp (str): Timestamp of training
        test (Input[Dataset]): Test dataset
        features (Input[Artifact]): Features
        model (Input[Model]): Model
        test_metric (Output[Metrics]): Test metric
        pred (Output[Artifact]): Predictions
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
    test_df = pd.read_csv(test.path + '.csv')
    X_test, y_test = test_df[features_list], test_df['created_account']
    dtest = xgb.DMatrix(X_test, y_test, feature_names=features_list)
    xgb_model = joblib.load(model.path + '.joblib')
    ### Make predictions and scoring

    logging.info('Make predictions on test set')
    prediction = xgb_model.predict(dtest)
    score = f1_score(y_test,
                     np.where(prediction > threshold, 1, 0),
                     pos_label=0)
    logging.info(f'Using threshold = {threshold}')
    logging.info(f'F1 score for test set is {score}')

    logging.info(
        f'The number of potential target customers on the test set is {(prediction > threshold).sum()}, out of {len(prediction)} which is {((prediction > threshold).sum() / len(prediction)) * 100} % of the customers'
    )
    aip.init(experiment=exp_name, project=project_id, location=region)
    aip.start_run(run=timestamp, resume=True)
    with open(test_metric.path + '.json', 'w') as metrics_file:
        json.dump({'f1_score': score}, metrics_file)
    pd.DataFrame(prediction).to_csv(pred.path + '.csv', index=False)
    aip.log_metrics({'test_f1_score_0': score})
    aip.log_params({'test_shape': str(test_df.shape)})
    aip.end_run()
    # The model predicted  that 3,004 customers out of 28,949 will likely create an account.
    # This is 10.38% of the customers and approximately corresponding to the distribution of train and validation set.
