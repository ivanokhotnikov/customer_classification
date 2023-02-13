from kfp.v2.dsl import Artifact, Dataset, Input, Model, Output, component

from components.dependencies import (GOOGLE_CLOUD_AIPLATFORM, PANDAS, PYTHON38,
                                     SKLEARN, XGBOOST)


@component(
    base_image=PYTHON38,
    packages_to_install=[PANDAS, SKLEARN, XGBOOST, GOOGLE_CLOUD_AIPLATFORM])
def train(project_id: str, region: str, exp_name: str, timestamp: str,
          train: Input[Dataset], features: Input[Artifact],
          model: Output[Model], parameters: Output[Artifact],
          boost_rounds: int, eta: float, max_depth: int, min_child_weight: int,
          subsample: float, objective: str, eval_metric: str, grow_policy: str,
          reg_lambda: float, reg_alpha: float) -> None:
    """Train xgb model

    Args:
        project_id (str): GCP project id
        region (str): GCP project region
        timestamp (str): Timestamp of training
        exp_name (str): Vertex experiment name
        train (Input[Dataset]): Train dataset
        features (Input[Artifact]): Features
        model (Output[Model]): Model
        parameters (Output[Artifact]): Parameters
        boost_rounds (int): Number of boosting rounds
        eta (float): Learning rate
        max_depth (int): Max tree depth
        min_child_weight (int): Minimum sum of instance weight needed in a child
        subsample (float): Subsample ratio of the training instances
        objective (str): Objective to train on
        eval_metric (str): Evaluation metric
        grow_policy (str): Grow policy
        reg_lambda (float): L2 regularization term on weights. Increasing this value will make model more conservative
        reg_alpha (float): L1 regularization term on weights. Increasing this value will make model more conservative.
    """
    import json

    import google.cloud.aiplatform as aip
    import joblib
    import pandas as pd
    import xgboost as xgb

    with open(features.path, 'r') as features_file:
        features_list = list(json.load(features_file))
    train_df = pd.read_csv(train.path + '.csv')
    X_train, y_train = train_df[features_list], train_df['created_account']
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=features_list)
    params = {
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'objective': objective,
        'eval_metric': eval_metric,
        'grow_policy': grow_policy,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha
    }
    xgb_model = xgb.train(params, dtrain, boost_rounds, verbose_eval=False)
    joblib.dump(xgb_model, model.path + '.joblib')
    with open(parameters.path + '.json', 'w') as params_file:
        json.dump(params, params_file)
    aip.init(experiment=exp_name, project=project_id, location=region)
    aip.start_run(run=timestamp, resume=True)
    aip.log_params({'train_shape': str(train_df.shape)})
