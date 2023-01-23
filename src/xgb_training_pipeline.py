import argparse
import os
from datetime import datetime

import google.cloud.aiplatform as aip
from dotenv import load_dotenv
from kfp.v2 import compiler
from kfp.v2.dsl import Artifact, Condition, importer, pipeline

from components import (combine_data, compare_models, engineer_features,
                        evaluate, import_champion_metrics, predict, process,
                        read_campaign_data, read_exchange_rates,
                        read_mortgage_data, split_data, train,
                        upload_model_to_registry)

load_dotenv()


@pipeline(name='trianing-pipeline', pipeline_root=os.environ['PIPELINES_PATH'])
def xgb_training_pipeline(project_id: str, region: str, threshold: float,
                          num_cat_cols: int, seed: int,
                          val_to_train_split: float, boost_rounds: int,
                          exp_name: str, timestamp: str, deploy_image: str,
                          data_path: str, models_path: str, eta: float,
                          max_depth: int, min_child_weight: int,
                          subsample: float, objective: str, eval_metric: str,
                          grow_policy: str, reg_lambda: float,
                          reg_alpha: float) -> None:
    read_campaign_data_task = read_campaign_data(data_path=data_path)
    read_mortgage_data_task = read_mortgage_data(data_path=data_path)
    combine_task = combine_data(
        mortgage=read_mortgage_data_task.outputs['mortgage'],
        campaign=read_campaign_data_task.outputs['campaign'])
    read_exchange_rates_task = read_exchange_rates(data_path=data_path)
    process_task = process(
        combined=combine_task.outputs['combined'],
        exchange_rates=read_exchange_rates_task.outputs['exchange_rates'])
    features_import = importer(
        artifact_uri='gs://features-store/features.json',
        artifact_class=Artifact).set_display_name('import features')
    feature_engineering_task = engineer_features(
        interim=process_task.outputs['interim'], num_cat_cols=num_cat_cols)
    split_data_task = split_data(
        seed=seed,
        processed=feature_engineering_task.outputs['processed'],
        val_to_train_split=val_to_train_split)
    train_task = train(project_id=project_id,
                       region=region,
                       exp_name=exp_name,
                       timestamp=timestamp,
                       train=split_data_task.outputs['train'],
                       features=features_import.output,
                       eta=eta,
                       max_depth=max_depth,
                       min_child_weight=min_child_weight,
                       subsample=subsample,
                       objective=objective,
                       eval_metric=eval_metric,
                       grow_policy=grow_policy,
                       boost_rounds=boost_rounds,
                       reg_alpha=reg_alpha,
                       reg_lambda=reg_lambda)
    evaluate_task = evaluate(project_id=project_id,
                             region=region,
                             exp_name=exp_name,
                             timestamp=timestamp,
                             threshold=threshold,
                             validation=split_data_task.outputs['validation'],
                             features=features_import.output,
                             model=train_task.outputs['model'])
    predict_task = predict(project_id=project_id,
                           region=region,
                           exp_name=exp_name,
                           threshold=threshold,
                           timestamp=timestamp,
                           test=split_data_task.outputs['test'],
                           features=features_import.output,
                           model=train_task.outputs['model'])
    import_champion_metrics_task = import_champion_metrics(
        models_path=models_path)
    compare_task = compare_models(
        challenger_metrics=predict_task.outputs['test_metric'],
        champion_metrics=import_champion_metrics_task.
        outputs['champion_metric'])
    with Condition(compare_task.output == 'true'):
        upload_model_to_registry(
            project_id=project_id,
            region=region,
            deploy_image=deploy_image,
            timestamp=timestamp,
            models_path=models_path,
            model=train_task.outputs['model'],
            params=train_task.outputs['parameters'],
            eval_metric=evaluate_task.outputs['eval_metric'],
            test_metric=predict_task.outputs['test_metric'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--compile_only', action='store_true')
    parser.add_argument('--enable_caching', action='store_true')
    # Training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.08)
    parser.add_argument('--num_cat_cols', type=int, default=100)
    parser.add_argument('--val_to_train_split', type=float, default=0.2)
    parser.add_argument('--boost_rounds', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='default')
    # Hyper parameters
    parser.add_argument('--eta', type=float, default=0.03)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--min_child_weight', type=int, default=4)
    parser.add_argument('--subsample', type=float, default=1.)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval_metric', type=str, default='logloss'),
    parser.add_argument('--grow_policy', type=str, default='depthwise')
    parser.add_argument('--reg_lambda', type=float, default=1.)
    parser.add_argument('--reg_alpha', type=float, default=0.)
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    compiler.Compiler().compile(
        pipeline_func=xgb_training_pipeline,
        package_path='./compiled/training_pipeline.json')
    if not args.compile_only:
        aip.init(project=os.environ['PROJECT_ID'],
                 location=os.environ['LOCATION'],
                 experiment=args.exp_name,
                 staging_bucket=os.environ['PIPELINES_PATH'])
        aip.start_run(run=timestamp)
        training_parameters = {
            'seed': args.seed,
            'threshold': args.threshold,
            'num_cat_cols': args.num_cat_cols,
            'val_to_train_split': args.val_to_train_split,
            'boost_rounds': args.boost_rounds
        }
        hyperparameters = {
            'eta': args.eta,
            'max_depth': args.max_depth,
            'min_child_weight': args.min_child_weight,
            'subsample': args.subsample,
            'objective': args.objective,
            'eval_metric': args.eval_metric,
            'grow_policy': args.grow_policy,
            'reg_lambda': args.reg_lambda,
            'reg_alpha': args.reg_alpha
        }
        aip.log_params(training_parameters)
        aip.log_params(hyperparameters)
        train_job = aip.PipelineJob(
            project=os.environ['PROJECT_ID'],
            location=os.environ['LOCATION'],
            pipeline_root=os.environ['PIPELINES_PATH'],
            enable_caching=args.enable_caching,
            display_name=timestamp,
            template_path='./compiled/training_pipeline.json',
            parameter_values={
                'project_id': os.environ['PROJECT_ID'],
                'region': os.environ['LOCATION'],
                'timestamp': timestamp,
                'exp_name': args.exp_name,
                'deploy_image': os.environ['DEPLOY_IMAGE'],
                'data_path': os.environ['DATA_PATH'],
                'models_path': os.environ['MODELS_PATH'],
                **training_parameters,
                **hyperparameters
            })
        train_job.submit(service_account=os.environ['SA_EMAIL'])
