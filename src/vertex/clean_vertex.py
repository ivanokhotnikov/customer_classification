import argparse
import logging
import sys
import time

import google.cloud.aiplatform as aip
from google.api_core.exceptions import NotFound
from google.cloud.storage import Client

from xgb_training_pipeline import get_env


def clean_vertex(
    clean_custom_jobs: bool = False,
    clean_pipeline_jobs: bool = False,
    clean_artifacts: bool = False,
    clean_models: bool = False,
    clean_tensorboards: bool = False,
    clean_experiments: bool = False,
    clean_metadata_store: bool = False,
    clean_tensorboard_events: bool = False,
    clean_all: bool = False,
    verbosity: bool = False,
):

    env = get_env()
    if verbosity:
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    logging.info(f'clean_custom_jobs is set to {clean_custom_jobs}')
    logging.info(f'clean_pipeline_jobs is set to {clean_pipeline_jobs}')
    logging.info(f'clean_artifacts is set to {clean_artifacts}')
    logging.info(f'clean_models is set to {clean_models}')
    logging.info(f'clean_tensorboards is set to {clean_tensorboards}')
    logging.info(f'clean_experiments is set to {clean_experiments}')
    logging.info(f'clean_metadata_store is set to {clean_metadata_store}')
    logging.info(
        f'clean_tensorboard_events is set to {clean_tensorboard_events}')
    logging.info(f'clean_all is set to {clean_all}')
    if clean_all:
        clean_custom_jobs = True
        clean_pipeline_jobs = True
        clean_artifacts = True
        clean_models = True
        clean_tensorboards = True
        clean_experiments = True
        clean_metadata_store = True
        clean_tensorboard_events = True
    elif not any([
            clean_custom_jobs,
            clean_pipeline_jobs,
            clean_artifacts,
            clean_models,
            clean_tensorboards,
            clean_experiments,
            clean_metadata_store,
            clean_tensorboard_events,
    ]):
        logging.info(f'No flags set to clean!')
        sys.exit()
    aip.init(
        project=env['project_id'],
        location=env['region'],
    )
    try:
        if clean_custom_jobs:
            for job in aip.CustomJob.list():
                job.delete()
                time.sleep(1.1)
            logging.info(f'Custom jobs have been cleaned in {env["region"]}')
        if clean_pipeline_jobs:
            for job in aip.PipelineJob.list():
                job.delete()
                time.sleep(1.1)
            logging.info(f'Custom jobs have been cleaned in {env["region"]}')
        if clean_artifacts:
            for art in aip.Artifact.list():
                art.delete()
                time.sleep(1.1)
            logging.info(f'Artifacts have been cleaned in {env["region"]}')
        if clean_models:
            for model in aip.Model.list():
                model.delete()
                time.sleep(1.1)
            logging.info(f'Models have been cleaned in {env["region"]}')
        if clean_tensorboards:
            for tb in aip.Tensorboard.list():
                tb.delete()
                time.sleep(1.1)
            logging.info(f'Tensorboards have been cleaned in {env["region"]}')
        if clean_experiments:
            for exp in aip.Experiment.list():
                for exp_run in aip.ExperimentRun.list(experiment=exp):
                    exp_run.delete()
                    time.sleep(1.1)
                logging.info(
                    f'Experiment runs in {exp.name} have been cleaned in {env["region"]}'
                )
                exp.delete()
                time.sleep(1.1)
            logging.info(f'Experiments have been cleaned in {env["region"]}')
        if clean_metadata_store:
            STORAGE_CLIENT = Client(project=env['project_id'])
            for blob in STORAGE_CLIENT.list_blobs(
                    bucket_or_name='xgb-pipelines',
                    prefix=env['project_number']):
                blob.delete()
            for blob in STORAGE_CLIENT.list_blobs(
                    bucket_or_name=env['pipeline_root'],
                    prefix='vertex_ai_auto_staging'):
                blob.delete()
            logging.info(f'Metadata store has been cleaned in {env["region"]}')
        if clean_tensorboard_events:
            for blob in STORAGE_CLIENT.list_blobs(
                    bucket_or_name=env['pipeline_root'],
                    prefix='tensorboards'):
                blob.delete()
            logging.info(
                f'Tensorboard events have been cleaned in {env["region"]}')
    except NotFound as NotFoundError:
        logging.info(f'{NotFoundError} in {env["region"]}')


def parse_agruments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-custom_jobs',
        '--clean_custom_jobs',
        action='store_true',
    )
    parser.add_argument(
        '-pipeline_jobs',
        '--clean_pipeline_jobs',
        action='store_true',
    )
    parser.add_argument(
        '-artifacts',
        '--clean_artifacts',
        action='store_true',
    )
    parser.add_argument(
        '-models',
        '--clean_models',
        action='store_true',
    )
    parser.add_argument(
        '-tensorboards',
        '--clean_tensorboards',
        action='store_true',
    )
    parser.add_argument(
        '-experiments',
        '--clean_experiments',
        action='store_true',
    )
    parser.add_argument(
        '-metadata',
        '--clean_metadata_store',
        action='store_true',
    )
    parser.add_argument(
        '-tb-events',
        '--clean_tensorboard_events',
        action='store_true',
    )
    parser.add_argument(
        '-all',
        '--clean_all',
        action='store_true',
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        action='store_true',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_agruments()
    clean_vertex(**vars(args))
