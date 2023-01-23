import itertools
import subprocess

ETA_SPACE = [0.02, 0.04]
MAX_DEPTH_SPACE = [6, 14]
SUBSAMPLE_SPACE = [0.1, 0.8]
MIN_CHILD_WEIGHT_SPACE = [1, 5]


def grid_search():
    for eta, max_depth, subsample, min_child_weight in itertools.product(
            ETA_SPACE, MAX_DEPTH_SPACE, SUBSAMPLE_SPACE,
            MIN_CHILD_WEIGHT_SPACE):
        subprocess.run([
            'python',
            './src/xgb_training_pipeline.py',
            '--eta',
            f'{eta}',
            '--max_depth',
            f'{max_depth}',
            '--subsample',
            f'{subsample}',
            '--min_child_weight',
            f'{min_child_weight}',
            '--enable_caching',
        ])


if __name__ == '__main__':
    grid_search()
