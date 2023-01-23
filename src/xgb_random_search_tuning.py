import random
import subprocess

random.seed(0)

TRIALS = 5


def random_search():
    for trial in range(TRIALS):
        subprocess.run([
            'python',
            'src/xgb_training_pipeline.py',
            '--eta',
            f'{random.uniform(0.02, 0.04)}',
            '--max_depth',
            f'{random.randint(6, 14)}',
            '--subsample',
            f'{random.uniform(0.1, 0.8)}',
            '--min_child_weight',
            f'{random.randint(1, 5)}',
            '--enable_caching',
        ])


if __name__ == '__main__':
    random_search()
