import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument('--exp-name', type=str, default='default')
    # Training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.08)
    parser.add_argument('--num-cat-cols', type=int, default=100)
    parser.add_argument('--val-to-train-split', type=float, default=0.2)
    parser.add_argument('--test-to-all-split', type=float, default=0.2)
    parser.add_argument('--boost-rounds', type=int, default=300)
    parser.add_argument('--folds', type=int, default=5)
    # Hyper parameters
    parser.add_argument('--eta', type=float, default=1e-4)
    parser.add_argument('--max-depth', type=int, default=50)
    parser.add_argument('--n_estimators', type=int, default=1000)
    parser.add_argument('--min-child-weight', type=int, default=4)
    parser.add_argument('--subsample', type=float, default=1.)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval-metric', type=str, default='logloss'),
    parser.add_argument('--grow-policy', type=str, default='depthwise')
    parser.add_argument('--reg-lambda', type=float, default=1.)
    parser.add_argument('--reg-alpha', type=float, default=0.)
    # Ignore Ipykernel
    parser.add_argument('--ip', default=argparse.SUPPRESS)
    parser.add_argument('--stdin', default=argparse.SUPPRESS)
    parser.add_argument('--control', default=argparse.SUPPRESS)
    parser.add_argument('--hb', default=argparse.SUPPRESS)
    parser.add_argument('--Session.signature_scheme',
                        default=argparse.SUPPRESS)
    parser.add_argument('--Session.key', default=argparse.SUPPRESS)
    parser.add_argument('--shell', default=argparse.SUPPRESS)
    parser.add_argument('--transport', default=argparse.SUPPRESS)
    parser.add_argument('--iopub', default=argparse.SUPPRESS)
    parser.add_argument('--f', default=argparse.SUPPRESS)
    return parser.parse_args()
