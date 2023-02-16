import logging

import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def train(args, df):
    """
    The train function trains a model in a cross-validation loop and returns the fold averaged f1-score.

    Args:
        args: Pass parameters to the train function
        df: Train the model

    Returns:
        The average validation score and the trained model
    """
    X, y = df.drop(columns=['created_account'
                            ]).values, df['created_account'].values.astype(int)
    params = {
        'eta': args.eta,
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective,
        'eval_metric': args.eval_metric,
        'grow_policy': args.grow_policy,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha,
        'random_state': args.seed,
        'tree_method': 'hist'
    }
    xgb_classifier = xgb.XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=args.folds,
                          shuffle=True,
                          random_state=args.seed)
    val_score = 0
    logging.info('Training started!')
    for i, (train_id, val_id) in enumerate(skf.split(X, y), start=1):
        xgb_classifier.fit(X[train_id], y[train_id])
        pred = xgb_classifier.predict(X[val_id])
        score = f1_score(y[val_id], pred)
        val_score += score
        logging.info('\tFold {}: val f1 score {:.3f}'.format(i, score))
    val_score /= args.folds
    logging.info('Training finished!\n')
    return score, xgb_classifier
