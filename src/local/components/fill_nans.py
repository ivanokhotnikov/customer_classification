import logging

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def fill_nans(args, df):
    """
    The fill_nans function fills in the nans in the created_account column with values from a trained XGBoost model.
    The function takes as input:
        args: The arguments passed into this script, which contains all of the hyperparameters needed to train and test an XGBoost model.
              This is used to pass into the XGBClassifier object that will be trained on non-null data from our training set.

    Args:
        args: Pass the command line arguments
        df: Fill the nans in the created_account column

    Returns:
        The dataframe with the nulls filled in
    """
    nonull_df = df.loc[df['created_account'].notnull()]
    null_df = df.loc[df['created_account'].isna()]
    X, y = nonull_df.drop(
        columns=['created_account'
                 ]).values, nonull_df['created_account'].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=args.seed,
                                                        stratify=y)
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
    xgb_classifier = XGBClassifier(**params)
    xgb_classifier.fit(X_train, y_train)
    pred = xgb_classifier.predict(X_test)
    score = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    null_X = null_df.drop(columns=['created_account']).values
    null_y = xgb_classifier.predict(null_X)
    df.loc[df['created_account'].isna(), 'created_account'] = null_y
    logging.info('Nans filled in train!')
    logging.info('\tnulls in target before: {}'.format(
        null_df['created_account'].isna().sum()))
    logging.info('\tnulls in target after: {}\n'.format(
        df['created_account'].isna().sum()))
    return df
