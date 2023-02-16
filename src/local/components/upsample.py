import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC


def upsample(args, df, cat_features):
    """
    The upsample function takes in a dataframe and upsamples the minority class to balance out the dataset.
    The function returns a new dataframe with balanced classes.

    Args:
        args: Pass parameters to the function
        df: Specify the dataframe that is being upsampled
        cat_features: Specify the indices of categorical features

    Returns:
        The upsampled training data
    """
    X, y = df.drop(columns=['created_account'
                            ]).values, df['created_account'].values.astype(int)
    sm = SMOTENC(sampling_strategy='auto',
                 random_state=args.seed,
                 categorical_features=cat_features)
    X_res, y_res = sm.fit_resample(X, y)
    logging.info('Train upsampled!')
    logging.info('\tclass imbalance before: {:.3f}'.format(
        sum(y == 1) / sum(y == 0)))
    logging.info('\tclass imbalance after: {}'.format(
        sum(y_res == 1) / sum(y_res == 0)))
    train_df = pd.DataFrame(np.concatenate(
        (X_res, y_res.astype(int).reshape(-1, 1)), axis=1),
                            columns=df.columns)
    logging.info('\tfinal train shape: {}\n'.format(train_df.shape))
    return train_df
