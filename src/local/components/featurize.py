import logging

import numpy as np
import pandas as pd


def featurize(df):
    """
    The featurize function :
        - Adds a new column called work_experience_length which is the product of years with employer and months with employer.
        - Creates dummy variables for categorical columns.  For example, if there are more than 100 people who identify as
          female in the dataset, then a female column will be added to represent that information.

    Args:
        df: Pass in the dataframe that is being featurized

    Returns:
        A dataframe with features added
    """
    df['work_experience_length'] = (df['years_with_employer'] * 12 +
                                    df['months_with_employer'])

    categorical_cols = [
        'sex', 'religion', 'workclass', 'name_title', 'education',
        'marital_status', 'occupation_level'
    ]

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], drop_first=True, prefix=col)
        df = pd.concat([df, dummies], axis=1)

    selected_categorical_cols = ['town']

    # go through all the columns selected above
    for col in selected_categorical_cols:
        categorical = pd.DataFrame(df[col].value_counts())
        categorical = categorical[categorical[col] > 100].index

        # go through all the unique values, which appeared more than 100 times
        # and create dummy variables
        for name in categorical:
            df[name] = np.where(df[col] == name, 1, 0)

    df['capital_change'] = df['capital_gain'] - df['capital_loss']
    df.loc[df['created_account'] == 'Yes', 'created_account'] = 1
    df.loc[df['created_account'] == 'No', 'created_account'] = 0
    logging.info('Features added!\n')
    return df
