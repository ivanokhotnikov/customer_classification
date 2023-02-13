import pandas as pd
import numpy as np


def featurize(df):
    df['work_experience_length'] = (df['years_with_employer'] * 12 +
                                    df['months_with_employer'])

    categorical_cols = [
        'sex', 'religion', 'workclass', 'name_title', 'education',
        'marital_status', 'occupation_level'
    ]

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], drop_first=True, prefix=col)
        df = pd.concat([df, dummies], 1)

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
    print('Features were added!')
    return df
