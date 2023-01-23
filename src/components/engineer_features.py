from kfp.v2.dsl import Dataset, Input, Output, component

from components.dependencies import NUMPY, PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS, NUMPY])
def engineer_features(interim: Input[Dataset], processed: Output[Dataset],
                      num_cat_cols: int) -> None:
    """Creates and adds new features (work_experience_length, dummy variables for the most common categorical features, capital change feature)

    #### Feature engineering
    # Feature preprocess to create features for a model input
    # Process:
    #     1. Create work_experience_length
    #     2. Create dummy variables
    #     3. Convert the target variable into numerical variable
    #     4. Create capital change feature

    Args:
        interim (Input[Dataset]): Interim dataset
        processed (Output[Dataset]): Processed dataset
        num_cat_cols (int): Number of categories to use in dummy variables generation
    """
    import logging

    import numpy as np
    import pandas as pd

    interim_df = pd.read_csv(interim.path + '.csv')
    logging.info('Creating work_experience_length')
    interim_df['work_experience_length'] = (
        interim_df['years_with_employer'] * 12 +
        interim_df['months_with_employer'])

    logging.info('Creating dummy variables')
    categorical_cols = [
        'sex',
        'religion',
        'workclass',
        'name_title',
        'education',
        'marital_status',
        'occupation_level',
    ]

    for col in categorical_cols:
        dummies = pd.get_dummies(interim_df[col], drop_first=True, prefix=col)
        interim_df = pd.concat([interim_df, dummies], 1)

    selected_categorical_cols = ['town']

    # go through all the columns selected above
    for col in selected_categorical_cols:
        categorical = pd.DataFrame(interim_df[col].value_counts())
        categorical = categorical[categorical[col] > num_cat_cols].index

        # go through all the unique values, which appeared more than 100 times
        # and create dummy variables
        for name in categorical:
            interim_df[name] = np.where(interim_df[col] == name, 1, 0)

    logging.info('Creating capital change feature training')
    interim_df['capital_change'] = interim_df['capital_gain'] - interim_df[
        'capital_loss']
    interim_df.to_csv(processed.path + '.csv', index=False)
