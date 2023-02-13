from kfp.v2.dsl import Dataset, Input, Output, component

from components.dependencies import (GOOGLE_CLOUD_AIPLATFORM, NUMPY, PANDAS,
                                     PYTHON38, SKLEARN)


@component(
    base_image=PYTHON38,
    packages_to_install=[PANDAS, SKLEARN, NUMPY, GOOGLE_CLOUD_AIPLATFORM])
def split_data(seed: int, val_to_train_split: float, processed: Input[Dataset],
               train: Output[Dataset], validation: Output[Dataset],
               test: Output[Dataset]) -> None:
    '''Splits the data in train, validation and test sets. Withhold those labels without created account for testing on.

    Args:
        seed (int): Random seed
        val_to_train_split (float): Split proportion
        processed (Input[Dataset]): Processed dataset
        train (Output[Dataset]): Train dataset
        validation (Output[Dataset]): Validation dataset
        test (Output[Dataset]): Test dataset
    '''
    import logging

    import google.cloud.aiplatform as aip
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    processed_df = pd.read_csv(processed.path + '.csv')
    logging.info('Treating the target variable')
    # Withhold those without created account for testing on
    train_df, test_df = processed_df[processed_df['created_account'].notnull(
    )], processed_df[processed_df['created_account'].isnull()]
    train_df['created_account'] = np.where(
        train_df['created_account'] == 'Yes', 1, 0)
    test_df['created_account'] = np.where(test_df['created_account'] == 'Yes',
                                          1, 0)
    train_df, validation_df = train_test_split(
        train_df,
        test_size=val_to_train_split,
        random_state=seed,
        stratify=train_df['created_account'])
    # Check the shape of the data
    for (name, df, ppl) in zip(['training', 'validation', 'test'],
                               [train_df, validation_df, test_df],
                               [train, validation, test]):
        logging.info(f'The shape of {name} set is ' + str(df.shape))
        df.to_csv(ppl.path + '.csv', index=False)
