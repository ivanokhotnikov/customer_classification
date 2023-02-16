import logging

from sklearn.model_selection import train_test_split


def split(args, df, features):
    """
    The split function takes in a dataframe and splits it into train and test sets.
    The split is stratified by the 'created_account' column to ensure that the same proportion of
    users are in each set as is represented in the original dataset. The function also accepts a list
    of features for use in split.

    Args:
        args: Pass arguments to the function
        df: Split the dataframe into train and test
        features: Specify the columns that are used for training

    Returns:
        Train and test dataframes
    """
    nonull_df = df.loc[df['created_account'].notnull(),
                       features + ['created_account']]
    _, test_df = train_test_split(nonull_df,
                                  shuffle=True,
                                  random_state=args.seed,
                                  test_size=args.test_to_all_split,
                                  stratify=nonull_df['created_account'])
    train_df = df.loc[:, features + ['created_account']].drop(test_df.index)
    logging.info('Processed data split!')
    logging.info('\tpreprocessed train shape: {}'.format(train_df.shape))
    logging.info('\ttest shape: {}\n'.format(test_df.shape))
    return train_df, test_df
