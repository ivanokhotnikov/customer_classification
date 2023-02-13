import numpy as np
from sklearn.model_selection import train_test_split


def split(args, df):
    train_df, test_df = df[df['created_account'].notnull()], df[
        df['created_account'].isnull()]
    train_df['created_account'] = np.where(
        train_df.loc[:, 'created_account'] == 'Yes', 1, 0)
    test_df['created_account'] = np.where(
        test_df.loc[:, 'created_account'] == 'Yes', 1, 0)
    train_df, validation_df = train_test_split(
        train_df,
        test_size=args.val_to_train_split,
        random_state=args.seed,
        stratify=train_df['created_account'])
    print('Processed data was split!')
    return train_df, test_df, validation_df
