import logging

from sklearn.metrics import f1_score


def test(test_df, xgb_model):
    """
    The test function takes a dataframe of test data and the trained model. It makes predictions on the test set,
    and returns a f1-score.

    Args:
        test_df: Pass the test dataframe
        xgb_model: Pass the trained model to the test function

    Returns:
        The f1-score on the test set
    """
    X_test, y_test = test_df.drop(
        columns=['created_account'
                 ]).values, test_df['created_account'].values.astype(int)
    prediction = xgb_model.predict(X_test)
    score = f1_score(y_test, prediction)
    logging.info('Testing finsihed!')
    logging.info('\tTest f1 score {:.3f}\n'.format(score))
    return score
