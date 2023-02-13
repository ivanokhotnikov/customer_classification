import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score


def test(args, test_df, xgb_model, features):
    X_test, y_test = test_df[features], test_df['created_account']
    dtest = xgb.DMatrix(X_test, y_test, feature_names=features)
    prediction = xgb_model.predict(dtest)
    print('Test finsihed!')
    return f1_score(y_test,
                    np.where(prediction > args.threshold, 1, 0),
                    pos_label=0)
