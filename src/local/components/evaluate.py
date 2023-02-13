import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score


def evaluate(args, validation_df, xgb_model, features):
    X_validation, y_validation = validation_df[features], validation_df[
        'created_account']
    dvalidation = xgb.DMatrix(X_validation,
                              y_validation,
                              feature_names=features)
    prediction = xgb_model.predict(dvalidation)
    print('Evaluation finished!')
    return f1_score(y_validation,
                    np.where(prediction > args.threshold, 1, 0),
                    pos_label=0)
