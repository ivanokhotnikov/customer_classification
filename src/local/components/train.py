import xgboost as xgb


def train(args, train_df, features):
    X_train, y_train = train_df[features], train_df['created_account']
    dtrain = xgb.DMatrix(X_train, y_train)
    params = {
        'eta': args.eta,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective,
        'eval_metric': args.eval_metric,
        'grow_policy': args.grow_policy,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha
    }
    xgb_model = xgb.train(params,
                          dtrain,
                          args.boost_rounds,
                          verbose_eval=False)
    print('Training finished!')
    return xgb_model
