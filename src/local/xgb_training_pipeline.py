import logging
import os
from datetime import datetime

from components import (featurize, fill_nans, preprocess, read, split, test,
                        train, upsample)
from dotenv import load_dotenv
from utils import parse_args

FEATS = [
    'capital_change', 'Edinburgh', 'marital_status_Married-civ-spouse',
    'religion_Christianity', 'education_num', 'view_nw', 'occupation_level_16',
    'work_experience_length', 'demographic_characteristic', 'yearly_salary',
    'interested_insurance', 'workclass_Private', 'familiarity_nw',
    'hours_per_week', 'age', 'education_Prof-school', 'name_title_Mr.'
]
cat_features = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]


def main():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join('logs', os.path.basename(__file__))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, f'{timestamp}.log'),
                        format='%(asctime)s: %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    load_dotenv()
    args = parse_args()
    logging.info('Passed arguments:')
    for k, v in vars(args).items():
        logging.info('\t{}: {}'.format(k, v))
    interim_df = read()
    processed_df = preprocess(interim_df)
    df = featurize(processed_df)
    train_df, test_df = split(args, df, features=FEATS)
    filled_train_df = fill_nans(args, train_df)
    upsampled_train_df = upsample(args,
                                  filled_train_df,
                                  cat_features=cat_features)
    val_score, xgb_model = train(args, upsampled_train_df)
    test_score = test(test_df, xgb_model)
    artifacts_dir = os.path.join('artifacts', timestamp)
    os.makedirs(artifacts_dir, exist_ok=True)
    xgb_model.save_model(os.path.join(artifacts_dir, 'model.json'))
    logging.info('Model saved!')
    with open(os.path.join(artifacts_dir, 'metrics.txt'), 'w') as fp:
        fp.write('Validation score:\n')
        fp.write(str(val_score) + '\n')
        fp.write('Test score:\n')
        fp.write(str(test_score) + '\n')
    logging.info('Metrics saved!')


if __name__ == '__main__':
    main()
