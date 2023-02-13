from components import (evaluate, featurize, preprocess, read, split, test,
                        train)
from dotenv import load_dotenv
from utils import parse_args

FEATS = [
    'capital_change', 'Edinburgh', 'marital_status_Married-civ-spouse',
    'religion_Christianity', 'education_num', 'view_nw', 'occupation_level_16',
    'work_experience_length', 'demographic_characteristic', 'yearly_salary',
    'interested_insurance', 'workclass_Private', 'familiarity_nw',
    'hours_per_week', 'age', 'education_Prof-school', 'name_title_Mr.'
]


def main():
    load_dotenv()
    args = parse_args()
    print('Passed arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    interim_df = read()
    processed_df = preprocess(interim_df)
    df = featurize(processed_df)
    train_df, test_df, validation_df = split(args, df)
    xgb_model = train(args, train_df, features=FEATS)
    val_score = evaluate(args, validation_df, xgb_model, features=FEATS)
    test_score = test(args, test_df, xgb_model, features=FEATS)


if __name__ == '__main__':
    main()
