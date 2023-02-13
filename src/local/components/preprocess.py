import pandas as pd
import numpy as np
import os


def preprocess(df):

    def get_salaries(x: str):
        salary_gbp = x.split()[0][1:]
        if 'yearly' in x:
            return pd.Series([salary_gbp, np.nan, np.nan, '£'])
        elif 'month' in x:
            return pd.Series([np.nan, salary_gbp, np.nan, '£'])
        elif 'pw' in x:
            return pd.Series([np.nan, np.nan, salary_gbp, '£'])
        elif 'range' in x:
            top_salary_gbp = x.split()[2]
            return pd.Series([
                str(np.mean((int(salary_gbp), int(top_salary_gbp)))), np.nan,
                np.nan, '£'
            ])
        salary_non_gbp = x.split()[0][:-3]
        currency = x.split()[0][-3:]
        return pd.Series([salary_non_gbp, np.nan, np.nan, currency])

    df[['yearly_salary', 'monthly_salary', 'weekly_salary',
        'currency']] = df.apply(lambda x: get_salaries(x['salary_band']),
                                axis=1,
                                result_type='expand')
    currency_map = {
        'ANG': 0.4,
        'TJS': 0.063,
        'IRR': 0.000017,
        'AFN': 0.0094,
        'CUC': 0.7223,
        'ZWD': 0.001995,
        'TVD': 0.5658,
        'XDR': 1.0404,
        'LTL': 0.2535,
        'SPL': 0.2307,
        'NAD': 0.0496,
        'KPW': 0.0008,
        'NIS': 0.2218,
        'BYR': 0.000027,
        'MRO': 0.0019,
        '£': 1.,
        'GBP': 1.,
        'GIP': 1.,
        'IMP': 1.,
        'GGP': 1.,
        'JEP': 1.,
        'SHP': 1.
    }
    exchange_rates_df = pd.read_csv(os.path.join(os.environ['DATA_PATH'],
                                                 'exchange_rates.csv'),
                                    encoding='latin1',
                                    low_memory=False)
    exchange_rates_df = pd.concat([
        exchange_rates_df,
        pd.DataFrame(
            data={
                'Currency Code': currency_map.keys(),
                'Currency units per £1 ': currency_map.values()
            })
    ])
    df = df.merge(
        exchange_rates_df[['Currency Code', 'Currency units per £1 ']],
        how='left',
        left_on='currency',
        right_on='Currency Code',
    )
    df = df.drop_duplicates(subset=['participant_id'], keep='first')
    df[['yearly_salary', 'monthly_salary', 'weekly_salary'
        ]] = df[['yearly_salary', 'monthly_salary',
                 'weekly_salary']].apply(pd.to_numeric, errors='coerce')
    df['yearly_salary'] = np.where(
        df['yearly_salary'].notnull(),
        df['yearly_salary'],
        np.where(
            df['monthly_salary'].notnull(),
            df['monthly_salary'] * 12,
            df['weekly_salary'] * 52,
        ),
    )
    df = df[df['yearly_salary'].notnull()]
    print('Data was preprocessed!')
    return df
