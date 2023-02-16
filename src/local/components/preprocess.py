import logging
import os

import pandas as pd


def preprocess(df):
    """
    The preprocess function takes a dataframe as input and returns a new dataframe with the following changes:
        1. The salary_band column is split into three columns: yearly_salary, monthly_salary, weekly_salary
        2. The currency column is converted to GBP using exchange rates from the file exchange_rates.csv

    Args:
        df: Pass the dataframe to be preprocessed

    Returns:
        Interim dataframe
    """
    df['yearly_salary'] = df.loc[df['salary_band'].str.contains('yearly'),
                                 'salary_band'].str.removeprefix(
                                     '£').str.removesuffix(' yearly').astype(
                                         float)
    df['monthly_salary'] = df.loc[df['salary_band'].str.contains('per month'),
                                  'salary_band'].str.removeprefix(
                                      '£').str.removesuffix(
                                          ' per month').astype(float)
    df['weekly_salary'] = df.loc[df['salary_band'].str.contains('pw'),
                                 'salary_band'].str.removeprefix(
                                     '£').str.removesuffix(' pw').astype(float)
    df['range_start'] = df.loc[
        df['salary_band'].str.contains('range'),
        'salary_band'].str.removeprefix('£').str.removesuffix(
            ' range').str.replace(' - ',
                                  ' ').str.split().str.get(0).astype(float)
    df['range_end'] = df.loc[df['salary_band'].str.contains('range'),
                             'salary_band'].str.removeprefix(
                                 '£').str.removesuffix(' range').str.replace(
                                     ' - ',
                                     ' ').str.split().str.get(1).astype(float)
    df.loc[df['salary_band'].str.contains('range'),
           'yearly_salary'] = df.loc[df['salary_band'].str.contains('range'),
                                     ['range_start', 'range_end']].mean(axis=1)
    df['currency'] = df.loc[df['salary_band'].str.contains('£'),
                            'salary_band'].str.get(0)
    df.loc[df['salary_band'].str.contains('GBP'), 'currency'] = '£'

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
    df.loc[(df['yearly_salary'].isna()) & (df['monthly_salary'].notnull()),
           'yearly_salary'] = df['monthly_salary'] * 12
    df.loc[(df['yearly_salary'].isna()) & (df['monthly_salary'].isna()) &
           (df['weekly_salary'].notnull()),
           'yearly_salary'] = df['weekly_salary'] * 52
    df = df[df['yearly_salary'].notnull()]
    logging.info('Data preprocessed!\n')
    return df
