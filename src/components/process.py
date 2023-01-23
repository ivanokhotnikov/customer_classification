from kfp.v2.dsl import Dataset, Input, Output, component

from components.dependencies import NUMPY, PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS, NUMPY])
def process(combined: Input[Dataset], exchange_rates: Input[Dataset],
            interim: Output[Dataset]) -> None:
    """Runs data processing
    ### Preprocessing
    # This is our preprocessing step we do the following:
    # 1. Create periodic salaries (yearly, monthly, weekly)
    # 2. Apply exchange rate using available .csv
    # 3. Convert weekly and monthly salaries to yearly
    # 4. Exclude customers without a valid salary
    # 5. Exclude missing values from test and train sets

    Args:
        combined (Input[Dataset]): Combined dataset (campaign and mortgage)
        exchange_rates (Input[Dataset]): Exchange rates dataset
        interim (Output[Dataset]): Interim dataset

    Returns:
        _type_: _description_
    """
    import logging

    import numpy as np
    import pandas as pd

    logging.info('Preprocessing the datasets')
    logging.info('Creating periodic salary')

    combined_df = pd.read_csv(combined.path + '.csv')

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

    combined_df[[
        'yearly_salary', 'monthly_salary', 'weekly_salary', 'currency'
    ]] = combined_df.apply(lambda x: get_salaries(x['salary_band']),
                           axis=1,
                           result_type='expand')
    # Apply exchange rate
    logging.info('Applying exchange rate')
    exchange_rates_df = pd.read_csv(exchange_rates.path + '.csv',
                                    encoding='latin1',
                                    low_memory=False)
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
        'SHP': 1.,
    }
    exchange_rates_df = pd.concat([
        exchange_rates_df,
        pd.DataFrame(
            data={
                'Currency Code': currency_map.keys(),
                'Currency units per £1 ': currency_map.values()
            })
    ])
    combined_df = combined_df.merge(
        exchange_rates_df[['Currency Code', 'Currency units per £1 ']],
        how='left',
        left_on='currency',
        right_on='Currency Code',
    )
    combined_df = combined_df.drop_duplicates(subset=['participant_id'],
                                              keep='first')
    # Normalise salaries
    combined_df[['yearly_salary', 'monthly_salary',
                 'weekly_salary']] = combined_df[[
                     'yearly_salary', 'monthly_salary', 'weekly_salary'
                 ]].apply(pd.to_numeric, errors='coerce')

    logging.info('Converting weekly and monthly salary to yearly')
    combined_df['yearly_salary'] = np.where(
        combined_df['yearly_salary'].notnull(),
        combined_df['yearly_salary'],
        np.where(
            combined_df['monthly_salary'].notnull(),
            combined_df['monthly_salary'] * 12,
            combined_df['weekly_salary'] * 52,
        ),
    )

    logging.info('Excluding customers who do not have valid salary')
    combined_df = combined_df[combined_df['yearly_salary'].notnull()]
    combined_df.to_csv(interim.path + '.csv', index=False)
