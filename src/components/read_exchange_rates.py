from kfp.v2.dsl import Dataset, Output, component

from components.dependencies import PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS])
def read_exchange_rates(data_path: str,
                        exchange_rates: Output[Dataset]) -> None:
    """Read exchange rates dataset from `data_path`

    Args:
        data_path (str): Data path
        exchange rates (Output[Dataset]): Exchange rates dataset in pipeline
    """
    import os

    import pandas as pd

    exchange_rates_df = pd.read_csv(os.path.join(data_path,
                                                 'exchange_rates.csv'),
                                    encoding='latin1',
                                    low_memory=False)
    exchange_rates_df.to_csv(exchange_rates.path + '.csv',
                             encoding='latin1',
                             index=False)
