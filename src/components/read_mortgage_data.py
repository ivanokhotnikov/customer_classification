from kfp.v2.dsl import Dataset, Output, component

from components.dependencies import PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS])
def read_mortgage_data(data_path: str, mortgage: Output[Dataset]) -> None:
    """Read mortgage dataset from `data_path`

    Args:
        data_path (str): Data path
        mortgage (Output[Dataset]): Mortgage dataset in pipeline
    """
    import os

    import pandas as pd

    mortgage_df = pd.read_csv(os.path.join(data_path, 'mortgage.csv'),
                              index_col=False)
    mortgage_df.to_csv(mortgage.path + '.csv', index=False)
