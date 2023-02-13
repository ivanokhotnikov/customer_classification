from kfp.v2.dsl import Dataset, Output, component

from components.dependencies import PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS])
def read_campaign_data(data_path: str, campaign: Output[Dataset]) -> None:
    """Read campaign dataset from `data_path`

    Args:
        data_path (str): Data path
        campaign (Output[Dataset]): Campaign dataset in pipeline
    """
    import os

    import pandas as pd

    campaign_df = pd.read_csv(os.path.join(data_path, 'campaign.csv'),
                              index_col=False)
    campaign_df.to_csv(campaign.path + '.csv', index=False)
