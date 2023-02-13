from kfp.v2.dsl import Dataset, Input, Output, component

from components.dependencies import PANDAS, PYTHON38


@component(base_image=PYTHON38, packages_to_install=[PANDAS])
def combine_data(mortgage: Input[Dataset], campaign: Input[Dataset],
                 combined: Output[Dataset]) -> None:
    """Combines the mortgage and campaign data concatenating along the columns and dropping mortgage samples without corresponding campaign data.

    Args:
        mortgage (Input[Dataset]): Mortgage dataset
        campaign (Input[Dataset]): Campaign dataset
        combined (Output[Dataset]): Combined dataset
    """
    import pandas as pd
    mortgage_df = pd.read_csv(mortgage.path + '.csv')
    campaign_df = pd.read_csv(campaign.path + '.csv')
    combined_df = pd.concat([mortgage_df[0:32060], campaign_df], 1)
    combined_df.to_csv(combined.path + '.csv')
