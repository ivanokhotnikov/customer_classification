import os

import pandas as pd


def read():
    campaign_df = pd.read_csv(os.path.join(os.environ['DATA_PATH'],
                                           'campaign.csv'),
                              index_col=False)
    mortgage_df = pd.read_csv(os.path.join(os.environ['DATA_PATH'],
                                           'mortgage.csv'),
                              index_col=False)
    combined_df = pd.concat([mortgage_df[0:len(campaign_df)], campaign_df], 1)
    print('Raw data was read!')
    return combined_df
