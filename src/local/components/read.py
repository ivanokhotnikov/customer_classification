import os
import logging
import pandas as pd


def read():
    """
    The read function reads in the mortgage and campaign datasets, concatenates
    them into a single dataframe, and returns it.

    Returns:
        The combined dataframe
    """
    campaign_df = pd.read_csv(os.path.join(os.environ['DATA_PATH'],
                                           'campaign.csv'),
                              index_col=False)
    mortgage_df = pd.read_csv(os.path.join(os.environ['DATA_PATH'],
                                           'mortgage.csv'),
                              index_col=False)
    combined_df = pd.concat([mortgage_df[0:len(campaign_df)], campaign_df],
                            axis=1)
    logging.info('Raw data read!')
    logging.info('\tcampaign shape: {}'.format(campaign_df.shape))
    logging.info('\tmortgage shape: {}'.format(mortgage_df.shape))
    logging.info('\tcombined shape: {}\n'.format(combined_df.shape))
    return combined_df
