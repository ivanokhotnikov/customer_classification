from components import (read_campaign_data, read_exchange_rates,
                        read_mortgage_data)
from xgb_training_pipeline import get_env

env = get_env()


def test_read_campaign_data(capsys):
    df = read_campaign_data(env['data_path'])
    out, err = capsys.readouterr()
    assert err == ''


def test_read_mortgage_data(capsys):
    df = read_mortgage_data(env['data_path'])
    out, err = capsys.readouterr()
    assert err == ''


def test_read_exchange_rates(capsys):
    df = read_exchange_rates(env['data_path'])
    out, err = capsys.readouterr()
    assert err == ''
