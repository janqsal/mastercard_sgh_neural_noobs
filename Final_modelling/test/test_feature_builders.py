import pandas as pd
import numpy as np
import pytest
from src.features.feature_builders import FeatureBuilder

@ pytest.fixture

def simple_df():
    data = {
        'user_id': [1, 1, 1, 2, 2],
        'merchant_id': ['a', 'a', 'b', 'b', 'b'],
        'amount': [10, 20, 30, 40, 50],
        'is_fraud': [0, 1, 0, 1, 0],
        'timestamp': [
            '2025-01-01 10:00',
            '2025-01-02 11:00',
            '2025-01-03 12:00',
            '2025-01-01 09:00',
            '2025-01-02 10:00'
        ]
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def test_rolling_count_sum_mean(simple_df):
    fb = FeatureBuilder(windows=[2])
    result = fb.fit_transform(simple_df)
    res = result.reset_index(drop=True)

    # User 1 (first 3 rows): count_last_2: [nan,1,2], sum_last_2: [nan,10,30], mean_last_2: [nan,10,15]
    assert np.isnan(res.at[0, 'count_last_2'])
    assert res.at[1, 'count_last_2'] == 1
    assert res.at[2, 'count_last_2'] == 2
    assert np.isnan(res.at[0, 'sum_last_2'])
    assert res.at[1, 'sum_last_2'] == 10
    assert res.at[2, 'sum_last_2'] == 30
    assert np.isnan(res.at[0, 'mean_last_2'])
    assert res.at[1, 'mean_last_2'] == 10
    assert res.at[2, 'mean_last_2'] == pytest.approx(15)


def test_merchant_bad_rate_and_user_good_rate(simple_df):
    fb = FeatureBuilder(windows=[1])  # windows not critical for fraud tests
    result = fb.fit_transform(simple_df)
    res = result.reset_index(drop=True)

    # Merchant 'a' (first 2 rows): fraud flags [0,1]
    assert np.isnan(res.at[0, 'merchant_bad_rate'])
    assert res.at[1, 'merchant_bad_rate'] == pytest.approx(0)

    # User good rate for user 1: fraud flags [0,1,0]
    # user_good_rate = 1 - expanding_mean(shifted fraud)
    assert np.isnan(res.at[0, 'user_good_rate'])
    assert res.at[1, 'user_good_rate'] == pytest.approx(1)
    # For third row: expanding mean on [0,1] -> 0.5, so good_rate = 0.5
    assert res.at[2, 'user_good_rate'] == pytest.approx(0.5)
