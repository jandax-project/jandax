import jax
import numpy as np
import pandas as pd
import pytest

import jandax as jdf


@pytest.fixture(scope="module")
def create_dataframe():
    stocks = ["AAPL", "GOOGL", "AMZN", "MSFT", "FB"] * 6
    datetimes = pd.date_range("2020-01-01", periods=30)
    features1 = np.random.randn(30)
    features2 = np.random.randn(30)
    returns = np.random.randn(30)
    df = pd.DataFrame.from_dict(
        {
            "stock": stocks,
            "datetime": datetimes,
            "feature1": features1,
            "feature2": features2,
            "return": returns,
        }
    )

    df = df.set_index(["stock", "datetime"])
    return df


def test_groupby_rank(create_dataframe):
    df = create_dataframe
    jdf_df = jdf.DataFrame(df)

    @jax.jit
    def process_data(df):
        df = (
            df[["feature1", "feature2", "return", "datetime"]]
            .groupby("datetime")
            .transform(jax.scipy.stats.rankdata)
        )
        return df

    jdf_rank = process_data(jdf_df)

    pandas_rank = df[["feature1", "feature2", "return"]].groupby("datetime").rank()

    np.testing.assert_allclose(
        jdf_rank[["feature1", "feature2", "return"]]._values,
        pandas_rank.values,
        rtol=1e-05,
    )
