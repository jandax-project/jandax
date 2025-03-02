import pandas as pd
import pytest

from jandax.core import DataFrame


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame with mixed data types."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": ["one", "two", "three", "four", "five"],
            "D": pd.date_range("2023-01-01", periods=5),
        }
    )


@pytest.fixture
def jax_dataframe(sample_dataframe):
    """Create a JaxDataFrame from the sample pandas DataFrame."""
    return DataFrame(sample_dataframe)


@pytest.fixture
def datetime_data():
    """Create a JaxDataFrame with datetime values."""
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({"date": dates, "value": [1, 2, 3, 4, 5]})
    return DataFrame(df)


@pytest.fixture
def categorical_data():
    """Create a JaxDataFrame with categorical values."""
    df = pd.DataFrame(
        {"category": ["red", "green", "blue", "red", "green"], "value": [1, 2, 3, 4, 5]}
    )
    return DataFrame(df)


@pytest.fixture
def numeric_data():
    """Create a JaxDataFrame with numeric values."""
    df = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [0.5, 1.5, 2.5, 3.5, 4.5]})
    return DataFrame(df)


@pytest.fixture
def empty_jaxdf():
    """Create an empty JaxDataFrame."""
    return DataFrame({})
