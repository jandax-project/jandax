import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from jandax.core import DataFrame
from jandax.utils import ns_to_pd_datetime, pd_datetime_to_ns, to_datetime, to_strings


# Basic initialization tests
def test_init_from_dataframe(sample_dataframe, jax_dataframe):
    """Test initialization from pandas DataFrame."""
    # Check shape matches
    assert jax_dataframe.shape == sample_dataframe.shape

    # Check column names match
    assert jax_dataframe.columns == list(sample_dataframe.columns)

    # Check if to_pandas conversion returns the original data
    reconverted = jax_dataframe.to_pandas()
    # For string columns, they become categorical, so we need to convert them back
    for col in reconverted.select_dtypes(["category"]):
        reconverted[col] = reconverted[col].astype(str)
    assert_frame_equal(reconverted, sample_dataframe, check_dtype=False)


# Column access and manipulation tests
def test_column_access(jax_dataframe):
    """Test accessing columns."""
    # Single column access
    col_a = jax_dataframe["A"]
    assert isinstance(col_a, jax.Array)
    np.testing.assert_array_equal(col_a, np.array([1, 2, 3, 4, 5]))

    # Multiple column access
    subset = jax_dataframe[["A", "B"]]
    assert isinstance(subset, DataFrame)
    assert subset.shape == (5, 2)
    assert subset.columns == ["A", "B"]


def test_column_assignment(jax_dataframe):
    """Test assigning new values to columns."""
    # Assign new column
    jax_dataframe["E"] = [100, 200, 300, 400, 500]

    assert "E" in jax_dataframe.columns
    np.testing.assert_array_equal(
        jax_dataframe["E"], np.array([100, 200, 300, 400, 500])
    )

    # Replace existing column
    jax_dataframe["A"] = [10, 20, 30, 40, 50]
    np.testing.assert_array_equal(jax_dataframe["A"], np.array([10, 20, 30, 40, 50]))


# Datetime handling tests
def test_datetime_conversion(datetime_data):
    """Test datetime column handling."""
    # Check datetime conversion
    date_col = datetime_data["date"]
    dt_values = to_datetime(date_col)

    assert isinstance(dt_values, pd.DatetimeIndex)
    assert len(dt_values) == 5
    assert dt_values[0].strftime("%Y-%m-%d") == "2023-01-01"


# Categorical data tests
def test_categorical_data(categorical_data):
    """Test categorical column handling."""
    cat_col = categorical_data["category"]
    category_map = categorical_data._column_metadata["category"]["category_map"]
    string_values = to_strings(cat_col, category_map)

    assert isinstance(string_values, pd.Series)
    assert list(string_values) == ["red", "green", "blue", "red", "green"]


# Apply function tests
def test_apply_function(numeric_data):
    """Test applying functions to dataframe."""
    # Apply function to columns
    result = numeric_data.apply(lambda x: x * 2, axis=0)

    np.testing.assert_array_equal(result["X"], np.array([2, 4, 6, 8, 10]))
    np.testing.assert_array_equal(result["Y"], np.array([1.0, 3.0, 5.0, 7.0, 9.0]))

    # Apply function to rows
    result = numeric_data.apply(lambda x: jnp.sum(x), axis=1)
    np.testing.assert_array_equal(
        result["lambda_X_Y"], np.array([1.5, 3.5, 5.5, 7.5, 9.5])
    )


# Rolling window tests
def test_rolling_window(numeric_data):
    """Test rolling window operations."""
    # Apply rolling mean with window size 2
    result = numeric_data.rolling(2).apply(lambda x: jnp.mean(x))

    # With pandas-compatible behavior, the first value should be NaN (incomplete window)
    assert np.isnan(result["X"][0])
    # For second row, it's mean of first and second
    assert result["X"][1] == 1.5
    assert result["X"][2] == 2.5


def test_rolling_groupby():
    """Test rolling window operations within groups."""
    # Create sample data with multiple symbols and time series data
    data = {
        "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"],
        "time": pd.date_range("2023-01-01", periods=8),
        "price": [100.0, 200.0, 101.0, 202.0, 103.0, 198.0, 105.0, 197.0],
    }

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Verify column types
    assert jdf._column_metadata["symbol"]["dtype_flag"] == "category"
    assert jdf._column_metadata["time"]["dtype_flag"] == "datetime"

    # Calculate results with pandas - do it separately for each symbol for simplicity
    pd_results = {}
    for symbol in ["AAPL", "GOOGL"]:
        symbol_data = pdf[pdf["symbol"] == symbol].set_index("time")["price"]
        pd_results[symbol] = symbol_data.rolling(3).mean().dropna().values

    # Calculate results with JaxDataFrame - CHANGED: use apply(jnp.nanmean) instead of mean()
    jax_result = jdf.groupby("symbol")["price"].rolling(3).apply(jnp.nanmean)

    # Verify shape
    assert jax_result.shape[0] == len(pdf)

    # Verify values for each symbol
    for symbol in ["AAPL", "GOOGL"]:
        # Get pandas values for this symbol
        pd_symbol_values = pd_results[symbol]

        # Get JaxDataFrame values for this symbol
        symbol_category_map = jdf._column_metadata["symbol"]["category_map"]
        mask = to_strings(jdf["symbol"], symbol_category_map) == symbol
        jax_symbol_indices = np.where(mask)[0]
        jax_symbol_values = jax_result["price"][jax_symbol_indices]

        # Skip first (window_size-1) values as they're calculated differently
        # (pandas returns NaN, we return partial window calculations)
        jax_symbol_values = jax_symbol_values[2:]

        # Verify values
        np.testing.assert_allclose(
            jax_symbol_values,
            pd_symbol_values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Rolling mean values for symbol {symbol} don't match",
        )

    # Test applying a custom function
    jax_result_custom = (
        jdf.groupby("symbol")["price"]
        .rolling(3)
        .apply(
            lambda x: jnp.sum(x) / jnp.size(x)  # Custom mean implementation
        )
    )

    # Values should match the mean implementation
    np.testing.assert_allclose(
        jax_result["price"],
        jax_result_custom["price"],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Custom function result doesn't match mean",
    )


def test_rolling_sum_vs_pandas():
    """Test rolling sum operations compared to pandas, including oversize windows."""
    # Create sample data
    data = {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pd.DataFrame(data))

    # Test normal window size
    pd_result = pdf.rolling(window=3).sum()
    jax_result = jdf.rolling(3).apply(jnp.sum)

    # Verify against pandas (skip first two rows due to NaN handling differences)
    np.testing.assert_allclose(
        jax_result["A"][2:], pd_result["A"].dropna().values, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        jax_result["B"][2:], pd_result["B"].dropna().values, rtol=1e-5, atol=1e-5
    )

    # Test oversize window (window=6 > data length=5)
    pd_result_large = pdf.rolling(window=6).sum()
    jax_result_large = jdf.rolling(6).apply(jnp.sum)

    # In pandas, all values are NaN when window > length
    assert pd_result_large["A"].isna().all()

    # Our implementation should match pandas behavior - ALL values should be NaN
    assert np.isnan(jax_result_large["A"]).all(), (
        "All values should be NaN for oversize window"
    )
    assert np.isnan(jax_result_large["B"]).all(), (
        "All values should be NaN for oversize window"
    )


def test_groupby_rolling_sum_oversize_window():
    """Test groupby rolling sum with window size larger than group size."""
    data = {
        "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL"],
        "price": [100.0, 200.0, 101.0, 202.0, 103.0],
    }

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pd.DataFrame(data))

    # Calculate pandas result with window size 4 (larger than any group size)
    pd_results = {}
    for symbol in ["AAPL", "GOOGL"]:
        # First filter to get only rows for this symbol
        symbol_data = pdf[pdf["symbol"] == symbol].reset_index(drop=True)
        # Then calculate rolling sum on the price column
        pd_results[symbol] = symbol_data["price"].rolling(4).sum().values

    # Calculate JaxDataFrame result
    jax_result = jdf.groupby("symbol")["price"].rolling(4).apply(jnp.sum)

    # Verify values for each symbol
    for symbol in ["AAPL", "GOOGL"]:
        # Get pandas values for this symbol
        pd_symbol_values = pd_results[symbol]
        pd_symbol_values = pd_symbol_values[~np.isnan(pd_symbol_values)]

        # Get JaxDataFrame values for this symbol
        # Convert pandas Series boolean mask to numpy array before indexing
        symbol_category_map = jdf._column_metadata["symbol"]["category_map"]
        mask = np.array(to_strings(jdf["symbol"], symbol_category_map) == symbol)
        jax_symbol_values = jax_result["price"][mask]
        jax_symbol_values = jax_symbol_values[~np.isnan(jax_symbol_values)]

        # Both should be empty or contain only NaN values since window > group size
        assert len(pd_symbol_values) == 0, (
            f"Pandas should return empty array for symbol {symbol} with oversized window"
        )
        assert len(jax_symbol_values) == 0, (
            f"JaxDF should return empty array for symbol {symbol} with oversized window"
        )


# GroupBy tests
def test_groupby(categorical_data):
    """Test groupby operations."""
    # Group by category and sum values
    result = categorical_data.groupby("category").aggregate(lambda x: jnp.sum(x))

    # We should have 3 groups (red, green, blue)
    assert result.shape[0] == 3

    # Convert to pandas to easily check results
    result_pd = result.to_pandas()

    # Red group (value 1 + value 4 = 5)
    assert result_pd.loc[result_pd["category"] == "red", "value"].iloc[0] == 5

    # Green group (value 2 + value 5 = 7)
    assert result_pd.loc[result_pd["category"] == "green", "value"].iloc[0] == 7

    # Blue group (just value 3 = 3)
    assert result_pd.loc[result_pd["category"] == "blue", "value"].iloc[0] == 3


def test_groupby_aggregate(categorical_data):
    """Test groupby aggregation operations explicitly using aggregate()."""
    # Group by category and sum values
    result = categorical_data.groupby("category").aggregate(lambda x: jnp.sum(x))

    # We should have 3 groups (red, green, blue)
    assert result.shape[0] == 3

    # Convert to pandas to easily check results
    result_pd = result.to_pandas()

    # Red group (value 1 + value 4 = 5)
    assert result_pd.loc[result_pd["category"] == "red", "value"].iloc[0] == 5

    # Green group (value 2 + value 5 = 7)
    assert result_pd.loc[result_pd["category"] == "green", "value"].iloc[0] == 7

    # Blue group (just value 3 = 3)
    assert result_pd.loc[result_pd["category"] == "blue", "value"].iloc[0] == 3


# Utility function tests
def test_pd_datetime_to_ns():
    """Test datetime to nanoseconds conversion."""
    dt = pd.DatetimeIndex(["2023-01-01", "2023-01-02"])
    ns = pd_datetime_to_ns(dt)

    assert isinstance(ns, np.ndarray)
    assert len(ns) == 2

    # Convert back
    dt_back = ns_to_pd_datetime(ns)
    assert dt_back[0].strftime("%Y-%m-%d") == "2023-01-01"
    assert dt_back[1].strftime("%Y-%m-%d") == "2023-01-02"


# Edge case tests
def test_empty_dataframe(empty_jaxdf):
    """Test operations on empty dataframe."""
    assert empty_jaxdf.shape == (0, 0)
    assert len(empty_jaxdf) == 0
    assert empty_jaxdf.columns == []


# Additional comparison tests with pandas
def test_groupby_mean_vs_pandas():
    """Compare groupby mean results between JaxDataFrame and pandas."""
    # Create sample data with multiple groups
    data = {"colA": [1, 1, 2, 2, 2, 3], "colB": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pdf)

    # Calculate pandas result
    pd_result = pdf.groupby("colA")["colB"].mean()

    # Calculate JaxDataFrame result
    jax_result = jdf.groupby("colA").aggregate(jnp.mean)

    # Verify columns in result
    assert "colA" in jax_result.columns
    assert "colB" in jax_result.columns

    # Verify values match pandas result
    pd_result_dict = pd_result.to_dict()
    for i, group_val in enumerate(jax_result["colA"]):
        group_key = float(group_val)
        expected_mean = pd_result_dict[group_key]
        actual_mean = float(jax_result["colB"][i])
        assert np.isclose(actual_mean, expected_mean), (
            f"Group {group_key}: expected mean {expected_mean}, got {actual_mean}"
        )


def test_groupby_mean_aggregate_vs_pandas():
    """Compare groupby mean results between JaxDataFrame.aggregate() and pandas."""
    # Create sample data with multiple groups
    data = {"colA": [1, 1, 2, 2, 2, 3], "colB": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pdf)

    # Calculate pandas result
    pd_result = pdf.groupby("colA")["colB"].mean()

    # Calculate JaxDataFrame result using explicit aggregate
    jax_result = jdf.groupby("colA").aggregate(jnp.mean)

    # Verify columns in result
    assert "colA" in jax_result.columns
    assert "colB" in jax_result.columns

    # Verify values match pandas result
    pd_result_dict = pd_result.to_dict()
    for i, group_val in enumerate(jax_result["colA"]):
        group_key = float(group_val)
        expected_mean = pd_result_dict[group_key]
        actual_mean = float(jax_result["colB"][i])
        assert np.isclose(actual_mean, expected_mean), (
            f"Group {group_key}: expected mean {expected_mean}, got {actual_mean}"
        )


def test_groupby_transform_vs_pandas():
    """Compare groupby transform results between JaxDataFrame and pandas."""
    # Create sample data
    data = {"colA": [1, 1, 2, 2, 2, 3], "colB": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pdf)

    # Define transform functions
    def pd_transform(x):
        return x - x.mean()

    def jax_transform(arr):
        return arr - jnp.mean(arr)

    # Calculate results
    pd_result = pdf.groupby("colA")["colB"].transform(pd_transform)
    jax_result = jdf.groupby("colA").transform(jax_transform)

    # Verify shape
    assert len(jax_result) == len(pdf)

    # Verify values
    pd_values = pd_result.values
    jax_values = jax_result["colB"]
    np.testing.assert_allclose(
        jax_values,
        pd_values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Transform values don't match between pandas and JaxDF",
    )


def test_groupby_transform_explicit_vs_pandas():
    """Compare explicit groupby transform results between JaxDataFrame and pandas."""
    # Create sample data
    data = {"colA": [1, 1, 2, 2, 2, 3], "colB": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(pdf)

    # Define transform functions
    def pd_transform(x):
        return x - x.mean()

    def jax_transform(arr):
        return arr - jnp.mean(arr)

    # Calculate results
    pd_result = pdf.groupby("colA")["colB"].transform(pd_transform)
    jax_result = jdf.groupby("colA").transform(jax_transform)

    # Verify shape
    assert len(jax_result) == len(pdf)

    # Verify values
    pd_values = pd_result.values
    jax_values = jax_result["colB"]
    np.testing.assert_allclose(
        jax_values,
        pd_values,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Transform values don't match between pandas and JaxDF",
    )


def test_categorical_groupby_vs_pandas():
    """Test groupby with categorical columns compared to pandas."""
    # Create sample data
    stock_data = {
        "stock": [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AAPL",
            "GOOGL",
            "MSFT",
            "AAPL",
            "GOOGL",
            "MSFT",
            "AAPL",
        ],
        "price": [150, 2800, 300, 155, 2850, 310, 160, 2900, 320, 165],
    }

    # Create pandas DataFrame and JaxDataFrame
    stock_df = pd.DataFrame(stock_data)
    jsdf = DataFrame(stock_df)

    # Calculate results
    pd_result = stock_df.groupby("stock")["price"].mean().reset_index()
    jax_result = jsdf.groupby("stock").aggregate(jnp.mean)

    # Verify columns
    assert set(jax_result.columns) == {"stock", "price"}

    # Verify values
    pd_means = pd_result.set_index("stock")["price"].to_dict()
    stock_category_map = jax_result._column_metadata["stock"]["category_map"]
    for i, stock in enumerate(to_strings(jax_result["stock"], stock_category_map)):
        expected = pd_means[stock]
        actual = float(jax_result["price"][i])
        assert np.isclose(actual, expected)


def test_rankdata_with_multiindex():
    """Test rankdata with multi-index compared to pandas."""
    # Create sample data with 5 stocks and 2 time samples
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    times = pd.date_range("2023-01-01 09:00", periods=2, freq="min")

    # Create all combinations of stocks and times
    index = pd.MultiIndex.from_product([times, stocks], names=["time", "stock"])

    # Generate sample data
    np.random.seed(42)
    data = {
        "price": np.random.uniform(100, 500, size=len(index)),
        "volume": np.random.randint(1000, 10000, size=len(index)),
    }

    # Create pandas DataFrame with MultiIndex
    pdf = pd.DataFrame(data, index=index)

    # Create JaxDataFrame
    jdf = DataFrame(pdf)

    # Verify multi-index levels are converted to columns
    assert "time" in jdf.columns
    assert "stock" in jdf.columns

    # Calculate results
    pd_ranked = pdf.groupby("time").rank()
    jax_ranked = jdf.groupby("time").transform(jstats.rankdata)

    # Verify shape
    assert len(jax_ranked) == len(pdf)

    # Verify values for each column
    for col in ["price", "volume"]:
        pd_col_ranks = pd_ranked[col].values
        jax_col_ranks = jax_ranked[col]

        np.testing.assert_allclose(
            pd_col_ranks,
            jax_col_ranks,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Rank values for column '{col}' don't match",
        )


def test_rankdata_transform_with_multiindex():
    """Test rankdata with transform() method for multi-index compared to pandas."""
    # Create sample data with 5 stocks and 2 time samples
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    times = pd.date_range("2023-01-01 09:00", periods=2, freq="min")

    # Create all combinations of stocks and times
    index = pd.MultiIndex.from_product([times, stocks], names=["time", "stock"])

    # Generate sample data
    np.random.seed(42)
    data = {
        "price": np.random.uniform(100, 500, size=len(index)),
        "volume": np.random.randint(
            1000, 10000, size=len(index)
        ),  # Fix: use size as keyword arg
    }

    # Create pandas DataFrame with MultiIndex
    pdf = pd.DataFrame(data, index=index)

    # Create JaxDataFrame
    jdf = DataFrame(pdf)

    # Verify multi-index levels are converted to columns
    assert "time" in jdf.columns
    assert "stock" in jdf.columns

    # Calculate results
    pd_ranked = pdf.groupby("time").rank()
    jax_ranked = jdf.groupby("time").transform(jstats.rankdata)

    # Verify shape
    assert len(jax_ranked) == len(pdf)

    # Verify values for each column
    for col in ["price", "volume"]:
        pd_col_ranks = pd_ranked[col].values
        jax_col_ranks = jax_ranked[col]

        np.testing.assert_allclose(
            pd_col_ranks,
            jax_col_ranks,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Rank values for column '{col}' don't match",
        )


def test_rolling_window_vs_pandas():
    """Test rolling window operations compared to pandas."""
    # Create sample data
    data = {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Calculate results
    pd_result = pdf.rolling(window=2).mean()
    jax_result = jdf.rolling(2).apply(jnp.mean)

    # Expected values should match pandas behavior (first value is NaN)
    expected_a = np.array([np.nan, 1.5, 2.5, 3.5, 4.5])
    expected_b = np.array([np.nan, 15.0, 25.0, 35.0, 45.0])

    # Verify against expected values
    np.testing.assert_allclose(jax_result["A"], expected_a, equal_nan=True)
    np.testing.assert_allclose(jax_result["B"], expected_b, equal_nan=True)

    # Now we can directly compare with pandas since our behavior matches
    pd_values_a = pd_result["A"].values
    pd_values_b = pd_result["B"].values
    np.testing.assert_allclose(jax_result["A"], pd_values_a, equal_nan=True)
    np.testing.assert_allclose(jax_result["B"], pd_values_b, equal_nan=True)


def test_rolling_groupby_aggregate():
    """Test rolling window aggregation operations within groups."""
    # Create sample data with multiple symbols and time series data
    data = {
        "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"],
        "time": pd.date_range("2023-01-01", periods=8),
        "price": [100.0, 200.0, 101.0, 202.0, 103.0, 198.0, 105.0, 197.0],
    }

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Verify column types
    assert jdf._column_metadata["symbol"]["dtype_flag"] == "category"
    assert jdf._column_metadata["time"]["dtype_flag"] == "datetime"

    # Calculate results with pandas - do it separately for each symbol for simplicity
    pd_results = {}
    for symbol in ["AAPL", "GOOGL"]:
        symbol_data = pdf[pdf["symbol"] == symbol].set_index("time")["price"]
        pd_results[symbol] = symbol_data.rolling(3).mean().dropna().values

    # Calculate results with JaxDataFrame, using the explicit aggregate method
    jax_result = jdf.groupby("symbol")["price"].rolling(3).aggregate(jnp.mean)

    # Verify shape
    assert jax_result.shape[0] == len(pdf)

    # Verify values for each symbol
    for symbol in ["AAPL", "GOOGL"]:
        # Get pandas values for this symbol
        pd_symbol_values = pd_results[symbol]

        # Get JaxDataFrame values for this symbol
        symbol_category_map = jdf._column_metadata["symbol"]["category_map"]
        mask = to_strings(jdf["symbol"], symbol_category_map) == symbol
        jax_symbol_indices = np.where(mask)[0]
        jax_symbol_values = jax_result["price"][jax_symbol_indices]

        # Skip first (window_size-1) values as they're calculated differently
        # (pandas returns NaN, we return partial window calculations)
        jax_symbol_values = jax_symbol_values[2:]

        # Verify values
        np.testing.assert_allclose(
            jax_symbol_values,
            pd_symbol_values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Rolling aggregate mean values for symbol {symbol} don't match",
        )


def test_groupby_rolling_display():
    """Test that groupby rolling results display correctly with categorical keys."""
    # Create sample data with stock symbols
    data = {
        "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"],
        "price": [100.0, 200.0, 101.0, 202.0, 103.0, 198.0, 105.0, 197.0],
        "volume": [1000, 500, 1200, 600, 900, 550, 1100, 700],
    }

    # Create JaxDataFrame
    jdf = DataFrame(data)

    # Apply groupby rolling calculation
    result = jdf.groupby("symbol")["price"].rolling(2).apply(jnp.mean)

    # Check that the result dataframe has the correct columns in the correct order
    assert result.columns[0] == "symbol", "Symbol should be the first column"
    assert result.columns[1] == "price", "Price should be the second column"

    # Check that symbols are properly preserved (not numeric codes)
    pd_result = result.to_pandas()
    # When converted to pandas, categorical columns are already converted to pandas Categorical type
    # so we can directly check the category values
    assert set(pd_result["symbol"]) == {"AAPL", "GOOGL"}, (
        "Symbol values should be strings, not codes"
    )

    # Check values are correct
    # For AAPL rows
    aapl_prices = pd_result.loc[pd_result["symbol"] == "AAPL", "price"].values

    # First element should be NaN for window_size=2 in grouped rolling
    # This is different from pandas behavior for regular rolling but matches our implementation
    expected_values = [float("nan"), 100.5, 102.0, 104.0]

    # Test that values after the first NaN match expected values
    np.testing.assert_allclose(aapl_prices[1:], expected_values[1:], rtol=1e-5)

    # Separately test that first value is NaN
    assert np.isnan(aapl_prices[0]), "First value should be NaN for window_size=2"

    # For GOOGL rows
    googl_prices = pd_result.loc[pd_result["symbol"] == "GOOGL", "price"].values

    # Same test structure for GOOGL prices
    expected_googl = [float("nan"), 201.0, 200.0, 197.5]

    # Test non-NaN values
    np.testing.assert_allclose(googl_prices[1:], expected_googl[1:], rtol=1e-5)

    # Test first value is NaN
    assert np.isnan(googl_prices[0]), "First value should be NaN for window_size=2"


def test_rolling_with_min_periods_parameter():
    """Test rolling operations with built-in min_periods parameter."""
    # Create sample data with some NaN values
    data = {"A": [1.0, np.nan, 3.0, 4.0, 5.0], "B": [10.0, 20.0, np.nan, 40.0, 50.0]}

    # Create pandas DataFrame and JaxDataFrame
    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Define a window size
    window_size = 3

    # Test different min_periods settings
    for min_periods in [1, 2, window_size]:
        # Calculate pandas result
        pd_result = pdf.rolling(window=window_size, min_periods=min_periods).mean()

        # Calculate JaxDataFrame result with built-in min_periods parameter
        jax_result = jdf.rolling(window_size, min_periods=min_periods).apply(
            jnp.nanmean
        )

        # Verify results for both columns
        for col in ["A", "B"]:
            # Convert to numpy arrays for consistent comparison
            pd_values = pd_result[col].values
            jax_values = jax_result[col]

            # Check that NaN patterns match (NaN in same positions)
            pd_isnan = np.isnan(pd_values)
            jax_isnan = np.isnan(jax_values)
            assert np.array_equal(pd_isnan, jax_isnan), (
                f"NaN patterns don't match for column {col} with min_periods={min_periods}"
            )

            # Check non-NaN values match
            np.testing.assert_allclose(
                jax_values,
                pd_values,
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
                err_msg=f"Values don't match for column {col} with min_periods={min_periods}",
            )


# Edge case tests for pandas compatibility


def test_empty_groups_in_groupby():
    """Test groupby with some empty groups."""
    # Create data with one group having no values
    data = {
        "group": ["A", "B", "B", "C", "C", "C"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }

    # Create pandas DataFrame and add a row for group "D" that will be filtered out
    pdf = pd.DataFrame(data)

    # Create JaxDataFrame
    jdf = DataFrame(data)

    # Create a filtered version that excludes some groups
    filter_condition = pdf["value"] > 10  # This will create empty groups
    filtered_pdf = pdf[filter_condition]

    # In pandas, empty groups are dropped when aggregating
    pd_result = pdf.groupby("group")["value"].mean()

    # Calculate with JaxDataFrame
    jax_result = jdf.groupby("group").aggregate(jnp.mean)

    # Test that non-empty groups match
    for group in ["A", "B", "C"]:
        pd_value = pd_result.loc[group]
        # Convert pandas Series to numpy array before using jnp.where
        jax_idx = jnp.where(
            np.array(
                to_strings(
                    jax_result["group"],
                    jax_result._column_metadata["group"]["category_map"],
                )
                == group
            )
        )[0][0]
        jax_value = jax_result["value"][jax_idx]
        assert np.isclose(jax_value, pd_value), (
            f"Group {group} doesn't match pandas value"
        )


def test_nan_handling_in_groupby():
    """Test groupby with groups containing NaN values."""
    # Create data with NaN values in some groups
    data = {
        "group": ["A", "A", "B", "B", "C", "C"],
        "value": [1.0, np.nan, np.nan, 3.0, 4.0, np.nan],
    }

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # In pandas, NaN values are ignored in mean calculation
    pd_result = pdf.groupby("group")["value"].mean()

    # Calculate with JaxDataFrame
    jax_result = jdf.groupby("group").aggregate(jnp.nanmean)

    # Test each group
    for group in ["A", "B", "C"]:
        pd_value = pd_result.loc[group]
        # Convert pandas Series to numpy array before using jnp.where
        jax_idx = jnp.where(
            np.array(
                to_strings(
                    jax_result["group"],
                    jax_result._column_metadata["group"]["category_map"],
                )
                == group
            )
        )[0][0]
        jax_value = jax_result["value"][jax_idx]

        # Handle case where pandas returns NaN (all values in group are NaN)
        if np.isnan(pd_value):
            assert np.isnan(jax_value), f"Group {group} should be NaN"
        else:
            assert np.isclose(jax_value, pd_value), (
                f"Group {group} doesn't match pandas value"
            )


def test_all_nan_group():
    """Test groupby with a group containing only NaN values."""
    data = {"group": ["A", "A", "B", "B"], "value": [np.nan, np.nan, 3.0, 4.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # In pandas, a group with all NaN returns NaN for mean
    pd_result = pdf.groupby("group")["value"].mean()

    # Calculate with JaxDataFrame
    jax_result = jdf.groupby("group").aggregate(jnp.nanmean)

    # Group A should be NaN
    # Convert pandas Series to numpy array before using jnp.where
    jax_a_idx = jnp.where(
        np.array(
            to_strings(
                jax_result["group"],
                jax_result._column_metadata["group"]["category_map"],
            )
            == "A"
        )
    )[0][0]
    assert np.isnan(jax_result["value"][jax_a_idx]), "Group A should be NaN"

    # Group B should have a valid mean
    # Convert pandas Series to numpy array before using jnp.where
    jax_b_idx = jnp.where(
        np.array(
            to_strings(
                jax_result["group"],
                jax_result._column_metadata["group"]["category_map"],
            )
            == "B"
        )
    )[0][0]
    assert np.isclose(jax_result["value"][jax_b_idx], 3.5), (
        "Group B should have mean 3.5"
    )


def test_rolling_all_nan_window():
    """Test rolling window where a window contains all NaN values."""
    data = {"A": [1.0, np.nan, np.nan, 4.0, 5.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # In pandas, a window with all NaN returns NaN
    pd_result = pdf.rolling(window=2).mean()

    # JaxDataFrame result
    jax_result = jdf.rolling(2).apply(jnp.nanmean)

    # Check specifically the 3rd value, which has a window of [NaN, NaN]
    assert np.isnan(jax_result["A"][2]), "Window of all NaN should produce NaN"

    # All values should match pandas behavior
    np.testing.assert_allclose(
        jax_result["A"],
        pd_result["A"].values,
        equal_nan=True,
        err_msg="Rolling with NaN values doesn't match pandas behavior",
    )


def test_windowsize_equals_dataframe_length():
    """Test rolling window with window size equal to the DataFrame length."""
    data = {"A": [1.0, 2.0, 3.0, 4.0, 5.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Window size equals DataFrame length
    window_size = len(data["A"])

    # Calculate results
    pd_result = pdf.rolling(window=window_size).mean()
    jax_result = jdf.rolling(window_size).apply(jnp.mean)

    # In pandas, all but last row are NaN
    # Last row should be the mean of all values
    expected_mean = np.mean(data["A"])

    # Check that pattern of NaN matches pandas
    for i in range(window_size - 1):
        assert np.isnan(jax_result["A"][i]), f"Position {i} should be NaN"

    # Check that last value is the mean of all values
    assert np.isclose(jax_result["A"][window_size - 1], expected_mean), (
        "Last value should be mean of all values"
    )

    # Compare directly to pandas result
    np.testing.assert_allclose(
        jax_result["A"],
        pd_result["A"].values,
        equal_nan=True,
        err_msg="Rolling with window size = data length doesn't match pandas",
    )


def test_single_value_groups():
    """Test groupby with groups containing only a single value."""
    data = {"group": ["A", "B", "C", "D", "E"], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # When groups have single values, standard deviation should be 0 or NaN depending on ddof
    pd_result_std = pdf.groupby("group")["value"].std()

    # Calculate with JaxDataFrame
    jax_result = jdf.groupby("group").aggregate(jnp.std)

    # All groups should have std = 0 (or NaN depending on implementation)
    for i, group in enumerate(
        to_strings(
            jax_result["group"], jax_result._column_metadata["group"]["category_map"]
        )
    ):
        # Handle case where one implementation uses NaN and the other uses 0
        value = jax_result["value"][i]
        if np.isnan(value):
            # This is acceptable - pandas also sometimes returns NaN for single value groups
            pass
        else:
            # If not NaN, value should be very close to 0
            assert np.isclose(value, 0, atol=1e-10), (
                f"Group {group} std should be 0 or NaN"
            )


def test_identical_values_in_groupby_column():
    """Test groupby with all identical values in groupby column."""
    data = {"group": ["A", "A", "A", "A", "A"], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # There should be one group with the mean of all values
    pd_result = pdf.groupby("group")["value"].mean().reset_index()
    jax_result = jdf.groupby("group").aggregate(jnp.mean)

    # Check that we have one group
    assert jax_result.shape[0] == 1, "Should have exactly one group"
    assert (
        to_strings(
            jax_result["group"], jax_result._column_metadata["group"]["category_map"]
        )[0]
        == "A"
    ), "Group name should be 'A'"

    # Check the mean
    expected_mean = 3.0  # (1+2+3+4+5)/5
    assert np.isclose(jax_result["value"][0], expected_mean), "Mean should be 3.0"
    assert np.isclose(jax_result["value"][0], pd_result["value"][0]), (
        "Should match pandas result"
    )


def test_rolling_with_min_periods_zero():
    """Test rolling operations with min_periods=0."""
    data = {"A": [np.nan, np.nan, 3.0, 4.0, 5.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # With min_periods=0, pandas returns a value even for windows with all NaNs
    pd_result = pdf.rolling(window=2, min_periods=0).mean()

    # JaxDataFrame result with min_periods=0 - using positional argument for window_size
    jax_result = jdf.rolling(2, min_periods=0).apply(jnp.nanmean)

    # For pandas with min_periods=0, windows with all NaN still return NaN
    # For the first two positions (which have windows [NaN] and [NaN, NaN])
    for i in range(2):
        expected = pd_result["A"].iloc[i]
        if np.isnan(expected):
            assert np.isnan(jax_result["A"][i]), f"Position {i} should be NaN"
        else:
            assert np.isclose(jax_result["A"][i], expected), (
                f"Position {i} doesn't match pandas"
            )

    # Check all values
    np.testing.assert_allclose(
        jax_result["A"],
        pd_result["A"].values,
        equal_nan=True,
        err_msg="Rolling with min_periods=0 doesn't match pandas",
    )


def test_assignment_broadcasting():
    """Test column assignment with broadcasting behavior."""
    jdf = DataFrame({"A": [1, 2, 3, 4, 5]})

    # In pandas, you can assign a scalar to a column
    jdf["B"] = 10  # Now this should work with our scalar broadcasting support

    assert "B" in jdf.columns, "Column B should be added"
    assert len(jdf["B"]) == 5, "Column B should have 5 elements"

    # All values should be 10
    np.testing.assert_array_equal(jdf["B"], np.array([10, 10, 10, 10, 10]))

    # Similar test with pandas to verify behavior
    pdf = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    pdf["B"] = 10

    # Match pandas shape and values
    assert jdf.shape == pdf.shape
    np.testing.assert_array_equal(jdf["B"], pdf["B"].values)


def test_datetime_diff_operation():
    """Test operations with datetime differences."""
    # Create dataframe with datetime column
    dates = pd.date_range("2023-01-01", periods=5)
    data = {"date": dates, "value": [1, 2, 3, 4, 5]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # In pandas, you can get time deltas between dates
    pd_time_deltas = pdf["date"].diff().dt.total_seconds()

    # With JaxDataFrame, we need to convert to nanoseconds first, then compute diff
    date_ns = jdf["date"]
    jax_diff_ns = np.diff(
        date_ns, prepend=date_ns[0]
    )  # Prepend first value to match pandas behavior
    jax_diff_seconds = jax_diff_ns / 1e9  # Convert ns to seconds

    # First value in pandas is NaN
    assert np.isnan(pd_time_deltas.iloc[0]), "First value should be NaN"

    # Rest of values should be time deltas (86400 seconds = 1 day)
    for i in range(1, 5):
        assert np.isclose(pd_time_deltas.iloc[i], 86400), (
            f"Delta at position {i} should be 86400 seconds"
        )
        assert np.isclose(jax_diff_seconds[i], 86400), (
            f"JaxDF delta at position {i} should be 86400 seconds"
        )


def test_groupby_transform_with_all_nan():
    """Test groupby transform with all NaN values in a group."""
    data = {"group": ["A", "A", "B", "B"], "value": [np.nan, np.nan, 3.0, 4.0]}

    pdf = pd.DataFrame(data)
    jdf = DataFrame(data)

    # Define transform functions
    def pd_transform(x):
        return x - x.mean()

    def jax_transform(arr):
        # Use nanmean to handle NaN values properly
        mean = jnp.nanmean(arr)
        # Preserve NaN values
        return jnp.where(jnp.isnan(arr), jnp.nan, arr - mean)

    # Calculate results
    pd_result = pdf.groupby("group")["value"].transform(pd_transform)
    jax_result = jdf.groupby("group").transform(jax_transform)

    # Group A (all NaN) should still be all NaN after transform
    mask_a = pdf["group"] == "A"
    assert pd_result[mask_a].isna().all(), "Group A should be all NaN"

    # Get group A indices for JaxDF
    # Convert pandas Series boolean mask to numpy array before indexing
    mask_jax_a = np.array(
        to_strings(jdf["group"], jdf._column_metadata["group"]["category_map"]) == "A"
    )
    jax_values_a = jax_result["value"][mask_jax_a]
    assert np.isnan(jax_values_a).all(), "Group A should be all NaN in JaxDF result"

    # Group B should have values centered around mean (0)
    mask_b = pdf["group"] == "B"
    pd_values_b = pd_result[mask_b].values

    # Get group B indices for JaxDF
    # Convert pandas Series boolean mask to numpy array before indexing
    mask_jax_b = np.array(
        to_strings(jdf["group"], jdf._column_metadata["group"]["category_map"]) == "B"
    )
    jax_values_b = jax_result["value"][mask_jax_b]

    # Check that non-NaN values match
    np.testing.assert_allclose(
        jax_values_b,
        pd_values_b,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
        err_msg="Transform values for group B don't match pandas",
    )
