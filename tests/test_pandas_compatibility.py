import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jandax.core import DataFrame


# Setup fixtures for common test data
@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    return DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    )


@pytest.fixture
def numeric_df():
    """Create a numeric DataFrame for testing."""
    return DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": [100, 200, 300, 400, 500],
        }
    )


@pytest.fixture
def missing_df():
    """Create a DataFrame with missing values for testing."""
    return DataFrame(
        {
            "A": [1, np.nan, 3, np.nan, 5],
            "B": [10.0, 20.0, np.nan, 40.0, 50.0],
        }
    )


@pytest.fixture
def date_df():
    """Create a DataFrame with date values for testing."""
    return DataFrame(
        {"date": pd.date_range("2023-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
    )


# Tests for common pandas operations
def test_scalar_assignment(simple_df):
    """Test scalar assignment (broadcasting) to columns."""
    # Assign scalar to new column
    simple_df["D"] = 100
    assert "D" in simple_df.columns
    np.testing.assert_array_equal(simple_df["D"], np.array([100, 100, 100, 100, 100]))

    # Assign scalar to existing column
    simple_df["A"] = 999
    np.testing.assert_array_equal(simple_df["A"], np.array([999, 999, 999, 999, 999]))


def test_scalar_arithmetic(numeric_df):
    """Test arithmetic operations with scalars."""
    # Addition
    result = numeric_df.apply(lambda x: x + 10, axis=0)
    np.testing.assert_array_equal(result["A"], np.array([11, 12, 13, 14, 15]))

    # Multiplication
    result = numeric_df.apply(lambda x: x * 2, axis=0)
    np.testing.assert_array_equal(result["A"], np.array([2, 4, 6, 8, 10]))
    np.testing.assert_array_equal(
        result["B"], np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    )


def test_boolean_indexing():
    """Test boolean indexing for row selection."""
    df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})

    # In pandas, you can do: df[df['A'] > 3]
    mask = np.array(df["A"] > 3)
    filtered = DataFrame({"A": df["A"][mask], "B": df["B"][mask]})

    assert filtered.shape == (2, 2)
    np.testing.assert_array_equal(filtered["A"], np.array([4, 5]))
    np.testing.assert_array_equal(filtered["B"], np.array([40, 50]))


def test_chained_indexing(simple_df):
    """Test chained indexing operations."""
    # In pandas, you can do: df['A'].iloc[0]
    first_value = simple_df["A"][0]
    assert first_value == 1

    # Get first three values of column
    first_three = simple_df["A"][:3]
    np.testing.assert_array_equal(first_three, np.array([1, 2, 3]))


def test_filling_nan_values(missing_df):
    """Test filling NaN values."""
    # Fill with scalar
    filled_0 = missing_df.apply(lambda x: jnp.nan_to_num(x, nan=0.0), axis=0)
    np.testing.assert_array_equal(filled_0["A"], np.array([1.0, 0.0, 3.0, 0.0, 5.0]))

    # Fill with different value
    filled_999 = missing_df.apply(lambda x: jnp.nan_to_num(x, nan=999.0), axis=0)
    np.testing.assert_array_equal(
        filled_999["A"], np.array([1.0, 999.0, 3.0, 999.0, 5.0])
    )


def test_basic_math_functions(numeric_df):
    """Test applying basic math functions to dataframe."""
    # Square root
    sqrt_result = numeric_df.apply(lambda x: jnp.sqrt(x), axis=0)
    expected = np.sqrt(np.array([1, 2, 3, 4, 5]))
    np.testing.assert_allclose(sqrt_result["A"], expected)

    # Absolute value
    abs_result = numeric_df.apply(lambda x: jnp.abs(x), axis=0)
    np.testing.assert_array_equal(abs_result["A"], np.array([1, 2, 3, 4, 5]))


def test_multiple_column_assignment():
    """Test assigning values to multiple columns at once."""
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Create a new dataframe with modified values for columns A and B
    new_df = DataFrame({"A": [10, 20, 30], "B": [40, 50, 60], "C": df["C"]})

    # Verify the changes
    np.testing.assert_array_equal(new_df["A"], np.array([10, 20, 30]))
    np.testing.assert_array_equal(new_df["B"], np.array([40, 50, 60]))
    np.testing.assert_array_equal(new_df["C"], np.array([7, 8, 9]))


def test_dataframe_slicing():
    """Test slicing operations on dataframes."""
    df = DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [100, 200, 300, 400, 500],
        }
    )

    # Row slicing - select first 3 rows
    # This would create a view in pandas
    first_three_rows = DataFrame({"A": df["A"][:3], "B": df["B"][:3], "C": df["C"][:3]})

    assert first_three_rows.shape == (3, 3)
    np.testing.assert_array_equal(first_three_rows["A"], np.array([1, 2, 3]))

    # Column slicing - already tested with df[["A", "B"]]


def test_assignment_with_condition():
    """Test conditional assignment (where operation)."""
    df = DataFrame({"A": [1, 2, 3, 4, 5]})

    # In pandas: df.loc[df['A'] > 3, 'A'] = 0
    # We can implement similar behavior with JAX's where
    mask = np.array(df["A"] > 3)
    df["A"] = jnp.where(mask, 0, df["A"])

    expected = np.array([1, 2, 3, 0, 0])
    np.testing.assert_array_equal(df["A"], expected)


def test_mixed_type_operations():
    """Test operations with mixed data types."""
    df = DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

    # Add string prefix to categorical column
    new_strings = [f"prefix_{s}" for s in df["B"].to_strings()]
    df["B"] = new_strings

    # Check the result
    expected = ["prefix_a", "prefix_b", "prefix_c", "prefix_d", "prefix_e"]
    assert list(df["B"].to_strings()) == expected


def test_datetime_operations(date_df):
    """Test operations with datetime columns."""
    # Extract year from datetime column
    dt_values = date_df["date"].to_datetime()
    years = [dt.year for dt in dt_values]
    date_df["year"] = years

    # All dates should be from 2023
    expected = [2023, 2023, 2023, 2023, 2023]
    np.testing.assert_array_equal(date_df["year"], np.array(expected))


def test_explict_type_conversion():
    """Test explicit type conversion."""
    df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [1.5, 2.5, 3.5, 4.5, 5.5]})

    # Convert float to int
    int_values = np.array(df["B"]).astype(np.int32)
    df["B_int"] = int_values

    expected = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(df["B_int"], expected)


def test_add_row():
    """Test adding a row to the DataFrame."""
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # In pandas, we might use df.loc[len(df)] = [4, 7]
    # We'll have to create a new dataframe with the appended row
    new_df = DataFrame({"A": np.append(df["A"], [4]), "B": np.append(df["B"], [7])})

    assert new_df.shape == (4, 2)
    np.testing.assert_array_equal(new_df["A"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(new_df["B"], np.array([4, 5, 6, 7]))


def test_describe_statistics():
    """Test computing descriptive statistics."""
    df = DataFrame({"A": [1, 2, 3, 4, 5]})

    # Computing statistics
    mean_val = jnp.mean(df["A"])
    std_val = jnp.std(df["A"])
    min_val = jnp.min(df["A"])
    max_val = jnp.max(df["A"])

    # Check results
    assert mean_val == 3.0
    assert min_val == 1.0
    assert max_val == 5.0
    assert np.isclose(std_val, 1.4142135)


def test_column_rename():
    """Test renaming columns."""
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Create new dataframe with renamed columns
    renamed = DataFrame({"X": df["A"], "Y": df["B"]})

    assert renamed.columns == ["X", "Y"]
    np.testing.assert_array_equal(renamed["X"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(renamed["Y"], np.array([4, 5, 6]))


def test_boolean_operators():
    """Test boolean operators on columns."""
    df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

    # Compute boolean conditions
    mask1 = np.array(df["A"] > 2)
    mask2 = np.array(df["B"] > 2)

    # Logical AND
    and_mask = np.logical_and(mask1, mask2)
    expected_and = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(and_mask, expected_and)

    # Logical OR
    or_mask = np.logical_or(mask1, mask2)
    # Fix: Update the expected result to match actual behavior
    expected_or = np.array([True, True, True, True, True])
    np.testing.assert_array_equal(or_mask, expected_or)


def test_numeric_indexing():
    """Test numeric indexing of rows and columns."""
    df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

    # Get 2nd value of column A (index 1)
    assert df["A"][1] == 2

    # Get last value of column B
    assert df["B"][-1] == 1


def test_shifting_data():
    """Test shifting data (like df.shift())."""
    df = DataFrame({"A": [1, 2, 3, 4, 5]})

    # Shift values forward by 1 position
    shifted = np.hstack([[np.nan], df["A"][:-1]])
    df["A_shifted"] = shifted

    expected = np.array([np.nan, 1, 2, 3, 4])
    # Fix: Use assert_allclose instead of assert_array_equal for NaN comparison
    np.testing.assert_allclose(df["A_shifted"], expected, equal_nan=True)
