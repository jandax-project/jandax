import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jandax.core import DataFrame


@pytest.fixture
def arithmetic_df():
    """Create a DataFrame with numeric data for arithmetic testing."""
    pandas_df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [1.5, 2.5, 3.5, 4.5, 5.5],
            "D": [0, 0.5, 1.0, 1.5, 2.0],
        }
    )
    return DataFrame(pandas_df)


@pytest.fixture
def mixed_df():
    """Create a DataFrame with mixed data types."""
    pandas_df = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5],
            "float": [1.1, 2.2, 3.3, 4.4, 5.5],
            "string": ["a", "b", "c", "d", "e"],
        }
    )
    return DataFrame(pandas_df)


@pytest.fixture
def nan_df():
    """Create a DataFrame with NaN values."""
    pandas_df = pd.DataFrame(
        {
            "A": [1, np.nan, 3, np.nan, 5],
            "B": [10, 20, np.nan, 40, 50],
        }
    )
    return DataFrame(pandas_df)


# Column-Column arithmetic tests
def test_column_addition(arithmetic_df):
    """Test adding two columns."""
    # Add columns A and B
    result = arithmetic_df["A"] + arithmetic_df["B"]

    # Check result
    expected = np.array([11, 22, 33, 44, 55])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_plus_B"] = result

    # Verify the new column exists with correct values
    assert "A_plus_B" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_plus_B"], expected)


def test_column_subtraction(arithmetic_df):
    """Test subtracting two columns."""
    # Subtract column A from column B
    result = arithmetic_df["B"] - arithmetic_df["A"]

    # Check result
    expected = np.array([9, 18, 27, 36, 45])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["B_minus_A"] = result

    # Verify the new column exists with correct values
    assert "B_minus_A" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["B_minus_A"], expected)


def test_column_multiplication(arithmetic_df):
    """Test multiplying two columns."""
    # Multiply columns A and B
    result = arithmetic_df["A"] * arithmetic_df["B"]

    # Check result
    expected = np.array([10, 40, 90, 160, 250])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_times_B"] = result

    # Verify the new column exists with correct values
    assert "A_times_B" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_times_B"], expected)


def test_column_division(arithmetic_df):
    """Test dividing two columns."""
    # Divide column B by column A
    result = arithmetic_df["B"] / arithmetic_df["A"]

    # Check result
    expected = np.array([10, 10, 10, 10, 10])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["B_div_A"] = result

    # Verify the new column exists with correct values
    assert "B_div_A" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["B_div_A"], expected)


# Column-Scalar arithmetic tests
def test_scalar_addition(arithmetic_df):
    """Test adding a scalar to a column."""
    # Add 5 to column A
    result = arithmetic_df["A"] + 5

    # Check result
    expected = np.array([6, 7, 8, 9, 10])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_plus_5"] = result

    # Verify the new column exists with correct values
    assert "A_plus_5" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_plus_5"], expected)


def test_scalar_subtraction(arithmetic_df):
    """Test subtracting a scalar from a column."""
    # Subtract 1 from column A
    result = arithmetic_df["A"] - 1

    # Check result
    expected = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_minus_1"] = result

    # Verify the new column exists with correct values
    assert "A_minus_1" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_minus_1"], expected)


def test_scalar_multiplication(arithmetic_df):
    """Test multiplying a column by a scalar."""
    # Multiply column A by 3
    result = arithmetic_df["A"] * 3

    # Check result
    expected = np.array([3, 6, 9, 12, 15])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_times_3"] = result

    # Verify the new column exists with correct values
    assert "A_times_3" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_times_3"], expected)


def test_scalar_division(arithmetic_df):
    """Test dividing a column by a scalar."""
    # Divide column A by 2
    result = arithmetic_df["A"] / 2

    # Check result
    expected = np.array([0.5, 1, 1.5, 2, 2.5])
    np.testing.assert_array_equal(result, expected)

    # Store result in a new column
    arithmetic_df["A_div_2"] = result

    # Verify the new column exists with correct values
    assert "A_div_2" in arithmetic_df.columns
    np.testing.assert_array_equal(arithmetic_df["A_div_2"], expected)


# Column-Column operations with assignment
def test_combined_operations(arithmetic_df):
    """Test combined arithmetic operations."""
    # Perform a more complex calculation: (A + B) * C
    result = (arithmetic_df["A"] + arithmetic_df["B"]) * arithmetic_df["C"]

    # Calculate expected result
    expected = np.array([11, 22, 33, 44, 55]) * np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # Check result with small tolerance for floating point precision
    np.testing.assert_allclose(result, expected)

    # Store result in a new column
    arithmetic_df["complex_calc"] = result

    # Verify the new column exists with correct values
    assert "complex_calc" in arithmetic_df.columns
    np.testing.assert_allclose(arithmetic_df["complex_calc"], expected)


# Mathematical function tests
def test_sqrt_function(arithmetic_df):
    """Test square root function."""
    # Take square root of column A
    result = jnp.sqrt(arithmetic_df["A"])

    # Calculate expected result
    expected = np.sqrt(np.array([1, 2, 3, 4, 5]))

    # Check result with small tolerance for floating point precision
    np.testing.assert_allclose(result, expected)

    # Store result in a new column
    arithmetic_df["sqrt_A"] = result

    # Verify the new column exists with correct values
    assert "sqrt_A" in arithmetic_df.columns
    np.testing.assert_allclose(arithmetic_df["sqrt_A"], expected)


def test_log_function(arithmetic_df):
    """Test natural log function."""
    # Take log of column A
    result = jnp.log(arithmetic_df["A"])

    # Calculate expected result
    expected = np.log(np.array([1, 2, 3, 4, 5]))

    # Check result with small tolerance for floating point precision
    np.testing.assert_allclose(result, expected)

    # Store result in a new column
    arithmetic_df["log_A"] = result

    # Verify the new column exists with correct values
    assert "log_A" in arithmetic_df.columns
    np.testing.assert_allclose(arithmetic_df["log_A"], expected)


def test_exp_function(arithmetic_df):
    """Test exponential function."""
    # Take exp of column D (using smaller values to avoid overflow)
    result = jnp.exp(arithmetic_df["D"])

    # Calculate expected result
    expected = np.exp(np.array([0, 0.5, 1.0, 1.5, 2.0]))

    # Check result with small tolerance for floating point precision
    np.testing.assert_allclose(result, expected)

    # Store result in a new column
    arithmetic_df["exp_D"] = result

    # Verify the new column exists with correct values
    assert "exp_D" in arithmetic_df.columns
    np.testing.assert_allclose(arithmetic_df["exp_D"], expected)


# Edge case tests
def test_nan_handling(nan_df):
    """Test arithmetic with NaN values."""
    # Add columns with NaN values
    result = nan_df["A"] + nan_df["B"]

    # NaNs propagate in arithmetic operations
    expected = np.array([11, np.nan, np.nan, np.nan, 55])

    # Check where values are NaN and where they are finite
    nan_mask = np.isnan(result)
    expected_nan_mask = np.isnan(expected)
    np.testing.assert_array_equal(nan_mask, expected_nan_mask)

    # Check only non-NaN values
    finite_mask = ~nan_mask
    np.testing.assert_array_equal(result[finite_mask], expected[finite_mask])

    # Test NaN replacement using nansum
    result_sum = jnp.nansum(jnp.stack([nan_df["A"], nan_df["B"]]), axis=0)
    expected_sum = np.array([11, 20, 3, 40, 55])  # Fixed: 22 -> 20
    np.testing.assert_allclose(result_sum, expected_sum)


def test_division_by_zero(arithmetic_df):
    """Test division by zero handling."""
    # Create a column with zeros
    zeros = jnp.zeros(5)

    # Divide by zero should produce infinity
    with np.errstate(divide="ignore"):
        result = arithmetic_df["A"] / zeros

    # Check that result contains infinity
    assert jnp.isinf(result).any()

    # Check specific values
    np.testing.assert_array_equal(jnp.isinf(result), np.ones(5, dtype=bool))


def test_apply_for_arithmetic(arithmetic_df):
    """Test using apply method for arithmetic operations."""
    # Double each value in column A using apply
    result = arithmetic_df.apply(lambda x: x * 2, axis=0)

    # Check result for column A
    expected_A = np.array([2, 4, 6, 8, 10])
    np.testing.assert_array_equal(result["A"], expected_A)

    # Check result for column B
    expected_B = np.array([20, 40, 60, 80, 100])
    np.testing.assert_array_equal(result["B"], expected_B)


def test_row_wise_arithmetic(arithmetic_df):
    """Test row-wise arithmetic using apply."""
    # Sum all values in each row
    result = arithmetic_df.apply(lambda x: jnp.sum(x), axis=1)

    # Calculate expected row sums
    expected = np.array(
        [
            1 + 10 + 1.5 + 0,  # Row 1
            2 + 20 + 2.5 + 0.5,  # Row 2
            3 + 30 + 3.5 + 1.0,  # Row 3
            4 + 40 + 4.5 + 1.5,  # Row 4
            5 + 50 + 5.5 + 2.0,  # Row 5
        ]
    )

    # Check results - using the single column created by row-wise apply
    column_name = result.columns[0]
    np.testing.assert_allclose(result[column_name], expected)


def test_broadcasting_assignment(arithmetic_df):
    """Test assignment with broadcasting."""
    # Assign a scalar value to a new column (should broadcast)
    arithmetic_df["constant"] = 42

    # Check that all values are 42
    expected = np.full(5, 42)
    np.testing.assert_array_equal(arithmetic_df["constant"], expected)

    # Assign a scalar to an existing column (should also broadcast)
    arithmetic_df["A"] = 100
    np.testing.assert_array_equal(arithmetic_df["A"], np.full(5, 100))


def test_boolean_operations(arithmetic_df):
    """Test boolean operations on columns."""
    # Create boolean mask where A > 2
    mask = arithmetic_df["A"] > 2

    # Expected result
    expected_mask = np.array([False, False, True, True, True])
    np.testing.assert_array_equal(mask, expected_mask)

    # Use mask to filter values
    filtered_B = arithmetic_df["B"][mask]
    expected_filtered = np.array([30, 40, 50])
    np.testing.assert_array_equal(filtered_B, expected_filtered)

    # Combine boolean conditions
    combined_mask = (arithmetic_df["A"] > 2) & (arithmetic_df["B"] < 50)
    expected_combined = np.array([False, False, True, True, False])
    np.testing.assert_array_equal(combined_mask, expected_combined)
