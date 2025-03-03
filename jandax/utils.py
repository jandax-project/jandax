from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


# Utility functions for datetime handling
def pd_datetime_to_ns(
    dt_values: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
) -> np.ndarray:
    """Convert pandas datetime values to nanoseconds since epoch."""
    if isinstance(dt_values, pd.DatetimeIndex):
        # Convert DatetimeIndex to nanoseconds directly. ensure we return a numpy array.
        return np.array(dt_values.astype(np.int64))
    elif isinstance(dt_values, pd.Series) and pd.api.types.is_datetime64_any_dtype(
        dt_values.dtype
    ):
        # Convert datetime Series to nanoseconds
        return np.array(dt_values.astype(np.int64))
    elif isinstance(dt_values, np.ndarray) and np.issubdtype(
        dt_values.dtype, np.datetime64
    ):
        # Convert numpy datetime array to nanoseconds
        return dt_values.astype(np.int64)
    else:
        # Try to convert to pd.DatetimeIndex first, then to nanoseconds
        try:
            return np.array(pd.DatetimeIndex(dt_values).astype(np.int64))
        except Exception as e:
            raise ValueError(
                f"""Cannot convert values of type {type(dt_values)} to datetime
                nanoseconds. \nOriginal Error: {e}"""
            ) from e


def ns_to_pd_datetime(ns_array):
    """
    Convert a numeric array containing nanosecond timestamps to pandas datetime objects.

    Args:
        ns_array: Array containing nanosecond timestamp values

    Returns:
        pandas.DatetimeIndex with converted datetime values
    """
    return pd.to_datetime(np.asarray(ns_array), unit="ns")


def to_datetime(values: jax.Array) -> pd.Series:
    """
    Convert a JAX array of nanosecond timestamps to pandas datetime Series.

    Args:
        values: JAX array containing nanosecond timestamp values

    Returns:
        pandas Series with datetime values
    """
    return ns_to_pd_datetime(values)


def to_strings(values: jax.Array, category_map: Dict[int, str]) -> pd.Series:
    """
    Convert a JAX array of category codes to pandas Series of strings.

    Args:
        values: JAX array containing category codes
        category_map: Dictionary mapping category codes to string values

    Returns:
        pandas Series with string values
    """
    return pd.Series(
        [
            category_map.get(int(x), None) if x >= 0 and not jnp.isnan(x) else None
            for x in values
        ]
    )
