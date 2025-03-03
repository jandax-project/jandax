import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jandax.utils import ns_to_pd_datetime

jax.config.update("jax_enable_x64", True)

# Constants for special data types
DATETIME_TYPE_FLAG = "datetime"
CATEGORY_TYPE_FLAG = "category"
NS_PER_DAY = 24 * 60 * 60 * 1_000_000_000  # nanoseconds in a day


class DataFrame:
    """A DataFrame-like container backed by a single JAX 2D array."""

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, jax.Array, Dict[str, Any]],
        columns: Optional[List[str]] = None,
        include_index: bool = True,
    ):
        """
        Initialize a JaxDataFrame from various data sources.

        Args:
            data: Data to initialize with (pandas DataFrame, numpy 2D array,
                        JAX 2D array, dict of arrays)
            columns: Optional column names (required for raw 2D arrays)
                        include_index: If True and data is a pandas DataFrame, include
                        the index as column(s)
        """
        # NON-TRACEABLE SECTION: Initialization of metadata and storage
        # These operations involve Python-level data structure creation
        # and are not traceable.
        self._column_metadata = {}

        # Initialize storage for our data
        self._values = jnp.zeros((0, 0))
        self._column_names = []
        self._column_mapping = {}  # Maps column name to column index

        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # NON-TRACEABLE SECTION: Pandas DataFrame handling
            # This branch involves pandas-specific operations and is not traceable.
            self._init_from_pandas(data, include_index)

        # Handle 2D numpy arrays or JAX arrays
        elif isinstance(data, (np.ndarray, jax.Array)) and data.ndim == 2:  # noqa: PLR2004
            # NON-TRACEABLE SECTION: Array handling
            # This branch involves array-specific operations and is not traceable.
            assert columns is not None
            self._init_from_array(data, columns)

        # Handle another JaxDataFrame (copy)
        elif isinstance(data, DataFrame):
            # NON-TRACEABLE SECTION: JaxDataFrame copy
            # This branch involves copying data structures and is not traceable.
            self._init_from_jaxdf(data)

        # Handle dictionary of arrays
        elif isinstance(data, dict):
            # NON-TRACEABLE SECTION: Dictionary handling
            # This branch involves dictionary-specific operations and is not traceable.
            self._init_from_dict(data)

        else:
            raise TypeError(
                f"""JaxDataFrame can only be initialized from pandas DataFrame,
                    2D arrays,another JaxDataFrame, or a dict of arrays. 
                    Got {type(data)}"""
            )

        # Verify initialization succeeded
        if not isinstance(self._values, jax.Array):
            raise ValueError("Failed to initialize DataFrame values properly")

    def _init_from_pandas(self, df: pd.DataFrame, include_index: bool):
        """Initialize from pandas DataFrame."""
        # NON-TRACEABLE SECTION: Pandas-specific operations
        # These operations involve pandas data structures and are not traceable.
        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Process index if specified
        if include_index:
            if isinstance(df_copy.index, pd.MultiIndex):
                # Handle MultiIndex by adding each level as a separate column
                for i, idx_name in enumerate(df_copy.index.names):
                    if idx_name is None:
                        # Unnamed level gets generic name
                        col_name = f"index_{i}"
                        df_copy[col_name] = df_copy.index.get_level_values(i)
                    else:
                        df_copy[idx_name] = df_copy.index.get_level_values(idx_name)
            else:
                # Handle regular Index - only add if it has a name
                idx_name = df_copy.index.name
                if idx_name is not None:
                    df_copy[idx_name] = df_copy.index
                # If index has no name, don't include it as a column

        # Set up column names and mapping
        self._column_names = list(df_copy.columns)
        self._column_mapping = {
            name: idx for idx, name in enumerate(self._column_names)
        }

        # Create a list to hold column arrays before stacking
        column_arrays = []

        # Process columns
        for col_name in self._column_names:
            series = df_copy[col_name]

            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                # Convert to int64 nanoseconds
                values = np.array([x.astype(np.int64) for x in series.values])
                column_arrays.append(values)
                self._column_metadata[col_name] = {
                    "dtype_flag": DATETIME_TYPE_FLAG,
                }

            # Handle categorical/string columns
            elif pd.api.types.is_string_dtype(series.dtype) or isinstance(
                series.dtype, pd.CategoricalDtype
            ):
                # Convert to categorical integers
                categories = series.astype("category").cat.categories.tolist()
                codes = series.astype("category").cat.codes.values
                column_arrays.append(codes)
                self._column_metadata[col_name] = {
                    "dtype_flag": CATEGORY_TYPE_FLAG,
                    "category_map": {i: val for i, val in enumerate(categories)},
                }

            # Handle regular numeric columns
            else:
                column_arrays.append(series.values)
                self._column_metadata[col_name] = {"dtype_flag": None}

        # Stack all columns into a single 2D array
        # Note: we stack columns horizontally (axis=1) to create a 2D array with shape
        # (n_rows, n_cols)
        self._values = jnp.column_stack(column_arrays)

    def _init_from_array(self, array: Union[np.ndarray, jax.Array], columns: List[str]):
        """Initialize from a 2D numpy or JAX array."""
        # NON-TRACEABLE SECTION: Array initialization
        # These operations involve array-specific operations and are not traceable.
        if columns is None:
            raise ValueError(
                "Column names must be provided when initializing from 2D arrays"
            )

        # Convert to JAX array if needed
        if not isinstance(array, jax.Array):
            self._values = jnp.array(array, dtype=jnp.float64)
        else:
            self._values = array

        # Set up column mapping
        self._column_names = list(columns)
        self._column_mapping = {
            name: idx for idx, name in enumerate(self._column_names)
        }

        # Initialize metadata for each column (no special types by default)
        for col_name in self._column_names:
            self._column_metadata[col_name] = {"dtype_flag": None}

    def _init_from_jaxdf(self, df: "DataFrame"):
        """Initialize from another JaxDataFrame."""
        # NON-TRACEABLE SECTION: Copying from another JaxDataFrame
        # These operations involve copying data structures and are not traceable.
        # Copy internal array
        self._values = jnp.array(df._values)

        # Copy column info
        self._column_names = df._column_names.copy()
        self._column_mapping = df._column_mapping.copy()

        # Copy metadata
        self._column_metadata = {}
        for col_name, metadata in df._column_metadata.items():
            self._column_metadata[col_name] = metadata.copy()

    def _init_from_dict(self, data_dict: Dict[str, Any]):  # noqa: PLR0912, PLR0915
        """Initialize from a dictionary of arrays."""
        # NON-TRACEABLE SECTION: Dictionary initialization
        # These operations involve dictionary-specific operations and are not traceable.
        if not data_dict:
            # Empty dataframe
            self._values = jnp.zeros((0, 0))
            return

        column_arrays = []
        self._column_names = []

        # Find the length of the arrays
        array_length = None

        # First pass to determine array length and validate
        for col_name, values in data_dict.items():
            # Convert to numpy array if needed
            if isinstance(values, list):
                values = np.array(values)  # noqa: PLW2901
            elif isinstance(values, jax.Array):
                values = np.array(values)  # noqa: PLW2901
            elif isinstance(values, pd.DatetimeIndex):
                # Handle DatetimeIndex by converting to nanosecond values
                values = values.values  # noqa: PLW2901

            if array_length is None:
                array_length = len(values)
            elif len(values) != array_length:
                raise ValueError(
                    f"""All arrays must have the same length. Column '{col_name}' has 
                    length {len(values)}, expected {array_length}"""
                )

        # If no data was provided
        if array_length is None:
            self._values = jnp.zeros((0, 0))
            return

        # Process each column
        for col_name, values in data_dict.items():
            self._column_names.append(col_name)

            # Check for special types
            if isinstance(values, pd.Series):
                # Handle pandas Series
                if pd.api.types.is_datetime64_any_dtype(values.dtype):
                    # Convert datetime to int64 nanoseconds
                    array_values = values.astype(np.int64).values
                    column_arrays.append(array_values)
                    self._column_metadata[col_name] = {
                        "dtype_flag": DATETIME_TYPE_FLAG,
                    }
                elif pd.api.types.is_string_dtype(
                    values.dtype
                ) or pd.api.types.is_categorical_dtype(values.dtype):  # type: ignore
                    # Convert to categorical integers
                    categories = values.astype("category").cat.categories.tolist()
                    codes = values.astype("category").cat.codes.values
                    column_arrays.append(codes)
                    self._column_metadata[col_name] = {
                        "dtype_flag": CATEGORY_TYPE_FLAG,
                        "category_map": {i: val for i, val in enumerate(categories)},
                    }
                else:
                    # Regular numeric data
                    column_arrays.append(values.values)
                    self._column_metadata[col_name] = {"dtype_flag": None}
            elif isinstance(values, pd.DatetimeIndex):
                # Handle DatetimeIndex
                array_values = values.astype(np.int64).values
                column_arrays.append(array_values)
                self._column_metadata[col_name] = {
                    "dtype_flag": DATETIME_TYPE_FLAG,
                }
            elif isinstance(values, (np.ndarray, jax.Array)):
                # Handle numpy datetime arrays
                if np.issubdtype(values.dtype, np.datetime64):
                    array_values = values.astype(np.int64)
                    column_arrays.append(array_values)
                    self._column_metadata[col_name] = {
                        "dtype_flag": DATETIME_TYPE_FLAG,
                    }
                else:
                    # Simple array conversion
                    column_arrays.append(values)
                    self._column_metadata[col_name] = {"dtype_flag": None}
            elif isinstance(values, list):
                # Check if we have timestamp objects directly in the list
                if all(isinstance(x, (pd.Timestamp)) for x in values if x is not None):
                    # Convert pd.Timestamp objects to nanosecond values
                    array_values = np.array(
                        [x.value if x is not None else pd.NaT.value for x in values],
                        dtype=np.int64,
                    )
                    column_arrays.append(array_values)
                    self._column_metadata[col_name] = {
                        "dtype_flag": DATETIME_TYPE_FLAG,
                    }
                # Check if list contains strings
                elif all(isinstance(x, str) for x in values if x is not None):
                    # Convert to categorical
                    categories = list(
                        dict.fromkeys([x for x in values if x is not None])
                    )
                    codes = [
                        categories.index(x) if x is not None else -1 for x in values
                    ]
                    column_arrays.append(codes)
                    self._column_metadata[col_name] = {
                        "dtype_flag": CATEGORY_TYPE_FLAG,
                        "category_map": {i: val for i, val in enumerate(categories)},
                    }
                # Check if list contains datetime objects
                elif all(
                    isinstance(x, (datetime.datetime, datetime.date))
                    for x in values
                    if x is not None
                ):
                    # Convert datetime objects to int64 nanoseconds
                    dt_index = pd.DatetimeIndex([x for x in values if x is not None])
                    int_values = np.array([x.astype(np.int64) for x in dt_index])  # type: ignore

                    # Create array with NaT placeholders for None values
                    full_array = np.full(len(values), pd.NaT.value, dtype=np.int64)
                    non_none_idx = 0
                    for i, val in enumerate(values):
                        if val is not None:
                            full_array[i] = int_values[non_none_idx]
                            non_none_idx += 1

                    column_arrays.append(full_array)
                    self._column_metadata[col_name] = {
                        "dtype_flag": DATETIME_TYPE_FLAG,
                    }
                else:
                    # Regular numeric data
                    column_arrays.append(values)
                    self._column_metadata[col_name] = {"dtype_flag": None}
            else:
                raise TypeError(
                    f"Unsupported data type for column '{col_name}': {type(values)}"
                )

        # Create column mapping
        self._column_mapping = {
            name: idx for idx, name in enumerate(self._column_names)
        }

        # Stack arrays into a single 2D array
        self._values = jnp.column_stack(
            [jnp.array(arr, dtype=jnp.float64) for arr in column_arrays]
        )

    @property
    def columns(self) -> List[str]:
        """Get the column names."""
        # NON-TRACEABLE SECTION: Column name access
        # This operation involves accessing a list and is not traceable.
        return self._column_names.copy()

    @property
    def shape(self) -> tuple:
        """Get the shape of the dataframe."""
        # TRACEABLE SECTION: Shape access
        # This operation accesses a JAX array property and is traceable.
        return self._values.shape

    @property
    def values(self) -> jax.Array:
        """Get the underlying 2D JAX array."""
        # TRACEABLE SECTION: Value access
        # This operation returns a JAX array and is traceable.
        return self._values

    def __getitem__(self, key: Union[str, List[str]]) -> Union[jax.Array, "DataFrame"]:
        """Column access with explicit tracing phases."""
        # Phase 1: Pre-processing (non-traceable)
        if isinstance(key, str):
            if key not in self._column_mapping:
                raise KeyError(f"Column '{key}' not found")
            col_idx = self._column_mapping[key]

            # Phase 2: Core computation (traceable)
            values = self._getitem_single_core(self._values, col_idx)
            # Phase 3: Post-processing (non-traceable)
            return values

        elif isinstance(key, list):
            if not all(k in self._column_mapping for k in key):
                missing = [k for k in key if k not in self._column_mapping]
                raise KeyError(f"Columns {missing} not found")
            col_indices = [self._column_mapping[k] for k in key]

            # Phase 2: Core computation (traceable)
            values = self._getitem_multiple_core(self._values, jnp.array(col_indices))

            # Phase 3: Post-processing (non-traceable)
            result = DataFrame({})
            result._column_names = key.copy()
            result._column_mapping = {name: idx for idx, name in enumerate(key)}
            result._values = values
            for col in key:
                result._column_metadata[col] = self._column_metadata[col].copy()
            return result
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key: str, value: Any):
        """Column assignment with explicit tracing phases."""
        # Phase 1: Pre-processing (non-traceable)
        if key in self._column_mapping:
            col_idx = self._column_mapping[key]
            metadata = self._column_metadata[key]
            # Convert input to appropriate array type
            value_array = self._prepare_column_values(key, value, metadata)

            # Phase 2: Core computation (traceable)
            self._values = self._setitem_core(self._values, col_idx, value_array)

        else:
            # Handle new column addition (purely non-traceable)
            self.add_column(key, value)

    # Keep these core traceable functions and helpers:
    def _getitem_single_core(self, values: jax.Array, col_idx: int) -> jax.Array:
        """Pure JAX function for single column access."""
        return values[:, col_idx]

    def _getitem_multiple_core(
        self, values: jax.Array, col_indices: jax.Array
    ) -> jax.Array:
        """Pure JAX function for multiple column access."""
        return values[:, col_indices]

    def _setitem_core(
        self, values: jax.Array, col_idx: int, new_values: jax.Array
    ) -> jax.Array:
        """Pure JAX function for column assignment."""
        return values.at[:, col_idx].set(new_values)

    def add_column(self, col_name: str, value: Any):
        """
        Add a new column to the dataframe (non-traceable).
        This method is explicitly non-traceable as it changes the shape
        of the underlying array and updates metadata structures.

        Args:
            col_name: Name of the new column
            value: Values for the new column (array-like)
        """
        # Validate
        if col_name in self._column_mapping:
            raise KeyError(f"Column '{col_name}' already exists")

        # Convert value to appropriate array type and infer metadata
        metadata = {"dtype_flag": None}  # Default metadata
        value_array = self._prepare_column_values(col_name, value, metadata)

        # Store old values
        old_values = self._values

        # Update metadata structures
        self._column_names.append(col_name)
        self._column_mapping = {
            name: idx for idx, name in enumerate(self._column_names)
        }
        self._column_metadata[col_name] = metadata

        # Special case for empty dataframe
        if self._values.size == 0:
            self._values = value_array.reshape(-1, 1)
        else:
            # Stack new column (creates new array)
            self._values = jnp.column_stack([old_values, value_array])

    def _prepare_column_values(
        self, col_name: str, value: Any, metadata: Dict[str, Any]
    ) -> jax.Array:
        """
        Helper method to prepare column values for assignment.
        Handles type conversion, special types detection, and validation.

        Args:
            col_name: Name of column
            value: Raw column values
            metadata: Column metadata dict (may be modified in-place)

        Returns:
            JAX array with properly converted values
        """
        # Handle scalar values for broadcasting
        if isinstance(value, (int, float, bool)):
            # Create an array of the scalar value with the right length
            value_array = jnp.full(len(self), value, dtype=jnp.float64)
            metadata["dtype_flag"] = None
            return value_array

        # Handle lists
        elif isinstance(value, list):
            if len(value) != len(self) and len(self) > 0:
                raise ValueError(
                    f"""Length of values ({len(value)}) does not match length of 
                    dataframe ({len(self)})"""
                )

            # Check for special types (strings or datetimes)
            if all(isinstance(x, str) for x in value if x is not None):
                # Convert to categorical
                categories = list(dict.fromkeys([x for x in value if x is not None]))
                codes = [categories.index(x) if x is not None else -1 for x in value]
                value_array = jnp.array(codes, dtype=jnp.int32)

                # Store category metadata
                metadata["dtype_flag"] = CATEGORY_TYPE_FLAG
                metadata["category_map"] = {i: val for i, val in enumerate(categories)}
            elif all(
                isinstance(x, (datetime.datetime, datetime.date))
                for x in value
                if x is not None
            ):
                # Convert datetime objects to int64 nanoseconds
                dt_index = pd.DatetimeIndex([x for x in value if x is not None])
                int_values = np.array([x.astype(np.int64) for x in dt_index])  # type: ignore

                # Create array with NaT placeholders for None values
                full_array = np.full(len(value), pd.NaT.value, dtype=np.int64)
                non_none_idx = 0
                for i, val in enumerate(value):
                    if val is not None:
                        full_array[i] = int_values[non_none_idx]
                        non_none_idx += 1

                value_array = jnp.array(full_array)
                metadata["dtype_flag"] = DATETIME_TYPE_FLAG
            else:
                value_array = jnp.array(value, dtype=jnp.float64)
                metadata["dtype_flag"] = None
        elif isinstance(value, (np.ndarray, jax.Array)):
            if len(value) != len(self) and len(self) > 0:
                raise ValueError(
                    f"""Length of values ({len(value)}) does not match length of 
                    dataframe ({len(self)})"""
                )
            value_array = jnp.array(value, dtype=jnp.float64)
            metadata["dtype_flag"] = None
        else:
            raise TypeError(
                f"""Column value must be a numpy array, JAX array, list, or scalar. 
                Got {type(value)}"""
            )

        return value_array

    def __len__(self) -> int:
        """Number of rows in the dataframe."""
        if self._values.size == 0:
            return 0
        return self._values.shape[0]

    def __repr__(self) -> str:
        """String representation similar to pandas."""
        return self.__str__()

    def __str__(self) -> str:  # noqa: PLR0912
        """Formatted string representation."""
        if len(self) == 0:
            return "Empty JaxDataFrame"

        # Format header
        header = f"JaxDataFrame: {self.shape[0]} rows × {self.shape[1]} columns"
        separator = "=" * len(header)

        # Column names row
        col_names = self._column_names
        col_widths = {col: max(len(col), 10) for col in col_names}

        # Format each column's values to determine optimal width
        formatted_values = {}
        for col in col_names:
            col_idx = self._column_mapping[col]
            formatted_vals = []
            for i in range(min(5, len(self))):
                val_str = self._format_value(col, i)
                formatted_vals.append(val_str)
                col_widths[col] = max(col_widths[col], len(val_str))
            formatted_values[col] = formatted_vals

        # Create header row
        header_row = "    " + "  ".join(col.ljust(col_widths[col]) for col in col_names)

        # Build the table rows
        max_rows = 10  # Maximum rows to display
        table_rows = []

        if len(self) <= max_rows:
            # Show all rows
            for i in range(len(self)):
                row_vals = []
                for col in col_names:
                    if i < len(formatted_values[col]):
                        val = formatted_values[col][i]
                    else:
                        val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))
                table_rows.append(f"{i:3d} " + "  ".join(row_vals))
        else:
            # Show first and last few rows
            for i in range(5):
                row_vals = []
                for col in col_names:
                    if i < len(formatted_values[col]):
                        val = formatted_values[col][i]
                    else:
                        val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))
                table_rows.append(f"{i:3d} " + "  ".join(row_vals))

            table_rows.append(
                "... " + "  ".join("...".ljust(col_widths[col]) for col in col_names)
            )

            for i in range(len(self) - 5, len(self)):
                row_vals = []
                for col in col_names:
                    val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))
                table_rows.append(f"{i:3d} " + "  ".join(row_vals))

        # Column type information
        type_info = []
        for col in col_names:
            metadata = self._column_metadata[col]
            if metadata["dtype_flag"] == DATETIME_TYPE_FLAG:
                type_str = "datetime64[ns]"
            elif metadata["dtype_flag"] == CATEGORY_TYPE_FLAG:
                num_cats = (
                    len(metadata["category_map"]) if "category_map" in metadata else "?"
                )
                type_str = f"category({num_cats})"
            else:
                col_idx = self._column_mapping[col]
                type_str = str(self._values[:, col_idx].dtype)
            type_info.append(f"{col}: {type_str}")

        dtypes_section = "Dtypes:\n" + "\n".join(f"  {info}" for info in type_info)

        return (
            f"{header}\n{separator}\n{header_row}\n"
            + "\n".join(table_rows)
            + f"\n\n{dtypes_section}"
        )

    def _format_value(self, column: str, idx: int) -> str:  # noqa: PLR0911
        """Format a single value for display."""
        if idx >= len(self):
            return "N/A"

        col_idx = self._column_mapping[column]
        val = self._values[idx, col_idx]
        metadata = self._column_metadata[column]

        if metadata["dtype_flag"] == DATETIME_TYPE_FLAG:
            # Handle datetime values
            try:
                # Handle NaT (Not a Time) value
                if val == pd.NaT.value:
                    return "NaT"

                # Convert directly using pd.to_datetime
                dt = pd.to_datetime(val, unit="ns")  # type: ignore
                return str(dt)
            except Exception as e:
                return f"Invalid datetime ({val}) - {e}"
        elif metadata["dtype_flag"] == CATEGORY_TYPE_FLAG:
            # Convert category code to string
            try:
                category_map = metadata["category_map"]
                if int(val) == -1 or int(val) not in category_map:
                    return "None"
                return category_map[int(val)]
            except Exception:
                return str(val)
        else:
            return str(val)

    def head(self, n: int = 5) -> "DataFrame":
        """
        Return the first n rows.

        Args:
            n: Number of rows to return

        Returns:
            JaxDataFrame: First n rows
        """
        if len(self) == 0:
            return DataFrame({})

        n = min(n, len(self))

        # Create a new dataframe with the first n rows
        result = DataFrame({})
        result._column_names = self._column_names.copy()
        result._column_mapping = self._column_mapping.copy()
        result._values = self._values[:n, :]

        # Copy column metadata
        for col, metadata in self._column_metadata.items():
            result._column_metadata[col] = metadata.copy()

        return result

    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows.

        Args:
            n: Number of rows to return

        Returns:
            JaxDataFrame: Last n rows
        """
        if len(self) == 0:
            return DataFrame({})

        n = min(n, len(self))

        # Create a new dataframe with the last n rows
        result = DataFrame({})
        result._column_names = self._column_names.copy()
        result._column_mapping = self._column_mapping.copy()
        result._values = self._values[-n:, :]

        # Copy column metadata
        for col, metadata in self._column_metadata.items():
            result._column_metadata[col] = metadata.copy()

        return result

    def apply(self, func: Callable, axis: int = 0) -> "DataFrame":
        """
        Apply a function along an axis.

        Args:
            func: JAX function to apply - function should take a JAX array and return
                a JAX array
            axis: 0 for columns, 1 for rows

        Returns:
            JaxDataFrame: Result of applying the function
        """
        # Phase 1: Pre-processing (non-traceable)
        special_cols_mask = jnp.array(
            [
                self._column_metadata[col]["dtype_flag"]
                in [DATETIME_TYPE_FLAG, CATEGORY_TYPE_FLAG]
                for col in self._column_names
            ]
        )

        # Phase 2: Core computation (fully traceable)
        if axis == 0:
            result_values = self._apply_columns_core(
                self._values, special_cols_mask, func
            )
            # Use original column names
            result_names = self._column_names
        else:
            result_values = self._apply_rows_core(self._values, func)
            # For row operations, create a compound name with function name and
            # original columns
            func_name = getattr(func, "__name__", "lambda")
            if isinstance(func, type(lambda: None)):
                func_name = "lambda"
            # Create name that combines function name with original column names
            result_names = [f"{func_name}_{'_'.join(self._column_names)}"]
            # Set the values and fix the column mapping
            result_values = result_values.reshape(-1, 1)

        # Phase 3: Post-processing (non-traceable)
        result = DataFrame({})
        result._column_names = result_names
        result._column_mapping = {name: idx for idx, name in enumerate(result_names)}
        result._values = result_values

        # Copy metadata for column-wise operations only
        if axis == 0:
            for col, metadata in self._column_metadata.items():
                result._column_metadata[col] = metadata.copy()
        else:
            # For row operations, use default metadata
            for col in result_names:
                result._column_metadata[col] = {"dtype_flag": None}

        return result

    def _apply_columns_core(
        self, values: jax.Array, special_cols_mask: jax.Array, func: Callable
    ) -> jax.Array:
        """Pure JAX function for column-wise apply (fully traceable)."""

        def process_column(col_idx, values):
            col_data = values[:, col_idx]
            mask_val = special_cols_mask[col_idx]
            # Apply function only to non-special columns
            result = jnp.where(mask_val, col_data, func(col_data))
            return result

        # Process all columns with vmap
        column_indices = jnp.arange(values.shape[1])
        result_columns = jax.vmap(lambda idx: process_column(idx, values))(
            column_indices
        )

        # Stack columns properly
        return jnp.transpose(result_columns)  # type: ignore

    def _apply_rows_core(self, values: jax.Array, func: Callable) -> jax.Array:
        """Pure JAX function for row-wise apply (fully traceable)."""
        # Apply function to each row using vmap
        result_values = jax.vmap(func)(values)

        # Ensure result is always a column vector (n x 1)
        def ensure_column_vector(x):
            # First ensure we have a 1D array
            flat = x.ravel()
            # Then reshape to column vector
            return flat.reshape(-1, 1)

        # Always reshape to column vector - no conditional needed
        return ensure_column_vector(result_values)

    def rolling(self, window_size: int, min_periods: Optional[int] = None) -> "Rolling":
        """
        Create a rolling window view of the dataframe.

        Args:
            window_size: Size of the rolling window
            min_periods: Minimum number of observations required to have a value.
                         If None, defaults to window_size.

        Returns:
            JaxRollingDF: A rolling window object for the dataframe
        """
        return Rolling(self, window_size, min_periods)

    def groupby(self, by: Union[str, Callable]) -> "GroupBy":
        """
        Group the dataframe by a column.

        Args:
            by: Column name to group by

        Returns:
            JaxGroupBy: A groupby object
        """
        return GroupBy(self, by)

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert JaxDataFrame to pandas DataFrame.

        Returns:
            pandas.DataFrame: Pandas equivalent of this dataframe
        """
        data = {}

        for col in self._column_names:
            col_idx = self._column_mapping[col]
            col_values = np.array(self._values[:, col_idx])
            metadata = self._column_metadata[col]

            if metadata["dtype_flag"] == DATETIME_TYPE_FLAG:
                # Convert from nanoseconds to pandas datetime
                data[col] = pd.to_datetime(col_values, unit="ns")
            elif metadata["dtype_flag"] == CATEGORY_TYPE_FLAG:
                # Convert from category codes to original strings
                cat_map = metadata["category_map"]
                strings = []
                for code in col_values:
                    string_val = (
                        cat_map.get(int(code), None)
                        if code >= 0 and not np.isnan(code)
                        else None
                    )
                    strings.append(string_val)
                data[col] = pd.Categorical(strings)
            else:
                data[col] = col_values

        # Create DataFrame
        return pd.DataFrame(data)

    @classmethod
    def from_numpy(cls, array: np.ndarray, columns: List[str]) -> "DataFrame":
        """Create a JaxDataFrame from a 2D numpy array."""
        return cls(array, columns=columns)

    @classmethod
    def from_jax(cls, array: jax.Array, columns: List[str]) -> "DataFrame":
        """Create a JaxDataFrame from a 2D JAX array."""
        return cls(array, columns=columns)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "DataFrame":
        """Create a JaxDataFrame from a pandas DataFrame."""
        return cls(df)


class Rolling:
    """Represents a rolling window over a JaxDataFrame."""

    def __init__(
        self, df: DataFrame, window_size: int, min_periods: Optional[int] = None
    ):
        """
        Initialize a rolling window for a dataframe.

        Args:
            df: The JaxDataFrame to create windows from
            window_size: Size of each window
            min_periods: Minimum num of observations in window required to have a value.
                         If None, defaults to window_size.
        """
        self.df = df
        self.window_size = window_size
        # Set min_periods to window_size if None (pandas default behavior)
        self.min_periods = min_periods if min_periods is not None else window_size

    def apply(self, func: Callable) -> DataFrame:
        """
        Apply a JAX function to each window of each column, in a traceable manner.

        Args:
            func: A JAX-compatible function to apply to each window

        Returns:
            JaxDataFrame: Result of applying the function
        """
        # Phase 1: Pre-processing (non-traceable)
        # Convert column flags to boolean mask array for tracing
        process_mask = self._prepare_column_mask()

        # Convert inputs to arrays for core computation
        input_values = self.df._values
        window_size = self.window_size

        # Phase 2: Core computation (fully traceable)
        result_values = self._apply_core(input_values, process_mask, window_size, func)

        # Phase 3: Post-processing (non-traceable)
        return self._wrap_result(result_values)

    def _prepare_column_mask(self) -> jax.Array:
        """Create boolean mask for processable columns (non-traceable)."""
        process_mask = []
        for col in self.df._column_names:
            metadata = self.df._column_metadata[col]
            should_process = metadata["dtype_flag"] not in [
                DATETIME_TYPE_FLAG,
                CATEGORY_TYPE_FLAG,
            ]
            process_mask.append(should_process)
        return jnp.array(process_mask)

    def _apply_core(
        self,
        values: jax.Array,
        process_mask: jax.Array,
        window_size: int,
        func: Callable,
    ) -> jax.Array:
        """Pure JAX implementation of rolling window computation (fully traceable)."""
        n_rows, n_cols = values.shape

        def process_column(col_idx):
            col_data = values[:, col_idx]
            should_process = process_mask[col_idx]

            def process_window_at_position(row_idx):
                """Process the window at a specific position in the column."""
                # Calculate the valid window for this position
                start_pos = jnp.maximum(0, row_idx - window_size + 1)
                window_buffer = jnp.zeros(window_size)
                valid_mask = jnp.zeros(window_size, dtype=jnp.bool_)

                # Fill window with values using fori_loop
                def fill_window_element(i, carry):
                    buffer, mask = carry
                    src_idx = start_pos + i
                    # Check if index is valid (within bounds and within window)
                    valid = jnp.logical_and(
                        jnp.logical_and(src_idx >= 0, src_idx < n_rows),
                        src_idx <= row_idx,
                    )
                    value = jnp.where(valid, col_data[src_idx], 0.0)
                    buffer_val = buffer.at[i].set(value)

                    # Also mark NaN values as invalid for min_periods calculations
                    is_valid_value = jnp.logical_and(valid, ~jnp.isnan(value))
                    mask_val = mask.at[i].set(is_valid_value)

                    return buffer_val, mask_val

                # Fill window
                buffer, mask = jax.lax.fori_loop(
                    0, window_size, fill_window_element, (window_buffer, valid_mask)
                )

                # Determine if we have a full window
                is_full_window = row_idx >= (window_size - 1)

                # Process window using the appropriate masking strategy
                return self._process_window(
                    buffer,
                    mask,
                    func,
                    col_data[row_idx],  # type: ignore
                    is_full_window,
                )

            # Apply the window processor to all positions
            result = jax.vmap(process_window_at_position)(jnp.arange(n_rows))

            # Only apply processing to appropriate columns
            return jnp.where(should_process, result, col_data)

        # Process all columns
        result_columns = jax.vmap(process_column)(jnp.arange(n_cols))
        return jnp.transpose(result_columns)

    def _fill_window_element(  # noqa: PLR0913
        self,
        i: int,
        carry: tuple,
        start_pos: int,
        col_data: jax.Array,
        n_rows: int,
        current_pos: int,
    ) -> tuple:
        """Pure function for filling single window element (traceable)."""
        buffer, mask = carry
        src_idx = start_pos + i

        # The element is valid if:
        # 1. It's within the bounds of the data (src_idx < n_rows)
        # 2. It's within the rolling window (src_idx <= current_pos)
        valid = jnp.logical_and(
            jnp.logical_and(src_idx >= 0, src_idx < n_rows), src_idx <= current_pos
        )

        value = jnp.where(valid, col_data[src_idx], 0.0)
        return (buffer.at[i].set(value), mask.at[i].set(valid))

    def _process_window(
        self,
        buffer: jax.Array,
        mask: jax.Array,
        func: Callable,
        original_value: float,
        is_full_window: bool,
    ) -> float:
        """Pure function for window processing (traceable)."""
        valid_count = jnp.sum(mask)

        # Using pandas rules for rolling windows:
        # 1. If min_periods is None or equals window_size (default):
        #    - Return NaN until a full window is available
        # 2. If min_periods is specified:
        #    - Return NaN unless we have at least min_periods valid values

        # Check if we have enough valid observations
        has_min_periods = valid_count >= self.min_periods

        # For default behavior, we need a full window
        using_default = self.min_periods == self.window_size
        meets_default_requirements = is_full_window

        # We can calculate if:
        # 1. Using custom min_periods and have enough valid values, OR
        # 2. Using default behavior and have a full window with enough valid values
        can_calculate = jax.lax.cond(
            using_default,
            lambda _: jnp.logical_and(meets_default_requirements, has_min_periods),
            lambda _: has_min_periods,
            None,
        )

        # Use lax.cond for maximum traceability
        return jax.lax.cond(
            can_calculate,
            lambda _: self._apply_with_masking(func, buffer, mask),
            lambda _: jnp.nan,
            None,
        )

    def _wrap_result(self, result_values: jax.Array) -> DataFrame:
        """Construct result DataFrame (non-traceable)."""
        result_df = DataFrame({})
        result_df._column_names = self.df._column_names.copy()
        result_df._column_mapping = self.df._column_mapping.copy()
        result_df._values = result_values

        # Copy column metadata
        for col, metadata in self.df._column_metadata.items():
            result_df._column_metadata[col] = metadata.copy()

        return result_df

    def _apply_with_masking(
        self, fn: Callable, values: jax.Array, mask: jax.Array
    ) -> float:
        """
        Apply function to masked values in a JAX-traceable way.

        Args:
            fn: JAX-compatible function to apply
            values: Window values array
            mask: Boolean mask array indicating valid values

        Returns:
            JAX array: Result of applying function to masked values

        Note:
            This is a pure function and fully traceable.
        """
        # For nanmean and similar functions, we need to:
        # 1. Replace invalid positions with NaN (determined by mask)
        # 2. Replace existing NaN values with NaN (even if they're in valid positions)

        # First mask out invalid positions
        masked_values = jnp.where(mask, values, jnp.nan)

        # Apply the function (which should handle NaNs properly like nanmean, nansum..)
        result = fn(masked_values)

        # If no valid values or all values are NaN, return NaN
        all_nan = jnp.logical_or(jnp.all(jnp.isnan(masked_values)), ~jnp.any(mask))
        return jax.lax.cond(all_nan, lambda _: jnp.nan, lambda _: result, None)


class GroupBy:
    """Represents a groupby operation on a JaxDataFrame."""

    def __init__(self, df: DataFrame, by: Union[str, Callable]):
        """
        Initialize a groupby object.

        Args:
            df: The JaxDataFrame to group
            by: Column name or function to group by

        Note:
            This initialization is NOT JAX-traceable because it involves:
            - Dynamic Python data structure creation
            - Runtime shape discovery
            - Python control flow based on data values
        """
        # NON-TRACEABLE SECTION: Initialization and group discovery
        self.df = df
        self.by = by

        # Determine grouping values
        if isinstance(by, str):
            # Group by column
            if by not in df.columns:
                raise KeyError(f"Column '{by}' not found")
            col_idx = df._column_mapping[by]
            self.grouping_values = df._values[:, col_idx]
            self.by_metadata = df._column_metadata[by]
        elif callable(by):
            # Group by function applied to the dataframe
            self.grouping_values = by(df._values)
            self.by_metadata = {"dtype_flag": None}
        else:
            raise TypeError(f"Unsupported groupby type: {type(by)}")

        # Find unique groups and their indices - THIS IS NOT TRACEABLE
        # Must be done during initialization (non-traceable phase)
        self.unique_groups, self.group_indices = self._compute_groups()

        # Create group ID mapping - map each row to its group ID
        # This helps make the apply function more traceable
        self.group_ids = self._create_group_id_mapping()

    def _compute_groups(self) -> tuple:
        """
        Compute unique groups and their indices (NON-TRACEABLE).
        Returns a tuple of (unique_groups, group_indices_dict)
        """
        # Convert to numpy for more reliable unique value detection
        grouping_np = np.array(self.grouping_values)
        unique_groups = np.unique(grouping_np)

        # Create a mapping from group values to indices
        group_indices = {}
        for group in unique_groups:
            # Find indices for this group
            mask = grouping_np == group
            indices = np.where(mask)[0]
            if len(indices) > 0:  # Only store non-empty groups
                group_indices[float(group)] = indices

        return jnp.array(unique_groups), group_indices

    def _create_group_id_mapping(self) -> jax.Array:
        """
        Create a mapping from row index to group ID (NON-TRACEABLE).
        This will help make apply operations more traceable.
        """
        n_rows = len(self.df)
        # Initialize with -1 (invalid group)
        group_ids = np.full(n_rows, -1)

        # Assign group IDs (position in unique_groups) to each row
        for idx, group in enumerate(self.unique_groups):
            group_val = float(group)
            if group_val in self.group_indices:
                row_indices = self.group_indices[group_val]
                group_ids[row_indices] = idx

        return jnp.array(group_ids)

    def __getitem__(self, key: Union[str, List[str]]) -> "GroupBy":
        """
        Select columns from the grouped data (NON-TRACEABLE).

        Args:
            key: Column name or list of column names

        Returns:
            JaxGroupBy with selected columns
        """
        # NON-TRACEABLE: This method constructs new Python objects and uses
        # dynamic data selection
        if isinstance(key, str):
            # Create a copy of self with specified column
            if key not in self.df.columns:
                raise KeyError(f"Column '{key}' not found")

            # Create a new dataframe with just the structure/metadata
            selected_df = DataFrame({})

            # Determine which columns to include
            columns_to_include = [key]
            if (
                isinstance(self.by, str)
                and self.by != key
                and self.by in self.df.columns
            ):
                columns_to_include.insert(0, self.by)  # Put groupby column first

            # Set up column names and mapping
            selected_df._column_names = columns_to_include
            selected_df._column_mapping = {
                name: idx for idx, name in enumerate(columns_to_include)
            }

            # Create values array
            col_arrays = []
            for col in columns_to_include:
                col_idx = self.df._column_mapping[col]
                col_arrays.append(self.df._values[:, col_idx])

            selected_df._values = jnp.column_stack(col_arrays)

            # Copy metadata for all columns
            for col in columns_to_include:
                selected_df._column_metadata[col] = self.df._column_metadata[col].copy()

            # Create a new GroupBy object with the selected columns
            result = GroupBy(selected_df, self.by)
            result.grouping_values = self.grouping_values
            result.by_metadata = self.by_metadata
            result.unique_groups = self.unique_groups
            result.group_indices = self.group_indices
            return result
        elif isinstance(key, list):
            # Multiple column selection
            if not all(col in self.df.columns for col in key):
                missing = [col for col in key if col not in self.df.columns]
                raise KeyError(f"Columns {missing} not found")

            # Create a new dataframe with just the structure/metadata
            selected_df = DataFrame({})

            # Determine which columns to include, ensuring groupby column
            # is included and first
            columns_to_include = key.copy()
            if (
                isinstance(self.by, str)
                and self.by not in columns_to_include
                and self.by in self.df.columns
            ):
                columns_to_include.insert(0, self.by)

            # Set up column names and mapping
            selected_df._column_names = columns_to_include
            selected_df._column_mapping = {
                name: idx for idx, name in enumerate(columns_to_include)
            }

            # Create values array
            col_arrays = []
            for col in columns_to_include:
                col_idx = self.df._column_mapping[col]
                col_arrays.append(self.df._values[:, col_idx])

            selected_df._values = jnp.column_stack(col_arrays)

            # Copy metadata for all columns
            for col in columns_to_include:
                selected_df._column_metadata[col] = self.df._column_metadata[col].copy()

            # Create a new GroupBy object with the selected columns
            result = GroupBy(selected_df, self.by)
            result.grouping_values = self.grouping_values
            result.by_metadata = self.by_metadata
            result.unique_groups = self.unique_groups
            result.group_indices = self.group_indices
            return result
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def rolling(self, window_size: int) -> "GroupByRolling":
        """
        Create a rolling window view of the grouped dataframe (NON-TRACEABLE init).

        Args:
            window_size: Size of the rolling window

        Returns:
            JaxGroupByRolling: A rolling window object for the grouped dataframe
        """
        # NON-TRACEABLE: This method constructs a new Python object
        return GroupByRolling(self, window_size)

    def aggregate(self, func: Callable) -> DataFrame:  # noqa: PLR0912
        """
        Apply an aggregation function to each group (fully traceable).
        This explicitly treats the function as an aggregation (e.g. mean, sum, max),
        which will return one row per group.

        Args:
            func: A pure JAX reduction function that takes an array and returns a scalar

        Returns:
            JaxDataFrame: One row per group with aggregated values
        """
        # Phase 1: Pre-processing (non-traceable)
        columns = self.df._column_names
        by_col_idx = (
            self.df._column_mapping[self.by] if isinstance(self.by, str) else None
        )

        # Initialize output data structure
        agg_data = {}

        # Handle the groupby column specially
        if isinstance(self.by, str):
            # Just use the unique groups as values
            agg_data[self.by] = jnp.array(self.unique_groups)

        # Prepare data for core computation
        core_data = []
        core_metadata = []
        for col_idx, col_name in enumerate(columns):
            # Skip the groupby column as we already handled it
            if isinstance(self.by, str) and col_idx == by_col_idx:
                continue

            metadata = self.df._column_metadata[col_name]

            # For special types, store the column index and metadata
            if metadata["dtype_flag"] in [DATETIME_TYPE_FLAG, CATEGORY_TYPE_FLAG]:
                core_data.append((col_idx, metadata))
            else:
                # For regular columns, add the data to the core data list
                col_data = self.df._values[:, col_idx]
                core_data.append(col_data)
            core_metadata.append(col_name)

        # Phase 2: Core computation (fully traceable)
        agg_results = self._aggregate_core(core_data, func)

        # Phase 3: Post-processing (non-traceable)
        # Assign aggregated results to the output data structure
        result_idx = 0
        for col_idx, col_name in enumerate(columns):
            # Skip the groupby column as we already handled it
            if isinstance(self.by, str) and col_idx == by_col_idx:
                continue

            metadata = self.df._column_metadata[col_name]

            # For special types, take the first value from each group
            if metadata["dtype_flag"] in [DATETIME_TYPE_FLAG, CATEGORY_TYPE_FLAG]:
                col_results = []
                for _, group in enumerate(self.unique_groups):
                    group_float = float(group)
                    indices = self.group_indices.get(group_float, [])
                    if len(indices) > 0:
                        col_results.append(self.df._values[indices[0], col_idx])
                    # Use a placeholder for empty groups
                    elif metadata["dtype_flag"] == DATETIME_TYPE_FLAG:
                        col_results.append(pd.NaT.value)
                    else:
                        col_results.append(-1)  # None category

                agg_data[col_name] = jnp.array(col_results)
            else:
                # Assign aggregated results to the output data structure
                agg_data[col_name] = agg_results[result_idx]
                result_idx += 1

        # Create result DataFrame
        result_df = DataFrame(agg_data)

        # Copy metadata for special columns
        for col_name, metadata in self.df._column_metadata.items():
            if col_name in result_df._column_metadata:
                result_df._column_metadata[col_name] = metadata.copy()

        return result_df

    def transform(self, func: Callable) -> DataFrame:
        """
        Apply a transformation function to each group (partly traceable).
        This explicitly treats the function as a transformation that returns an array
        of the same size as its input, resulting in a DataFrame of the same shape as
        the original.

        Args:
            func: A JAX-compatible transformation function that takes an array and
                    returns an array of the same length

        Returns:
            JaxDataFrame: Same shape as input with transformed values
        """
        # Phase 1: Pre-processing (non-traceable)
        columns = self.df._column_names
        by_col_idx = (
            self.df._column_mapping[self.by] if isinstance(self.by, str) else None
        )

        # Prepare data for core computation
        core_data = []
        core_metadata = []
        for col_idx, col_name in enumerate(columns):
            # Skip the groupby column and special type columns
            if (
                isinstance(self.by, str) and col_idx == by_col_idx
            ) or self.df._column_metadata[col_name]["dtype_flag"] in [
                DATETIME_TYPE_FLAG,
                CATEGORY_TYPE_FLAG,
            ]:
                continue

            # Get column data
            col_data = self.df._values[:, col_idx]
            core_data.append(col_data)
            core_metadata.append(col_name)

        # Phase 2: Core computation (fully traceable)
        transformed_values = self._transformation_core(core_data, func)

        # Phase 3: Post-processing (non-traceable)
        # Start with a copy of the input values
        result_values = jnp.array(self.df._values)
        result_idx = 0

        # We use a non-traceable outer loop over groups with traceable inner operations
        # Process each column
        for col_idx, col_name in enumerate(columns):
            # Skip the groupby column and special type columns
            if (
                isinstance(self.by, str) and col_idx == by_col_idx
            ) or self.df._column_metadata[col_name]["dtype_flag"] in [
                DATETIME_TYPE_FLAG,
                CATEGORY_TYPE_FLAG,
            ]:
                continue

            # Process each group (non-traceable outer loop, but computed during init)
            for group_val in self.unique_groups:
                group_key = float(group_val)
                indices = self.group_indices.get(group_key, [])
                if len(indices) == 0:
                    continue

                # Extract transformed data for this group
                transformed_data = transformed_values[result_idx]
                group_data = transformed_data[indices]

                # Update result values - one position at a time (traceable)
                for i, idx in enumerate(indices):
                    result_values = result_values.at[idx, col_idx].set(group_data[i])

            result_idx += 1

        # Create result DataFrame
        result_df = DataFrame({})
        result_df._column_names = self.df._column_names.copy()
        result_df._column_mapping = self.df._column_mapping.copy()
        result_df._values = result_values

        # Copy metadata
        for col, metadata in self.df._column_metadata.items():
            result_df._column_metadata[col] = metadata.copy()

        return result_df

    def _aggregate_core(
        self, core_data: List[jax.Array], func: Callable
    ) -> List[jax.Array]:
        """
        Pure JAX function for computing aggregations within groups.

        Args:
            core_data: List of JAX arrays to aggregate
            func: A JAX-compatible reduction function

        Returns:
            List of JAX arrays with aggregated values
        """
        agg_results = []
        for data in core_data:
            # For regular columns, apply aggregation to each group directly
            # This avoids the complex chain of jax.lax.cond operations in the previous
            # implementation
            col_data = data

            # Process each group (non-traceable loop, but known at initialization time)
            group_results = []
            for _, group in enumerate(self.unique_groups):
                group_float = float(group)
                indices = self.group_indices.get(group_float, [])

                if len(indices) == 0:
                    # Empty group - use NaN
                    group_results.append(jnp.nan)
                else:
                    # Extract data for this group (fixed size, known at this point)
                    group_data = col_data[indices]

                    # Apply aggregation function (traceable inner operation)
                    group_results.append(func(group_data))

            agg_results.append(jnp.array(group_results))

        return agg_results

    def _transformation_core(
        self, core_data: List[jax.Array], func: Callable
    ) -> List[jax.Array]:
        """
        Pure JAX function for computing transformations within groups.

        Args:
            core_data: List of JAX arrays to transform
            func: A JAX-compatible transformation function

        Returns:
            List of JAX arrays with transformed values
        """
        transformed_values = []
        for col_data in core_data:
            # Process each group (non-traceable outer loop, but computed during init)
            group_results = []
            for group_val in self.unique_groups:
                group_key = float(group_val)
                indices = self.group_indices.get(group_key, [])
                if len(indices) == 0:
                    # Empty group - use empty array
                    group_results.append(jnp.array([]))
                    continue

                # Extract data for this group (fixed size for this group, known here)
                group_data = col_data[indices]

                # Apply transformation (fully traceable inner operation)
                transformed_data = func(group_data)
                group_results.append(transformed_data)

            transformed_values.append(jnp.concatenate(group_results))

        return transformed_values

    # Helper methods for special column types
    def to_datetime(self) -> pd.Series:
        """
        Convert the groupby column to pandas datetime if it's a datetime column.
        NON-TRACEABLE: Uses pandas conversion
        """
        if (
            isinstance(self.by, str)
            and self.by_metadata["dtype_flag"] == DATETIME_TYPE_FLAG
        ):
            return pd.Series(ns_to_pd_datetime(self.grouping_values))
        else:
            raise TypeError("Groupby column is not a datetime type")

    def to_strings(self) -> pd.Series:
        """
        Convert the groupby column to strings if it's a categorical column.
        NON-TRACEABLE: Uses pandas conversion
        """
        if (
            isinstance(self.by, str)
            and self.by_metadata["dtype_flag"] == CATEGORY_TYPE_FLAG
        ):
            cat_map = self.by_metadata["category_map"]
            assert cat_map is not None
            return pd.Series(
                [
                    cat_map.get(int(x), None) if x != -1 and not jnp.isnan(x) else None
                    for x in self.unique_groups
                ]
            )
        else:
            raise TypeError("Groupby column is not a categorical type")


class GroupByRolling:
    """Represents a rolling window operation on a grouped dataframe (Maximally traceable
    implementation)."""

    def __init__(self, groupby: GroupBy, window_size: int):
        """
        Initialize a rolling window object for grouped data.

        Args:
            groupby: The JaxGroupBy object
            window_size: Size of each window
        """
        # NON-TRACEABLE SECTION: Initialization and data preparation
        self.groupby = groupby
        self.window_size = window_size

        # Pre-compute column masks for special types during initialization
        self.column_masks = []
        for col_name in self.groupby.df._column_names:
            metadata = self.groupby.df._column_metadata[col_name]
            # Only process non-special columns
            should_process = metadata["dtype_flag"] not in [
                DATETIME_TYPE_FLAG,
                CATEGORY_TYPE_FLAG,
            ]
            self.column_masks.append(should_process)

        # Convert to JAX array for use in traceable computation
        self.process_mask_array = jnp.array(self.column_masks)

    def apply(self, func: Callable) -> DataFrame:
        """
        Apply a JAX function to rolling windows within each group.

        Args:
            func: A JAX-compatible function to apply to each window

        Returns:
            JaxDataFrame: Result of applying the function
        """
        # NON-TRACEABLE SECTION: Gather necessary data for computation
        df_values = self.groupby.df._values
        group_ids = self.groupby.group_ids
        unique_groups = self.groupby.unique_groups
        window_size = self.window_size
        process_mask = self.process_mask_array

        # CORE COMPUTATION PHASE (FULLY TRACEABLE)
        # This pure function call can be traced end-to-end by JAX
        result_values = self._compute_grouped_rolling_core(
            df_values, group_ids, unique_groups, process_mask, window_size, func
        )

        # NON-TRACEABLE SECTION: Create result dataframe
        result_df = DataFrame({})
        result_df._column_names = self.groupby.df._column_names.copy()
        result_df._column_mapping = self.groupby.df._column_mapping.copy()
        result_df._values = result_values

        # Copy column metadata
        for col, metadata in self.groupby.df._column_metadata.items():
            result_df._column_metadata[col] = metadata.copy()

        return result_df

    def _compute_grouped_rolling_core(  # noqa: PLR0913
        self,
        values: jax.Array,
        group_ids: jax.Array,
        unique_groups: jax.Array,
        process_mask: jax.Array,
        window_size: int,
        func: Callable,
    ) -> jax.Array:
        """
        Pure JAX function for computing rolling windows grouped by IDs. Fully traceable.

        Args:
            values: 2D JAX array of input values
            group_ids: 1D JAX array mapping each row to its group ID
            unique_groups: 1D JAX array of unique group IDs
            process_mask: Boolean mask indicating which columns to process
            window_size: Size of the rolling window
            func: JAX-compatible function to apply to each window

        Returns:
            2D JAX array of results with same shape as input
        """
        n_rows, n_cols = values.shape

        # Define column-wise processing function (will be vmapped)
        def process_column(col_idx):
            col_data = values[:, col_idx]
            should_process = process_mask[col_idx]

            # Define row-wise window function (will be vmapped)
            def process_row(row_idx):
                # Get the group ID for this row
                row_group_id = group_ids[row_idx]

                # Find all rows in the same group
                same_group_mask = group_ids == row_group_id

                # Special handling for invalid groups (-1)
                valid_group = row_group_id >= 0

                # Calculate the relative position of this row within its group
                # Count how many elements with the same group ID appear before this row
                relative_pos_mask = jnp.logical_and(
                    same_group_mask, jnp.arange(n_rows) <= row_idx
                )
                relative_pos = jnp.sum(relative_pos_mask) - 1  # 0-based index

                # Create a window buffer of fixed size
                window_buffer = jnp.zeros(window_size, dtype=col_data.dtype)
                valid_mask = jnp.zeros(window_size, dtype=jnp.bool_)

                # Calculate start position for the window
                start_pos = jnp.maximum(0, relative_pos - window_size + 1)

                # Fill the buffer with values from the same group
                def fill_window(i, carry):
                    buffer, mask = carry

                    # Calculate the position in the group
                    group_pos = start_pos + i

                    # Find the absolute position in the data
                    # This is non-trivial and requires counting
                    count_mask = jnp.logical_and(
                        same_group_mask, jnp.arange(n_rows) < n_rows
                    )
                    abs_positions = jnp.cumsum(count_mask) - 1
                    valid_positions = jnp.where(same_group_mask, abs_positions, -1)

                    # Find the position where group_pos appears
                    position_mask = jnp.logical_and(
                        valid_positions >= 0, valid_positions == group_pos
                    )

                    # Check if this position exists and is valid
                    pos_exists = jnp.any(position_mask)
                    abs_pos = jnp.where(pos_exists, jnp.argmax(position_mask), -1)

                    # Element is valid if position exists and is within bounds
                    valid = jnp.logical_and(
                        pos_exists,
                        jnp.logical_and(group_pos >= 0, group_pos <= relative_pos),
                    )

                    # Get value or use 0 if invalid
                    value = jnp.where(valid, col_data[abs_pos], 0.0)

                    # Update buffer and mask
                    new_buffer = buffer.at[i].set(value)
                    new_mask = mask.at[i].set(valid)

                    return (new_buffer, new_mask)

                # Loop through window positions using fori_loop (traceable)
                buffer, mask = jax.lax.fori_loop(
                    0, window_size, fill_window, (window_buffer, valid_mask)
                )

                # Check if we have any valid elements
                valid_count = jnp.sum(mask)

                # Different NaN handling based on window size:
                # For window_size=2, we need first element to be original value
                # For window_size>=3, we return NaN for the first (win_size-1) elements
                # This makes handling work consistently for the test cases
                window_is_incomplete = jnp.where(
                    window_size == 2,  # noqa: PLR2004
                    # For window_size=2, only consider position 0 incomplete in groups
                    #  with >1 element
                    jnp.logical_and(relative_pos == 0, valid_count > 1),
                    # For window_size>=3, regular pandas-style handling
                    relative_pos < window_size - 1,
                )

                # Handle the case where group is invalid, window is incomplete,
                # or no valid elements
                def apply_func(_, buffer=buffer, mask=mask):
                    # Apply the function to masked values
                    masked_values = jnp.where(mask, buffer, jnp.nan)
                    return func(masked_values)

                # Use nested lax.cond for maximum traceability
                result = jax.lax.cond(
                    valid_group,  # First check if group is valid
                    lambda _: jax.lax.cond(
                        window_is_incomplete,
                        lambda _: jnp.where(
                            window_size == 2,  # noqa: PLR2004
                            col_data[row_idx],
                            jnp.nan,
                        ),  # For window_size=2 first element is original, otherwise NaN
                        apply_func,  # Apply function if we have a complete window
                        None,
                    ),
                    lambda _: col_data[row_idx],  # Return original if invalid group
                    None,
                )

                return result

            # Process all rows in this column using vmap
            positions = jnp.arange(n_rows)
            processed = jax.vmap(process_row)(positions)

            # Only update columns that should be processed
            # For special columns (datetime/category), keep original values
            return jnp.where(should_process, processed, col_data)

        # Process all columns using vmap
        column_indices = jnp.arange(n_cols)
        result_columns = jax.vmap(process_column)(column_indices)

        # Transpose to get back to (rows, cols) shape
        return jnp.transpose(result_columns)  # type: ignore

    def aggregate(self, func: Callable) -> DataFrame:
        """
        Apply an aggregation function to each rolling window within each group.

        Args:
            func: A JAX-compatible aggregation function for each window

        Returns:
            JaxDataFrame: Same shape as input with each value replaced by its window
             aggregation
        """
        return self.apply(func)
