from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import register_pytree_node

# Constants for special data types
DATETIME_TYPE_FLAG = "datetime"
CATEGORY_TYPE_FLAG = "category"


class DataFrame:
    """
    DataFrame implementation that uses a full Cartesian product of indices for efficient
    and fully traceable operations in JAX.

    This implementation allows specifying any columns as index columns, and automatically
    handles a Cartesian product of all combinations of these indices.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, Dict],
        index_columns: Optional[List[str]] = None,
        fill_value: float = np.nan,
    ):
        """
        Initialize CartesianDataFrame from a pandas DataFrame.

        Args:
            df: Pandas DataFrame
            index_columns: Columns to use as indices (if None, use DataFrame index)
            fill_value: Value to use for missing combinations in the Cartesian product
        """
        # Initialize storage
        self._values = None  # JAX array of values
        self._column_names = []  # Column names
        self._column_mapping = {}  # Maps column name to index
        self._column_dtypes = {}  # Original pandas dtypes
        self._column_metadata = {}  # Column metadata

        # Index information
        self._index_columns = []  # Names of index columns
        self._index_values = {}  # Maps index column to unique values
        self._index_mappings = {}  # Maps index values to integer indices
        self._index_sizes = {}  # Number of unique values for each index

        if isinstance(df, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame.from_dict(df)
        # Process the input DataFrame
        self._process_dataframe(df, index_columns, fill_value)

    def _process_dataframe(
        self, df: pd.DataFrame, index_columns: Optional[List[str]], fill_value: float
    ):
        """Process the input DataFrame and create the Cartesian grid."""
        # Step 1: Handle the index columns determination
        original_df = df.copy()
        has_multi_index = isinstance(original_df.index, pd.MultiIndex)

        # Handle index_columns specification
        if index_columns is None:
            if has_multi_index:
                # For MultiIndex, use all level names
                index_columns = list(original_df.index.names)
            elif original_df.index.name:
                # For named index, use it
                index_columns = [original_df.index.name]
            else:
                # For unnamed RangeIndex, don't use any index columns
                # This fixes the issue with test_arithmetic.py
                index_columns = []

        # Store index column names
        self._index_columns = index_columns

        # If we have no index columns, create a simple default index
        # This ensures we always have at least one index for the Cartesian grid
        if not index_columns:
            # Create a dummy index that just enumerates rows
            default_index_name = "_row"
            self._index_columns = [default_index_name]
            self._index_values = {default_index_name: list(range(len(original_df)))}
            self._index_mappings = {
                default_index_name: {i: i for i in range(len(original_df))}
            }
            self._index_sizes = {default_index_name: len(original_df)}
            self._column_metadata[default_index_name] = {"dtype_flag": None}

            # Then process only data columns
            working_df = original_df.copy()
            data_columns = list(working_df.columns)

            # Create a simple Cartesian index with just row numbers
            idx_product = pd.Index(
                self._index_values[default_index_name], name=default_index_name
            )

        else:
            # Step 2: Make sure the index columns are available as regular columns
            working_df = original_df.copy()
            if has_multi_index and any(
                col not in working_df.columns for col in index_columns
            ):
                # Reset the index to get index levels as columns
                working_df = working_df.reset_index()
            elif not has_multi_index and index_columns[0] not in working_df.columns:
                # Reset the index to get the index as a column
                working_df = working_df.reset_index()

            # Step 3: Process each index column to extract unique values
            for col in index_columns:
                # Get unique values
                unique_values = working_df[col].unique()

                # Store the column dtype for later use
                column_dtype = working_df[col].dtype

                # Store metadata and mappings based on data type
                if pd.api.types.is_categorical_dtype(
                    column_dtype
                ) or pd.api.types.is_string_dtype(column_dtype):
                    # For categorical or string, create a category mapping
                    categories = (
                        pd.Series(unique_values).astype("category").cat.categories
                    )
                    self._column_metadata[col] = {
                        "dtype_flag": CATEGORY_TYPE_FLAG,
                        "category_map": {i: cat for i, cat in enumerate(categories)},
                    }
                    # Store original values first
                    self._index_values[col] = list(categories)
                    self._index_mappings[col] = {
                        val: i for i, val in enumerate(categories)
                    }

                elif pd.api.types.is_datetime64_any_dtype(column_dtype):
                    # For datetime, keep original values for reindexing but store nanosecond representation
                    self._column_metadata[col] = {"dtype_flag": DATETIME_TYPE_FLAG}
                    # Store original datetime values
                    self._index_values[col] = pd.DatetimeIndex(
                        sorted(unique_values)
                    ).to_pydatetime()
                    # Create mapping from datetime to position
                    self._index_mappings[col] = {
                        val: i for i, val in enumerate(sorted(unique_values))
                    }

                else:
                    # For regular numeric columns
                    self._column_metadata[col] = {"dtype_flag": None}
                    # Store sorted values
                    self._index_values[col] = sorted(unique_values)
                    self._index_mappings[col] = {
                        val: i for i, val in enumerate(sorted(unique_values))
                    }

                # Store the number of unique values
                self._index_sizes[col] = len(unique_values)

            # Step 4: Create a MultiIndex with the Cartesian product of all index column values
            if len(index_columns) > 1:
                # Create a product of all index levels
                idx_product = pd.MultiIndex.from_product(
                    [self._index_values[col] for col in index_columns],
                    names=index_columns,
                )
            else:
                # For single index
                idx_product = pd.Index(
                    self._index_values[index_columns[0]], name=index_columns[0]
                )

            # Step 5: Create data columns (non-index columns)
            data_columns = [
                col for col in working_df.columns if col not in index_columns
            ]

        # Step 6: Reindex the DataFrame with the Cartesian product
        # Set up DataFrame with proper index
        if index_columns:
            if has_multi_index or len(index_columns) > 1:
                # Reindex with MultiIndex
                indexed_df = working_df.set_index(index_columns)
            else:
                # Reindex with simple Index
                indexed_df = working_df.set_index(index_columns[0])

            # Reindex to ensure all combinations exist
            cart_df = indexed_df[data_columns].reindex(idx_product)
        else:
            # No index columns to reindex on, keep the original data
            cart_df = working_df[data_columns]

        # Fill missing values
        cart_df = cart_df.fillna(fill_value)

        # Step 7: Store data columns information
        self._column_names = data_columns
        self._column_mapping = {name: i for i, name in enumerate(data_columns)}

        # Step 8: Process data columns
        data_arrays = []
        for col in data_columns:
            dtype = cart_df[col].dtype
            self._column_dtypes[col] = dtype

            # Handle special types
            if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_string_dtype(
                dtype
            ):
                # Convert to categorical integers
                categories = cart_df[col].astype("category").cat.categories
                codes = cart_df[col].astype("category").cat.codes.values
                data_arrays.append(codes)
                self._column_metadata[col] = {
                    "dtype_flag": CATEGORY_TYPE_FLAG,
                    "category_map": {i: cat for i, cat in enumerate(categories)},
                }

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                # Convert to nanoseconds since epoch
                values = pd.DatetimeIndex(cart_df[col]).astype(np.int64).values
                data_arrays.append(values)
                self._column_metadata[col] = {"dtype_flag": DATETIME_TYPE_FLAG}

            else:
                # Regular numeric column
                data_arrays.append(cart_df[col].values)
                self._column_metadata[col] = {"dtype_flag": None}

        # Step 9: Convert to JAX array
        if data_arrays:
            self._values = jnp.column_stack(
                [jnp.array(arr, dtype=jnp.float32) for arr in data_arrays]
            )
        else:
            # Handle empty DataFrame case
            self._values = jnp.zeros((len(cart_df), 0), dtype=jnp.float32)

        # Step 10: Update index value mappings to include integer representations for special types
        for col in self._index_columns:
            if (
                col in self._column_metadata
                and self._column_metadata[col]["dtype_flag"] == DATETIME_TYPE_FLAG
            ):
                # Convert index values to nanosecond representation for internal use
                self._index_values[col] = (
                    pd.DatetimeIndex(self._index_values[col]).astype(np.int64).tolist()
                )
            # We don't need to convert categorical values here as we already have the integer mappings

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the DataFrame (rows, columns)."""
        return self._values.shape

    @property
    def columns(self) -> List[str]:
        """Return the column names."""
        return self._column_names.copy()

    @property
    def index_columns(self) -> List[str]:
        """Return the index column names."""
        return self._index_columns.copy()

    def get_index_values(self, column: str) -> List:
        """Return the unique values for an index column."""
        if column not in self._index_columns:
            raise ValueError(f"Column '{column}' is not an index column")
        return self._index_values[column]

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._values.shape[0] if self._values is not None else 0

    def __getitem__(self, key: Union[str, List[str]]) -> Union[jax.Array, "DataFrame"]:
        """Column access."""
        if isinstance(key, str):
            if key not in self._column_mapping:
                raise KeyError(f"Column '{key}' not found")

            col_idx = self._column_mapping[key]
            return self._values[:, col_idx]

        elif isinstance(key, list):
            if not all(k in self._column_mapping for k in key):
                missing = [k for k in key if k not in self._column_mapping]
                raise KeyError(f"Columns {missing} not found")

            # Create a new CartesianDataFrame with the selected columns
            result = DataFrame.__new__(DataFrame)

            # Copy index information
            result._index_columns = self._index_columns.copy()
            result._index_values = {k: v.copy() for k, v in self._index_values.items()}
            result._index_mappings = {
                k: v.copy() for k, v in self._index_mappings.items()
            }
            result._index_sizes = self._index_sizes.copy()

            # Set up column information
            result._column_names = key.copy()
            result._column_mapping = {name: i for i, name in enumerate(key)}
            result._column_dtypes = {
                k: self._column_dtypes[k] for k in key if k in self._column_dtypes
            }
            result._column_metadata = {
                k: self._column_metadata[k].copy()
                for k in key
                if k in self._column_metadata
            }

            # Copy values for the selected columns
            col_indices = [self._column_mapping[k] for k in key]
            result._values = self._values[:, col_indices]

            return result

        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key: str, value: Any):
        """
        Set column values or add a new column.

        Args:
            key: Column name to set
            value: Values to assign to the column
        """
        if key in self._column_mapping:
            # Update existing column
            col_idx = self._column_mapping[key]

            # Convert value to a JAX array
            value_array = self._prepare_values_for_assignment(value)

            # Update the values
            self._values = self._values.at[:, col_idx].set(value_array)
        else:
            # Add a new column
            self.add_column(key, value)

    def _prepare_values_for_assignment(self, value: Any) -> jax.Array:
        """
        Convert a value to a JAX array suitable for assignment.

        Args:
            value: Value to convert (scalar, list, numpy array, or JAX array)

        Returns:
            JAX array with appropriate shape

        Raises:
            ValueError: If the length of the value doesn't match the DataFrame length
            TypeError: If the value type is not supported
        """
        if isinstance(value, (int, float, bool)):
            # Broadcast scalar to all rows
            return jnp.full(len(self), value, dtype=jnp.float32)
        elif isinstance(value, (list, np.ndarray)):
            if len(value) != len(self):
                raise ValueError(
                    f"Length of values ({len(value)}) does not match length of dataframe ({len(self)})"
                )
            return jnp.array(value, dtype=jnp.float32)
        elif isinstance(value, jax.Array):
            if value.shape[0] != len(self):
                raise ValueError(
                    f"Length of values ({value.shape[0]}) does not match length of dataframe ({len(self)})"
                )
            return value
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")

    def add_column(self, col_name: str, value: Any):
        """
        Add a new column to the DataFrame.

        Args:
            col_name: Name of the new column
            value: Values for the new column (scalar, list, numpy array, or JAX array)

        Raises:
            KeyError: If the column name already exists
        """
        if col_name in self._column_mapping:
            raise KeyError(f"Column '{col_name}' already exists")

        # Convert value to a JAX array
        value_array = self._prepare_values_for_assignment(value)

        # Update the column information
        new_idx = len(self._column_names)
        self._column_names.append(col_name)
        self._column_mapping[col_name] = new_idx

        # Default metadata for the new column
        self._column_dtypes[col_name] = np.float32
        self._column_metadata[col_name] = {"dtype_flag": None}

        # Add the new column to the values array
        if self._values is None or self._values.size == 0:
            # Handle empty DataFrame
            self._values = value_array.reshape(-1, 1)
        else:
            # Stack the new column
            self._values = jnp.column_stack([self._values, value_array])

    def _get_index_position(self, **indices) -> int:
        """
        Convert index values to a linear row index.

        In our Cartesian grid, the rows are arranged in order of the Cartesian product
        of all index columns.

        Args:
            **indices: Index values for each index column

        Returns:
            Linear row index
        """
        # Check that all index columns are provided
        for col in self._index_columns:
            if col not in indices:
                raise ValueError(f"Index column '{col}' not provided")

        # Convert index values to integer indices
        index_indices = []
        for col in self._index_columns:
            val = indices[col]
            if val not in self._index_mappings[col]:
                raise ValueError(f"Value '{val}' not found in index column '{col}'")
            idx = self._index_mappings[col][val]
            index_indices.append(idx)

        # Compute linear index
        # For a Cartesian product, the linear index is computed as:
        # idx = idx_n + size_n * (idx_{n-1} + size_{n-1} * (... + size_1 * idx_0))
        # This is essentially treating the indices as digits in a mixed-radix number system
        linear_idx = 0
        for i, col in enumerate(reversed(self._index_columns)):
            if i == 0:
                linear_idx = index_indices[-(i + 1)]
            else:
                linear_idx += index_indices[-(i + 1)]
                linear_idx *= self._index_sizes[self._index_columns[-(i + 2)]]

        return linear_idx

    def _compute_strides(self) -> Dict[str, int]:
        """
        Compute strides for each index column.

        For a Cartesian product, the stride for an index column is the product of the
        sizes of all index columns to its right.

        Returns:
            Dictionary mapping index column names to strides
        """
        strides = {}
        stride = 1
        for col in reversed(self._index_columns):
            strides[col] = stride
            stride *= self._index_sizes[col]
        return strides

    def groupby(self, by: str) -> "GroupBy":
        """
        Group by an index column.

        Args:
            by: Name of an index column to group by

        Returns:
            CartesianGroupBy object
        """
        if by not in self._index_columns:
            raise ValueError(
                f"Column '{by}' is not an index column. "
                f"Available index columns: {self._index_columns}"
            )

        return GroupBy(self, by)

    def apply(self, func: Callable, axis: int = 0) -> "DataFrame":
        """
        Apply a function along an axis.

        Args:
            func: JAX function to apply - function should take a JAX array and return a JAX array
            axis: 0 for columns, 1 for rows

        Returns:
            DataFrame: Result of applying the function
        """
        # Prepare column mask for special types
        special_cols_mask = jnp.array(
            [
                self._column_metadata[col].get("dtype_flag")
                in [DATETIME_TYPE_FLAG, CATEGORY_TYPE_FLAG]
                for col in self._column_names
            ]
        )

        # Core computation based on axis
        if axis == 0:
            result_values = self._apply_columns_core(
                self._values, special_cols_mask, func
            )
            # Use original column names
            result_names = self._column_names.copy()
        else:
            result_values = self._apply_rows_core(self._values, func)
            # For row operations, create a compound name with function name and original columns
            func_name = getattr(func, "__name__", "lambda")
            result_names = [f"{func_name}_{'_'.join(self._column_names)}"]
            # Set the values and fix the column shape
            result_values = result_values.reshape(-1, 1)

        # Create result DataFrame
        result = DataFrame.__new__(DataFrame)

        # Copy index information
        result._index_columns = self._index_columns.copy()
        result._index_values = {k: v.copy() for k, v in self._index_values.items()}
        result._index_mappings = {k: v.copy() for k, v in self._index_mappings.items()}
        result._index_sizes = self._index_sizes.copy()

        # Set up column information
        result._column_names = result_names
        result._column_mapping = {name: i for i, name in enumerate(result_names)}

        # Set values
        result._values = result_values

        # Copy metadata for column-wise operations only
        if axis == 0:
            result._column_dtypes = {k: self._column_dtypes[k] for k in result_names}
            result._column_metadata = {
                k: self._column_metadata[k].copy() for k in result_names
            }
        else:
            # For row operations, use default metadata
            result._column_dtypes = {k: np.float32 for k in result_names}
            result._column_metadata = {k: {"dtype_flag": None} for k in result_names}

        return result

    def _apply_columns_core(
        self, values: jax.Array, special_cols_mask: jax.Array, func: Callable
    ) -> jax.Array:
        """Pure JAX function for column-wise apply (fully traceable)."""

        def process_column(col_idx, values):
            col_data = values[:, col_idx]
            is_special = special_cols_mask[col_idx]
            # Apply function only to non-special columns
            result = jnp.where(is_special, col_data, func(col_data))
            return result

        # Process all columns with vmap
        column_indices = jnp.arange(values.shape[1])
        result_columns = jax.vmap(lambda idx: process_column(idx, values))(
            column_indices
        )

        # Stack columns properly
        return jnp.transpose(result_columns)

    def _apply_rows_core(self, values: jax.Array, func: Callable) -> jax.Array:
        """Pure JAX function for row-wise apply (fully traceable)."""
        # Apply function to each row using vmap
        result_values = jax.vmap(func)(values)

        # Ensure result is always a column vector
        def ensure_column_vector(x):
            # First ensure we have a 1D array
            flat = x.ravel()
            # Then reshape to column vector
            return flat.reshape(-1, 1)

        return ensure_column_vector(result_values)

    def __repr__(self) -> str:
        """String representation similar to pandas."""
        return self.__str__()

    def __str__(self) -> str:  # noqa: PLR0912
        """Formatted string representation with index columns."""
        if len(self) == 0:
            return "Empty JaxDataFrame"

        # Format header
        header = f"JaxDataFrame: {self.shape[0]} rows × {self.shape[1]} columns"
        separator = "=" * len(header)

        # Calculate widths for index columns
        index_widths = {}
        for col in self._index_columns:
            index_widths[col] = max(len(col), 10)
            # Check width needed for index values - format them properly based on type
            for val in self._index_values[col]:
                # Format value based on column type - add improved datetime detection
                is_datetime = False
                if col in self._column_metadata:
                    is_datetime = (
                        self._column_metadata[col].get("dtype_flag")
                        == DATETIME_TYPE_FLAG
                    )
                if is_datetime:
                    # For large integers, always try to convert to timestamps
                    if isinstance(val, (int, np.int64)) and val > 1000000000000:
                        val_str = str(pd.Timestamp(val))
                    else:
                        val_str = str(val)
                else:
                    val_str = str(val)
                index_widths[col] = max(index_widths[col], len(val_str))

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

        # Create header row with index columns
        index_part = "    " + "  ".join(
            col.ljust(index_widths[col]) for col in self._index_columns
        )
        if self._index_columns:
            index_part += "  "  # Add spacing between index and data columns
        header_row = index_part + "  ".join(
            col.ljust(col_widths[col]) for col in col_names
        )

        # Calculate index values for each row
        def get_index_values_for_row(row_idx):
            """Convert row index back to index values."""
            idx_values = {}
            remaining_idx = row_idx

            # Using the strides to calculate index values
            strides = self._compute_strides()

            for col in self._index_columns:
                size = self._index_sizes[col]
                stride = strides[col]
                col_idx = (remaining_idx // stride) % size
                remaining_idx %= stride

                # Get the actual value from the index
                raw_val = self._index_values[col][col_idx]

                # Format the value based on column type - with better detection
                if (
                    col in self._column_metadata
                    and self._column_metadata[col].get("dtype_flag")
                    == DATETIME_TYPE_FLAG
                ):
                    # Always treat large integers as potential timestamps
                    if isinstance(raw_val, (int, np.int64)) and raw_val > 1000000000000:
                        # Convert nanosecond int to pandas Timestamp
                        idx_values[col] = pd.Timestamp(raw_val)
                    else:
                        idx_values[col] = raw_val
                else:
                    idx_values[col] = raw_val

            return idx_values

        # Build the table rows
        max_rows = 10  # Maximum rows to display
        table_rows = []

        if len(self) <= max_rows:
            # Show all rows
            for i in range(len(self)):
                # Get index values for this row
                idx_values = get_index_values_for_row(i)

                # Format index part
                index_part = "  ".join(
                    str(idx_values[col]).ljust(index_widths[col])
                    for col in self._index_columns
                )
                if self._index_columns:
                    index_part += "  "  # Add spacing between index and data columns

                # Format data values
                row_vals = []
                for col in col_names:
                    if i < len(formatted_values[col]):
                        val = formatted_values[col][i]
                    else:
                        val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))

                # Combine into one row
                if self._index_columns:
                    table_rows.append(f"{i:3d} {index_part}" + "  ".join(row_vals))
                else:
                    table_rows.append(f"{i:3d} " + "  ".join(row_vals))
        else:
            # Show first and last few rows
            for i in range(5):
                # Get index values for this row
                idx_values = get_index_values_for_row(i)

                # Format index part
                index_part = "  ".join(
                    str(idx_values[col]).ljust(index_widths[col])
                    for col in self._index_columns
                )
                if self._index_columns:
                    index_part += "  "  # Add spacing between index and data columns

                # Format data values
                row_vals = []
                for col in col_names:
                    if i < len(formatted_values[col]):
                        val = formatted_values[col][i]
                    else:
                        val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))

                # Combine into one row
                if self._index_columns:
                    table_rows.append(f"{i:3d} {index_part}" + "  ".join(row_vals))
                else:
                    table_rows.append(f"{i:3d} " + "  ".join(row_vals))

            # Add separator row
            index_ellipsis = "  ".join(
                "...".ljust(index_widths[col]) for col in self._index_columns
            )
            if self._index_columns:
                index_ellipsis += "  "
            table_rows.append(
                "... "
                + index_ellipsis
                + "  ".join("...".ljust(col_widths[col]) for col in col_names)
            )

            for i in range(len(self) - 5, len(self)):
                # Get index values for this row
                idx_values = get_index_values_for_row(i)

                # Format index part
                index_part = "  ".join(
                    str(idx_values[col]).ljust(index_widths[col])
                    for col in self._index_columns
                )
                if self._index_columns:
                    index_part += "  "  # Add spacing between index and data columns

                # Format data values
                row_vals = []
                for col in col_names:
                    val = self._format_value(col, i)
                    row_vals.append(val.ljust(col_widths[col]))

                # Combine into one row
                if self._index_columns:
                    table_rows.append(f"{i:3d} {index_part}" + "  ".join(row_vals))
                else:
                    table_rows.append(f"{i:3d} " + "  ".join(row_vals))

        # Column type information
        type_info = []
        # Add index column types
        for col in self._index_columns:
            metadata = self._column_metadata[col]
            if metadata["dtype_flag"] == DATETIME_TYPE_FLAG:
                type_str = "datetime64[ns]"
            elif metadata["dtype_flag"] == CATEGORY_TYPE_FLAG:
                num_cats = len(self._index_values[col])
                type_str = f"category({num_cats})"
            else:
                type_str = "object"
            type_info.append(f"{col} (index): {type_str}")

        # Add data column types
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


class GroupBy:
    """
    GroupBy implementation for CartesianDataFrame that works with JAX tracing.

    This implementation leverages the Cartesian structure to avoid boolean indexing
    and make operations fully traceable.
    """

    def __init__(self, df: DataFrame, by: str):
        """
        Initialize GroupBy operation.

        Args:
            df: CartesianDataFrame to group
            by: Index column to group by
        """
        self.df = df
        self.by = by

        # Compute strides for each index column
        self._strides = df._compute_strides()

        # Get information for the groupby column
        self.values = df._index_values[by]
        self.n_groups = df._index_sizes[by]

        # Compute the size of each group
        # This is the product of the sizes of all other index columns
        self.group_size = 1
        for col in df._index_columns:
            if col != by:
                self.group_size *= df._index_sizes[col]

        # Get the stride for the groupby column
        self.stride = self._strides[by]

    def transform(self, func: Callable) -> DataFrame:
        """
        Apply a transformation function to each group.

        This method is fully traceable and can be used in JIT-compiled functions.

        Args:
            func: Function to apply to each group

        Returns:
            CartesianDataFrame with transformed values
        """
        # Create the core data for transformation
        transformed_values = self._transform_core(self.df._values, func)

        # Create result DataFrame
        result = DataFrame.__new__(DataFrame)

        # Copy index information
        result._index_columns = self.df._index_columns.copy()
        result._index_values = {k: v.copy() for k, v in self.df._index_values.items()}
        result._index_mappings = {
            k: v.copy() for k, v in self.df._index_mappings.items()
        }
        result._index_sizes = self.df._index_sizes.copy()

        # Copy column information
        result._column_names = self.df._column_names.copy()
        result._column_mapping = self.df._column_mapping.copy()
        result._column_dtypes = self.df._column_dtypes.copy()

        # Copy metadata but ensure index columns have metadata even if they're not data columns
        result._column_metadata = {
            k: v.copy() for k, v in self.df._column_metadata.items()
        }

        # Make sure all index columns have metadata
        for col in result._index_columns:
            if col not in result._column_metadata:
                # If the index column doesn't have metadata yet, add default metadata
                # based on its type
                if pd.api.types.is_datetime64_any_dtype(
                    pd.Series(result._index_values[col])
                ):
                    result._column_metadata[col] = {"dtype_flag": DATETIME_TYPE_FLAG}
                elif any(isinstance(x, str) for x in result._index_values[col]):
                    # For string types
                    unique_vals = sorted(set(str(v) for v in result._index_values[col]))
                    result._column_metadata[col] = {
                        "dtype_flag": CATEGORY_TYPE_FLAG,
                        "category_map": {i: val for i, val in enumerate(unique_vals)},
                    }
                else:
                    # Default metadata for numeric types
                    result._column_metadata[col] = {"dtype_flag": None}

        # Set transformed values
        result._values = transformed_values

        return result

    def _transform_core(self, values: jax.Array, func: Callable) -> jax.Array:
        """
        Core transformation logic using JAX operations.

        This implementation leverages the Cartesian product structure to make
        group operations fully traceable. Since we know the exact size and
        structure of each group, we can use fixed slicing patterns instead
        of boolean indexing.

        Args:
            values: Data values (n_rows × n_cols)
            func: Function to apply to each group

        Returns:
            Transformed values (n_rows × n_cols)
        """
        n_rows, n_cols = values.shape

        # Initialize result array
        result = jnp.zeros_like(values)

        # Process each group
        for group_idx in range(self.n_groups):
            # Process each column
            for col_idx in range(n_cols):
                # Skip special column types
                col_name = self.df._column_names[col_idx]
                col_metadata = self.df._column_metadata.get(col_name, {})
                if col_metadata.get("dtype_flag") in [
                    DATETIME_TYPE_FLAG,
                    CATEGORY_TYPE_FLAG,
                ]:
                    # For special types, copy the original values
                    if group_idx == 0:  # Only copy once
                        result = result.at[:, col_idx].set(values[:, col_idx])
                    continue

                # Gather all values for this group
                group_values = jnp.zeros(self.group_size)

                # Fill group values
                for i in range(self.group_size):
                    # Calculate the row index for this group and offset
                    # For a Cartesian product, the groups are arranged in blocks
                    # with a stride determined by the position of the groupby column
                    row_idx = (
                        (group_idx * self.stride)
                        + (i % self.stride)
                        + (i // self.stride * self.stride * self.n_groups)
                    )

                    # Check bounds
                    if row_idx < n_rows:
                        group_values = group_values.at[i].set(values[row_idx, col_idx])

                # Apply the transformation function
                transformed = func(group_values)

                # Scatter the transformed values back
                for i in range(self.group_size):
                    row_idx = (
                        (group_idx * self.stride)
                        + (i % self.stride)
                        + (i // self.stride * self.stride * self.n_groups)
                    )
                    if row_idx < n_rows:
                        result = result.at[row_idx, col_idx].set(transformed[i])

        return result

    def aggregate(self, func: Callable) -> pd.DataFrame:
        """
        Apply an aggregation function to each group.

        This method is fully traceable and can be used in JIT-compiled functions.
        The result is a pandas DataFrame for easy viewing and further processing.

        Args:
            func: Aggregation function

        Returns:
            pandas DataFrame with one row per group
        """
        # Compute aggregated values
        agg_values = self._aggregate_core(self.df._values, func)

        # Create pandas DataFrame
        return pd.DataFrame(
            agg_values, index=self.values, columns=self.df._column_names
        )

    def _aggregate_core(self, values: jax.Array, func: Callable) -> jax.Array:
        """
        Core aggregation logic using JAX operations.

        Args:
            values: Data values (n_rows × n_cols)
            func: Function to apply to each group

        Returns:
            Aggregated values (n_groups × n_cols)
        """
        n_rows, n_cols = values.shape

        # Initialize result array
        result = jnp.zeros((self.n_groups, n_cols))

        # Process each group
        for group_idx in range(self.n_groups):
            # Process each column
            for col_idx in range(n_cols):
                # Skip special column types
                col_name = self.df._column_names[col_idx]
                col_metadata = self.df._column_metadata.get(col_name, {})
                if col_metadata.get("dtype_flag") in [
                    DATETIME_TYPE_FLAG,
                    CATEGORY_TYPE_FLAG,
                ]:
                    # For special types, use the first value in the group
                    row_idx = group_idx * self.stride
                    if row_idx < n_rows:
                        result = result.at[group_idx, col_idx].set(
                            values[row_idx, col_idx]
                        )
                    continue

                # Gather all values for this group
                group_values = jnp.zeros(self.group_size)

                # Fill group values
                for i in range(self.group_size):
                    # Calculate the row index for this group and offset
                    row_idx = (
                        (group_idx * self.stride)
                        + (i % self.stride)
                        + (i // self.stride * self.stride * self.n_groups)
                    )

                    # Check bounds
                    if row_idx < n_rows:
                        group_values = group_values.at[i].set(values[row_idx, col_idx])

                # Apply the aggregation function
                agg_value = func(group_values)

                # Store the result
                result = result.at[group_idx, col_idx].set(agg_value)

        return result


# Register CartesianDataFrame as a pytree
def _dataframe_tree_flatten(
    df: DataFrame,
) -> Tuple[List[jax.Array], Dict[str, Any]]:
    """Flatten CartesianDataFrame for JAX tracing."""
    # Extract traceable arrays
    leaves = [df._values]

    # Extract metadata
    aux_data = {
        "column_names": df._column_names,
        "column_mapping": df._column_mapping,
        "column_dtypes": df._column_dtypes,
        "column_metadata": df._column_metadata,
        "index_columns": df._index_columns,
        "index_values": df._index_values,
        "index_mappings": df._index_mappings,
        "index_sizes": df._index_sizes,
    }

    return leaves, aux_data


def _dataframe_tree_unflatten(
    aux_data: Dict[str, Any], leaves: List[jax.Array]
) -> DataFrame:
    """Reconstruct CartesianDataFrame from traced arrays."""
    # Create new CartesianDataFrame
    result = DataFrame.__new__(DataFrame)

    # Set values
    result._values = leaves[0]

    # Set metadata
    result._column_names = aux_data["column_names"]
    result._column_mapping = aux_data["column_mapping"]
    result._column_dtypes = aux_data["column_dtypes"]
    result._column_metadata = aux_data["column_metadata"]
    result._index_columns = aux_data["index_columns"]
    result._index_values = aux_data["index_values"]
    result._index_mappings = aux_data["index_mappings"]
    result._index_sizes = aux_data["index_sizes"]

    return result


# Register CartesianGroupBy as a pytree
def _groupby_tree_flatten(gb: GroupBy) -> Tuple[List[Any], Dict[str, Any]]:
    """Flatten CartesianGroupBy for JAX tracing."""
    # No direct leaves, but we reference the dataframe
    leaves = []

    # Extract metadata
    aux_data = {
        "df": gb.df,
        "by": gb.by,
        "values": gb.values,
        "n_groups": gb.n_groups,
        "group_size": gb.group_size,
        "stride": gb.stride,
        "_strides": gb._strides,
    }

    return leaves, aux_data


def _groupby_tree_unflatten(aux_data: Dict[str, Any], leaves: List[Any]) -> GroupBy:
    """Reconstruct CartesianGroupBy from traced arrays."""
    # Create new CartesianGroupBy
    result = GroupBy.__new__(GroupBy)

    # Set metadata
    result.df = aux_data["df"]
    result.by = aux_data["by"]
    result.values = aux_data["values"]
    result.n_groups = aux_data["n_groups"]
    result.group_size = aux_data["group_size"]
    result.stride = aux_data["stride"]
    result._strides = aux_data["_strides"]

    return result


# Register both classes as pytrees
register_pytree_node(DataFrame, _dataframe_tree_flatten, _dataframe_tree_unflatten)
register_pytree_node(GroupBy, _groupby_tree_flatten, _groupby_tree_unflatten)
