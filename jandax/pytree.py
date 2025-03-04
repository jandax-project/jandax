from typing import Any, Dict, List, Tuple

import jax
from jax.tree_util import register_pytree_node

# Import the classes we need to register
from jandax.core import DataFrame, GroupBy, GroupByRolling, Rolling


# Define tree_flatten and tree_unflatten for DataFrame
def _dataframe_tree_flatten(df: DataFrame) -> Tuple[List[jax.Array], Dict[str, Any]]:
    """Flatten DataFrame into arrays and static data for JAX tracing."""
    # Leaves are the arrays that should be traced by JAX
    leaves = [df._values]

    # Auxiliary data contains all the metadata that shouldn't change during tracing
    aux_data = {
        "column_names": df._column_names,
        "column_mapping": df._column_mapping,
        "column_metadata": df._column_metadata,
        "column_unique": df._column_unique if hasattr(df, "_column_unique") else {},
    }

    return leaves, aux_data


def _dataframe_tree_unflatten(
    aux_data: Dict[str, Any], leaves: List[jax.Array]
) -> DataFrame:
    """Reconstruct DataFrame from leaves and auxiliary data."""
    # Create empty DataFrame
    self = DataFrame.__new__(DataFrame)

    # Restore the JAX array
    self._values = leaves[0]

    # Restore metadata from auxiliary data
    self._column_names = aux_data["column_names"]
    self._column_mapping = aux_data["column_mapping"]
    self._column_metadata = aux_data["column_metadata"]

    # Initialize _column_unique as empty dict to avoid jnp.unique in JIT context
    self._column_unique = {}

    return self


# Define tree_flatten and tree_unflatten for Rolling
def _rolling_tree_flatten(roll: Rolling) -> Tuple[List[jax.Array], Dict[str, Any]]:
    """Flatten Rolling into arrays and static data for JAX tracing."""
    # Rolling itself has no arrays to trace directly, but it contains a DataFrame
    leaves = []

    aux_data = {
        "df": roll.df,
        "window_size": roll.window_size,
        "min_periods": roll.min_periods,
    }

    return leaves, aux_data


def _rolling_tree_unflatten(
    aux_data: Dict[str, Any], leaves: List[jax.Array]
) -> Rolling:
    """Reconstruct Rolling from leaves and auxiliary data."""
    # Create empty Rolling
    self = Rolling.__new__(Rolling)

    # Restore from auxiliary data
    self.df = aux_data["df"]
    self.window_size = aux_data["window_size"]
    self.min_periods = aux_data["min_periods"]

    return self


# Define tree_flatten and tree_unflatten for GroupBy
def _groupby_tree_flatten(gb: GroupBy) -> Tuple[List[jax.Array], Dict[str, Any]]:
    """Flatten GroupBy into arrays and static data for JAX tracing."""
    # Leaves are JAX arrays that should be traced
    leaves = [
        gb.grouping_values,
        gb.unique_groups,
        gb.group_ids,
        gb.group_masks,
    ]

    # Auxiliary data contains metadata and the reference to DataFrame
    aux_data = {
        "df": gb.df,
        "by": gb.by,
        "by_metadata": gb.by_metadata,
        "group_indices": gb.group_indices,
    }

    return leaves, aux_data


def _groupby_tree_unflatten(
    aux_data: Dict[str, Any], leaves: List[jax.Array]
) -> GroupBy:
    """Reconstruct GroupBy from leaves and auxiliary data."""
    # Create empty GroupBy
    self = GroupBy.__new__(GroupBy)

    # Restore JAX arrays from leaves
    self.grouping_values = leaves[0]
    self.unique_groups = leaves[1]
    self.group_ids = leaves[2]
    self.group_masks = leaves[3]

    # Restore metadata from auxiliary data
    self.df = aux_data["df"]
    self.by = aux_data["by"]
    self.by_metadata = aux_data["by_metadata"]
    self.group_indices = aux_data["group_indices"]

    return self


# Define tree_flatten and tree_unflatten for GroupByRolling
def _gbrolling_tree_flatten(
    gbr: GroupByRolling,
) -> Tuple[List[jax.Array], Dict[str, Any]]:
    """Flatten GroupByRolling into arrays and static data for JAX tracing."""
    # Leaves are JAX arrays that should be traced
    leaves = [gbr.process_mask_array]

    # Auxiliary data contains metadata
    aux_data = {
        "groupby": gbr.groupby,
        "window_size": gbr.window_size,
        "column_masks": gbr.column_masks,
    }

    return leaves, aux_data


def _gbrolling_tree_unflatten(
    aux_data: Dict[str, Any], leaves: List[jax.Array]
) -> GroupByRolling:
    """Reconstruct GroupByRolling from leaves and auxiliary data."""
    # Create empty GroupByRolling
    self = GroupByRolling.__new__(GroupByRolling)

    # Restore JAX arrays from leaves
    self.process_mask_array = leaves[0]

    # Restore metadata from auxiliary data
    self.groupby = aux_data["groupby"]
    self.window_size = aux_data["window_size"]
    self.column_masks = aux_data["column_masks"]

    return self


# Register all classes as PyTrees
register_pytree_node(DataFrame, _dataframe_tree_flatten, _dataframe_tree_unflatten)

register_pytree_node(Rolling, _rolling_tree_flatten, _rolling_tree_unflatten)

register_pytree_node(GroupBy, _groupby_tree_flatten, _groupby_tree_unflatten)

register_pytree_node(GroupByRolling, _gbrolling_tree_flatten, _gbrolling_tree_unflatten)
