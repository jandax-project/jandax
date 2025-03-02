# Jandax

**Traceable and Portable DataFrames for C++ Integration**

Jandax provides a pandas-like DataFrame interface that enables fully traceable and portable chained operations, facilitating seamless integration with C++.

## Installation

```bash
pip install jandax
```

## Basic Usage

```python
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jandax import DataFrame

# Create from pandas DataFrame
pdf = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'date': pd.date_range('2023-01-01', periods=5),
    'category': ['a', 'b', 'a', 'c', 'b']
})
df = DataFrame(pdf)

# Create from dictionary
df = DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
})

# Computation is JAX-compatible
result = df.apply(lambda x: jnp.sin(x), axis=0)

# Column operations
df['C'] = df['A'] * df['B']

# Complex operations
grouped = df.groupby('category')
rolling_mean = df.rolling(3).apply(jnp.nanmean)
```

## Key Features

- **Traceable Operations**: Ensures all operations are fully traceable
- **Portable to C++**: Facilitates seamless integration with C++
- **Pandas-like API**: Familiar interface for data manipulation
- **Special data type support**: Handles datetime and categorical data
- **Seamless conversion**: Easy conversion to/from pandas DataFrames
- **GroupBy and rolling operations**: Support for grouped and windowed computations

## API Reference

### Core Data Structures

- **DataFrame**: Main dataframe class with traceable operations
- **Rolling**: Represents a rolling window over a DataFrame
- **GroupBy**: Handles grouped operations
- **GroupByRolling**: Combines groupby and rolling functionality

### Key Methods

- **DataFrame.apply()**: Apply a function along an axis
- **DataFrame.rolling()**: Create a rolling window
- **DataFrame.groupby()**: Group data by values
- **DataFrame.to_pandas()**: Convert to pandas DataFrame
- **DataFrame.from_pandas()**: Create from pandas DataFrame

## Traceability Paradigm

Jandax implements a structured approach to ensure traceability while maintaining a rich, pandas-like API. Operations in Jandax follow a three-phase architecture:

1. **Pre-processing Phase (Non-traceable)**
   - Dynamic data structure creation
   - Type checking and conversions
   - Metadata management

2. **Core Computation Phase (Fully traceable)**
   - Only use JAX operations
   - Avoid Python control flow based on data values
   - Maintain static shapes for inputs and outputs
   - Use JAX primitives (e.g., jax.lax.cond instead of Python if)

3. **Post-processing Phase (Non-traceable)**
   - Converting JAX arrays to appropriate Python/pandas objects
   - Reconstructing metadata
   - Creating new DataFrame instances

By carefully separating these concerns, Jandax allows users to leverage powerful program transformations while still offering a rich DataFrame API that is fully traceable and portable to C++.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
