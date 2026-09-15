"""Console-only numeric formatting with a common precision within each column."""
import numbers

import numpy as np
import pandas as pd
from .model_names import display_model_label


INTEGER_COLUMNS = {'n', 'k', 'observations', 'dimensions', 'parameters', 'rank', 'subperiods', 'block',
                   'aic_rank', 'bic_rank', 'aic_rank_family', 'bic_rank_family', 'aic_rank_all', 'bic_rank_all',
                   'conditional_parameters', 'total_parameters', 'clipped_entries', 'window'}


def is_time_column(name):
    return str(name).endswith(('_sec', '_seconds', '_elapsed')) or name == 'elapsed'


def aligned_table(frame):
    """Render a copy; preserve roughly five significant digits and align decimals.

    Floating columns use at least two decimal places. Very small/large or
    widely differing magnitudes use one scientific format for the whole column.
    Counts stay integral; missing values and infinities do not set precision.
    """
    table = frame.copy()
    for name in table.columns:
        values = frame[name]
        present = values.dropna()
        numeric = len(present) > 0 and all(isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) for v in present)
        if not numeric:
            table[name] = values.map(lambda v: 'n/a' if pd.isna(v) else
                                    str(display_model_label(v) if name in {'name', 'model', 'fit_label', 'model_p', 'model_q', 'mixture_model'} else v))
            continue
        finite = np.asarray([v for v in present if np.isfinite(v)], dtype=float)
        nonzero = np.abs(finite[finite != 0])
        integer = (name in INTEGER_COLUMNS or pd.api.types.is_integer_dtype(values.dtype)) and np.all(finite == np.trunc(finite))
        decimals = max(2, 4-int(np.floor(np.log10(nonzero.min())))) if len(nonzero) else 2
        scientific = not integer and len(nonzero) > 0 and (decimals > 10 or nonzero.max() >= 1e9 or nonzero.max()/nonzero.min() >= 1e8)
        if is_time_column(name):
            integer, scientific, decimals = False, False, 3

        def display(value):
            if pd.isna(value):
                return 'n/a'
            if not np.isfinite(value):
                return 'inf' if value > 0 else '-inf'
            if integer:
                return str(int(value))
            if scientific:
                mantissa, exponent = f'{value:.4e}'.split('e')
                return f'{mantissa}e{int(exponent):+04d}'
            return f'{value:.{decimals}f}'

        table[name] = values.map(display)
    return table.to_string(index=False)
