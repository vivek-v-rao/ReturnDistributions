"""Shared saved-price/return loading; never fill gaps."""
import numpy as np
import pandas as pd


def read_returns(path, symbols=None, input_type='prices', return_type='simple'):
    if input_type not in {'prices', 'returns'} or return_type not in {'simple', 'log'}:
        raise ValueError('Invalid input or return type')
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, errors='raise')
    if frame.index.has_duplicates or frame.index.isna().any():
        raise ValueError('Dates must be unique and nonmissing')
    frame = frame.sort_index()
    if symbols:
        frame = frame[list(dict.fromkeys(symbols))]
    frame = frame.apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy()).any():
        raise ValueError('Infinite input values are not allowed')
    if input_type == 'prices':
        if (frame <= 0).any().any():
            raise ValueError('Prices must be positive')
        ratio = frame/frame.shift(1)
        frame = np.log(ratio) if return_type == 'log' else ratio-1
    return frame
