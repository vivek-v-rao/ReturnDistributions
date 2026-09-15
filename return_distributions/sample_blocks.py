"""Inclusive return-date bounds and balanced chronological partitions."""
import datetime
import numpy as np
import pandas as pd


def iso_date(value):
    return datetime.date.fromisoformat(value).isoformat()


def add_block_options(parser):
    parser.add_argument('--date-min', type=iso_date, help='Inclusive earliest return date, YYYY-MM-DD')
    parser.add_argument('--date-max', type=iso_date, help='Inclusive latest return date, YYYY-MM-DD')
    parser.add_argument('--subperiods', nargs='+', type=int, default=[1], help='Partition counts, e.g. 1 2 4')


def validate_blocks(parser, args):
    if args.date_min and args.date_max and args.date_min > args.date_max:
        parser.error('--date-min must not exceed --date-max')
    if any(n < 1 for n in args.subperiods): parser.error('--subperiods must be positive integers')
    args.subperiods = list(dict.fromkeys(args.subperiods))


def date_mask(index, lower=None, upper=None):
    dates = index.strftime('%Y-%m-%d')
    return ((dates >= lower) if lower else np.ones(len(index), dtype=bool)) & ((dates <= upper) if upper else np.ones(len(index), dtype=bool))


def blocks(index, windows, partitions):
    for window in dict.fromkeys(windows or [None]):
        selected = index if window is None else index[-window:]
        for count in partitions:
            if len(selected)//count < 8:
                raise ValueError(f'{count} subperiods of {len(selected)} returns would have fewer than 8 observations per block')
            for number, positions in enumerate(np.array_split(np.arange(len(selected)), count), 1):
                yield window, count, number, selected[positions]


def rank_groups(table):
    return [table[key].fillna('all') for key in ('window', 'subperiods', 'block') if key in table]


def endpoint_scale(frame, scales, endpoint, decay, floor):
    return np.sqrt(np.maximum(decay*scales.loc[endpoint]**2+(1-decay)*frame.loc[endpoint]**2, floor**2))
