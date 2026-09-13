"""Fit saved price or return columns; no downloads or portfolio dependencies."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .fitting import DEFAULT_MODELS, fit_many, fitted_distribution
from .data import read_returns


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='CSV: first column is date, remaining columns are prices/returns')
    parser.add_argument('--input-type', choices=['prices', 'returns'], default='prices')
    parser.add_argument('--return-type', choices=['simple', 'log'], default='simple')
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS), help='Model names; default: eight requested families/variants')
    parser.add_argument('--days', nargs='+', type=int, help='Last N valid returns per asset; default all history')
    parser.add_argument('--location', type=float, help='Fix location in return units; default estimated')
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=Path('distribution_fits.csv'))
    parser.add_argument('--show-plot', action='store_true', help='Display density and Q-Q diagnostics for successful fits')
    args = parser.parse_args(argv)
    if args.days and min(args.days) < 8:
        parser.error('--days must be at least 8')
    if args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    if args.output.resolve() == args.file.resolve():
        parser.error('Output must not overwrite input')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions', *(sys.argv[1:] if argv is None else argv)]))
    try:
        frame = read_returns(args.file, args.symbols, args.input_type, args.return_type)
        data_elapsed = time.perf_counter()-start
        frames = []
        for symbol in frame:
            for window in dict.fromkeys(args.days or [None]):
                series = frame[symbol].dropna()
                if window is not None:
                    series = series.tail(window)
                print(f'\n{symbol}: {len(series)} returns; ' +
                      (f'{series.index.min().date()} to {series.index.max().date()}' if len(series) else 'no data'))
                if window is not None and len(series) < window:
                    print(f'Warning: fewer than {window} requested observations')
                fits = fit_many(series.to_numpy(), args.models, location=args.location, max_iterations=args.max_iterations)
                fits['symbol'] = symbol
                fits['window'] = window if window is not None else 'all'
                fits['first_date'] = str(series.index.min().date()) if len(series) else None
                fits['last_date'] = str(series.index.max().date()) if len(series) else None
                fits['return_type'] = args.return_type
                frames.append(fits)
                print(fits.to_string(index=False, float_format=lambda x: f'{x:.5g}', na_rep='n/a'))
                if args.show_plot:
                    plot_fits(series.to_numpy(), fits, f'{symbol}, {window or "all"} returns')
        if not frames:
            raise ValueError('No asset columns')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f'\nWrote {args.output}\nKS is descriptive; fitted-sample KS p-values are omitted.')
        print('AIC/BIC compare models on the same asset/window only; likelihood assumes independent observations.')
        print(f'Data elapsed: {data_elapsed:.3f} seconds\nFitting/output elapsed: {time.perf_counter()-start-data_elapsed:.3f} seconds\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        return 0 if combined.status.eq('ok').all() else 1
    except (ValueError, KeyError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


def plot_fits(x, fits, title):
    import matplotlib.pyplot as plt
    good = fits.loc[fits.status.eq('ok')]
    if good.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(x, bins='auto', density=True, alpha=.3)
    grid = np.linspace(x.min(), x.max(), 500)
    p = (np.arange(len(x))+.5)/len(x)
    for _, row in good.iterrows():
        dist = fitted_distribution(row)
        axes[0].plot(grid, dist.pdf(grid), label=row['name'])
        axes[1].plot(dist.ppf(p), np.sort(x), '.', markersize=2, label=row['name'])
    axes[1].plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    axes[1].set(xlabel='Fitted quantiles', ylabel='Observed quantiles')
    axes[0].legend(fontsize='small')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
