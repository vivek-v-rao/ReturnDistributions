"""Fit saved price or return columns; no downloads or portfolio dependencies."""
import argparse
from pathlib import Path
import sys
import time
import textwrap

import numpy as np
import pandas as pd

from .fitting import DEFAULT_MODELS, fit_many, fitted_distribution, specification, model_catalog
from .data import read_returns
from .model_names import display_model_name, canonical_model, unique_models, display_model_label
from .table_format import aligned_table
from .distribution_distances import add_distance_options, selected_metrics, compare_univariate
from .fit_timeout import positive_seconds, run_fit, FitTimeout
from .joint_finite_mixture import FINITE_MIXTURE_MODELS
from .univariate_mixture import fit_univariate_mixture, print_components
from .fit_ranks import criterion_ranks
from .joint_univariate import format_univariate_comparison
from .univariate_analysis import all_project_models, prepare_samples, samples, risk_rows, print_risk
from .sample_blocks import add_block_options, validate_blocks, rank_groups


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('file', type=Path, help='CSV: first column is date, remaining columns are prices/returns')
    parser.add_argument('--input-type', choices=['prices', 'returns'], default='prices')
    parser.add_argument('--return-type', choices=['simple', 'log'], default='simple')
    parser.add_argument('--return-scale', type=positive_seconds, default=1., help='Multiply input returns by this positive factor (default 1)')
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--common-sample', action='store_true', help='Use identical complete-case dates across selected assets')
    parser.add_argument('--standardize-vol', choices=['none', 'ewma'], nargs='+', help='Raw and/or lagged EWMA-standardized fits on identical dates')
    decay_group = parser.add_mutually_exclusive_group()
    decay_group.add_argument('--vol-lambda', nargs='+', type=float, help='EWMA decays; default .94')
    decay_group.add_argument('--vol-halflife', type=positive_seconds)
    parser.add_argument('--vol-warmup', type=int, default=63)
    parser.add_argument('--vol-floor', type=positive_seconds, default=1e-8)
    parser.add_argument('--risk-levels', nargs='+', type=float, help='Print per-asset VaR/ES at these confidence levels')
    model_help = [textwrap.fill('Default: ' + ', '.join(map(display_model_name, DEFAULT_MODELS)), width=56, break_on_hyphens=False)]
    names = sorted({display_model_name(name) for name in model_catalog()['Project models and aliases']})
    model_help.append('Project models (preferred names):\n' + textwrap.fill(', '.join(names), width=56, break_on_hyphens=False))
    model_help.append('Older spellings and equivalent aliases remain accepted.\nNames are validated before data loading; numerical fit\nfailures remain visible in the results.')
    parser.add_argument('--models', nargs='+', metavar='MODEL', default=list(DEFAULT_MODELS),
                        help='Use all alone for every project model (not every SciPy distribution).\n\n'+'\n\n'.join(model_help))
    parser.add_argument('--days', nargs='+', type=int, help='Last N valid returns per asset; default all history')
    parser.add_argument('--location', type=float, help='Fix location in fit input units (standardized for EWMA); default estimated')
    parser.add_argument('--max-iterations', type=int, default=10000)
    parser.add_argument('--max-components', type=int, default=1, help='Fit 1 through K components (1--8)')
    parser.add_argument('--mixture-models', nargs='+', type=canonical_model,
                        choices=(*FINITE_MIXTURE_MODELS, 'all'),
                        help='Mixture families; all/default selects supported --models families')
    parser.add_argument('--mixture-starts', type=int, default=5)
    parser.add_argument('--mixture-seed', type=int, default=12345)
    parser.add_argument('--mixture-min-weight', type=float, default=.01)
    parser.add_argument('--mixture-eigen-floor', type=positive_seconds, default=1e-4)
    parser.add_argument('--fit-timeout', type=positive_seconds, metavar='SECONDS',
                        help='Hard limit per model/asset/window fit, including worker startup and all starts; default unlimited')
    parser.add_argument('--output', type=Path, default=Path('distribution_fits.csv'))
    parser.add_argument('--no-save', action='store_true', help='Screen output only; do not write results (--output is ignored)')
    parser.add_argument('--show-plot', action='store_true', help='Display density and Q-Q diagnostics for successful fits')
    add_distance_options(parser)
    add_block_options(parser)
    args = parser.parse_args(argv)
    validate_blocks(parser, args)
    modes = list(dict.fromkeys(args.standardize_vol or ['none']))
    decays = list(dict.fromkeys([float(np.exp(np.log(.5)/args.vol_halflife))] if args.vol_halflife is not None else (args.vol_lambda or [.94])))
    if 'ewma' not in modes and (args.vol_lambda is not None or args.vol_halflife is not None or args.vol_warmup != 63 or args.vol_floor != 1e-8):
        parser.error('Volatility filter options require --standardize-vol ewma')
    if any(not np.isfinite(v) or not 0 < v < 1 for v in decays) or args.vol_warmup < 2:
        parser.error('Require 0 < --vol-lambda < 1 and --vol-warmup >=2')
    if args.risk_levels and any(not np.isfinite(v) or not 0 < v < 1 for v in args.risk_levels):
        parser.error('--risk-levels must be strictly between 0 and 1')
    args.risk_levels = list(dict.fromkeys(args.risk_levels or []))
    if 'all' in args.models:
        if args.models != ['all']: parser.error('--models all must be used alone')
        args.models = all_project_models()
    args.models = unique_models(args.models)
    if (not 1 <= args.max_components <= 8 or args.mixture_starts < 2 or args.mixture_seed < 0
            or not np.isfinite(args.mixture_min_weight) or not 0 < args.mixture_min_weight < 1/args.max_components):
        parser.error('Require 1--8 components, >=2 starts, nonnegative seed, and 0 < minimum weight < 1/max-components')
    if args.mixture_models and args.max_components == 1:
        parser.error('--mixture-models requires --max-components >=2')
    if args.mixture_models and 'all' in args.mixture_models:
        if args.mixture_models != ['all']:
            parser.error('--mixture-models all must be used alone')
        args.mixture_models = None
    mixture_models = unique_models(args.mixture_models if args.mixture_models is not None
                                   else [m for m in args.models if m in FINITE_MIXTURE_MODELS])
    args.models = unique_models([*args.models, *mixture_models])
    if args.max_components > 1 and not mixture_models:
        parser.error('No supported mixture families selected')
    metrics = selected_metrics(args)
    unknown = []
    for name in dict.fromkeys(args.models):
        try:
            specification(name)
        except ValueError:
            unknown.append(name)
    if unknown:
        parser.error('Unknown continuous distribution model(s): ' + ', '.join(unknown))
    if args.days and min(args.days) < 8:
        parser.error('--days must be at least 8')
    if args.max_iterations < 1:
        parser.error('--max-iterations must be positive')
    if not args.no_save and args.output.resolve() == args.file.resolve():
        parser.error('Output must not overwrite input')
    if not args.no_save and metrics:
        distance_path = args.output.with_name(args.output.stem+'_distances.csv')
        if distance_path.resolve() in {args.file.resolve(), args.output.resolve()}:
            parser.error('Distance output must not overwrite input or fit output')
    risk_path = args.output.with_name(args.output.stem+'_risk.csv') if not args.no_save and args.risk_levels else None
    if not args.no_save and args.risk_levels and risk_path.resolve() in {args.file.resolve(), args.output.resolve()}:
        parser.error('Risk output must not overwrite input or fit output')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions', *(sys.argv[1:] if argv is None else argv)]))
    if args.max_components > 1:
        print(f'Finite mixtures: {", ".join(map(display_model_label, mixture_models))}; 1--{args.max_components} components; {args.mixture_starts} starts.')
        print(f'Minimum weight: {args.mixture_min_weight}; scaled scatter floor: {args.mixture_eigen_floor}. Student-t/GED/NIG share df/power/psi respectively. Non-ok fits are excluded from plots and distances.')
    try:
        frame = read_returns(args.file, args.symbols, args.input_type, args.return_type)
        frame = frame * args.return_scale
        print(f'Return scale: {args.return_scale:g}; raw parameters and likelihoods use scaled return units.')
        cache, eligible = prepare_samples(frame, modes, decays, args.vol_warmup, args.vol_floor, args.common_sample)
        print('Sample policy: '+('common complete-case dates' if args.common_sample else 'per-asset available dates'))
        if cache:
            print('Lagged zero-mean EWMA; common dates across raw/decay variants; warmup excluded. Parameters/moments/plots/distances use fit units; loglik/AIC/BIC use scaled return units. Filter/decay selection uncertainty is not counted.')
        data_elapsed = time.perf_counter()-start
        frames = []
        distance_rows = []
        risks = []
        for symbol in frame:
            for window, mode, decay, series, adjustment, next_scale, count, block in samples(frame, symbol, args.days, modes, decays, cache, eligible,
                    partitions=args.subperiods, date_min=args.date_min, date_max=args.date_max, floor=args.vol_floor):
                print(f'\nPartition: {count}; block: {block}/{count}')
                print(f'\n{symbol}: {len(series)} returns; ' +
                      (f'{series.index.min().date()} to {series.index.max().date()}' if len(series) else 'no data'))
                print(f'Volatility standardization: {mode}'+(f'; lambda={decay}; next-period SD={next_scale:.6g} (scaled return units)' if mode == 'ewma' else ''))
                if count == 1 and window is not None and len(series) < window:
                    print(f'Warning: fewer than {window} requested observations')
                fits = fit_many(series.to_numpy(), args.models, location=args.location, max_iterations=args.max_iterations,
                                fit_timeout=args.fit_timeout)
                if args.max_components > 1:
                    fits['components'] = 1
                    mixture_rows = []
                    for model in mixture_models:
                        baseline = fits.loc[fits.name.eq(model)].iloc[0].to_dict()
                        for count in range(2, args.max_components+1):
                            print(f'Fitting {display_model_label(model)} [{count} components]...', flush=True)
                            fit_start = time.perf_counter()
                            try:
                                row = run_fit(fit_univariate_mixture, series.to_numpy(), model, count,
                                    baseline=baseline, location=args.location, max_iterations=args.max_iterations,
                                    starts=args.mixture_starts, seed=args.mixture_seed,
                                    min_weight=args.mixture_min_weight, eigen_floor=args.mixture_eigen_floor,
                                    fit_timeout=args.fit_timeout)
                                print_components(row, 'standardized return units' if mode == 'ewma' else 'return units')
                            except Exception as exc:
                                row = dict(name=f'{model} [{count} components]', components=count,
                                    status='timeout' if isinstance(exc, FitTimeout) else 'failed',
                                    converged=False, error=str(exc), fit_sec=time.perf_counter()-fit_start)
                            mixture_rows.append(row)
                    fits = pd.concat([fits, pd.DataFrame(mixture_rows)], ignore_index=True)
                if mode == 'ewma':
                    for key in ('loglik', 'aic', 'bic'):
                        if key not in fits: fits[key] = np.nan
                    fits['standardized_loglik'] = fits['loglik']
                    fits['loglik'] -= adjustment
                    for key in ('aic', 'bic'): fits[key] += 2*adjustment
                singles = fits.loc[fits.components.eq(1)] if 'components' in fits else fits
                print(format_univariate_comparison(singles.drop(columns=['mixture_fit', 'mixture_model'], errors='ignore')))
                if args.max_components > 1:
                    fits = criterion_ranks(fits, include_unconverged=True)
                    print('\nAggregate univariate fit comparison:')
                    print('Ranks within this asset/window; every finite AIC/BIC is included regardless of convergence. Mixture criteria are descriptive; inspect status. Shared tail shapes; constrained weights/scales; multiple starts do not guarantee a global optimum.')
                    columns = ['name', 'components', 'k', 'loglik', 'aic', 'bic', 'aic_rank', 'bic_rank', 'status', 'fit_sec', 'error']
                    print(aligned_table(fits[[c for c in columns if c in fits]]))
                fits['symbol'] = symbol
                fits['window'] = window if window is not None else 'all'
                fits['first_date'] = str(series.index.min().date()) if len(series) else None
                fits['last_date'] = str(series.index.max().date()) if len(series) else None
                fits['return_type'] = args.return_type
                fits['return_scale'] = args.return_scale
                fits['subperiods'] = count
                fits['block'] = block
                fits['vol_standardization'] = mode
                fits['vol_lambda'] = np.nan if decay is None else decay
                fits['vol_log_jacobian'] = adjustment
                fits['next_volatility'] = next_scale if mode == 'ewma' else np.nan
                fits['vol_warmup'] = args.vol_warmup if mode == 'ewma' else np.nan
                fits['vol_floor'] = args.vol_floor if mode == 'ewma' else np.nan
                fits['parameter_units'] = 'standardized returns' if mode == 'ewma' else 'scaled returns'
                fits['likelihood_units'] = 'scaled returns'
                fits['sample_policy'] = 'common-complete-case' if args.common_sample else 'per-asset'
                frames.append(fits)
                metadata = dict(symbol=symbol, window=window or 'all', first_date=str(series.index.min().date()),
                                last_date=str(series.index.max().date()), vol_standardization=mode, vol_lambda=decay,
                                return_type=args.return_type, return_scale=args.return_scale, observations=len(series), subperiods=count, block=block)
                if args.risk_levels:
                    rows = risk_rows(fits, series, args.risk_levels, next_scale/args.return_scale)
                    print_risk(rows, mode, args.risk_levels, len(series))
                    risks.extend(dict(r, **metadata) for r in rows)
                if metrics:
                    rows = compare_univariate(fits, metrics)
                    distance_rows.extend(dict(r, **metadata, scope='univariate') for r in rows)
                if args.show_plot:
                    plot_fits(series.to_numpy(), fits, f'{symbol}, {window or "all"} returns, {mode}, lambda={decay}')
        if not frames:
            raise ValueError('No asset columns')
        combined = pd.concat(frames, ignore_index=True)
        if len(modes) > 1 or ('ewma' in modes and len(decays) > 1):
            combined = criterion_ranks(combined, [combined.symbol, *rank_groups(combined)], include_unconverged=args.max_components > 1)
            print('\nAggregate raw/EWMA comparison (ranks within asset/window across all settings):')
            cols = ['symbol', 'window', 'subperiods', 'block', 'name', 'components', 'vol_standardization', 'vol_lambda', 'k', 'loglik', 'aic', 'bic', 'aic_rank', 'bic_rank', 'status', 'fit_sec']
            print(aligned_table(combined[[c for c in cols if c in combined]]))
        if not args.no_save:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(args.output, index=False)
            print(f'\nWrote {args.output}')
            if metrics:
                pd.DataFrame(distance_rows).to_csv(distance_path, index=False)
                print(f'Wrote {distance_path}')
            if risks:
                pd.DataFrame(risks).to_csv(risk_path, index=False)
                print(f'Wrote {risk_path}')
        print('KS is descriptive; fitted-sample KS p-values are omitted.')
        print('AIC/BIC compare models on the same asset/window only; likelihood assumes independent observations.')
        print(f'Data elapsed: {data_elapsed:.3f} seconds\nFitting/output elapsed: {time.perf_counter()-start-data_elapsed:.3f} seconds\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        return 0 if combined.status.eq('ok').all() and all(r['status']=='ok' for r in distance_rows+risks) else 1
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
        axes[0].plot(grid, dist.pdf(grid), label=display_model_label(row['name']))
        axes[1].plot(dist.ppf(p), np.sort(x), '.', markersize=2, label=display_model_label(row['name']))
    axes[1].plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    axes[1].set(xlabel='Fitted quantiles', ylabel='Observed quantiles')
    axes[0].legend(fontsize='small')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
