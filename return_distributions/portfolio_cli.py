"""Project saved joint distributions onto portfolio weights."""
import argparse
from .model_names import canonical_model
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .projection import project_distribution
from .tail_risk import portfolio_risk
from .copula_portfolio import simulate_portfolio
from .joint_portfolio_mc import simulate_joint_portfolio
from .joint_sdb import SDB_MODELS


def read_weights(entries=None, path=None):
    if path:
        frame = pd.read_csv(path)
        if not {'symbol', 'weight'} <= set(frame.columns):
            raise ValueError('Weights CSV requires symbol,weight columns')
        pairs = list(zip(frame.symbol, frame.weight))
    else:
        pairs = []
        for entry in entries:
            if '=' not in entry: raise ValueError('Use weights such as SPY=0.6 TLT=0.4')
            pairs.append(entry.split('=', 1))
    weights = {}
    for symbol, value in pairs:
        if pd.isna(symbol): raise ValueError('Missing weight symbol')
        symbol = str(symbol).strip().upper()
        if not symbol or symbol in weights: raise ValueError('Empty or duplicate weight symbol')
        value = float(value)
        if not np.isfinite(value): raise ValueError('Weights must be finite')
        weights[symbol] = value
    if not weights: raise ValueError('No weights provided')
    return weights


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fits_file', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--weights', nargs='+', metavar='SYMBOL=WEIGHT')
    group.add_argument('--weights-file', type=Path)
    parser.add_argument('--models', nargs='+', type=canonical_model, help='Optionally select families from saved fits; family aliases select skewed variants')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[.01, .05, .5, .95, .99])
    parser.add_argument('--risk-levels', type=float, nargs='+', help='Confidence levels for positive-loss VaR/ES, e.g. .95 .99')
    parser.add_argument('--simulations', type=int, default=100000, help='Monte Carlo draws per copula/SDB fit (default 100000)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--mc-batches', type=int, default=20, help='Independent batches for approximate Monte Carlo standard errors')
    parser.add_argument('--stage', choices=['all','two-stage','joint-refined','two-stage-retained'], default='all', help='Select copula estimation stage')
    parser.add_argument('--allow-log-linear-combination', action='store_true')
    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('portfolio_distributions.csv'))
    args = parser.parse_args(argv)
    if args.simulations < 1000 or args.seed < 0 or args.mc_batches < 2 or args.simulations//args.mc_batches < 50:
        parser.error('Require >=1000 simulations, >=2 batches, >=50 draws/batch and a nonnegative seed')
    if any(not 0 < q < 1 for q in args.quantiles): parser.error('Quantiles must be strictly between 0 and 1')
    if args.risk_levels and any(not 0 < c < 1 for c in args.risk_levels): parser.error('Risk levels must be strictly between 0 and 1')
    if args.output.resolve() in {p.resolve() for p in [args.fits_file, args.weights_file] if p}:
        parser.error('Output must not overwrite input')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions.portfolio_cli', *(sys.argv[1:] if argv is None else argv)]))
    try:
        weights = read_weights(args.weights, args.weights_file)
        fits = json.loads(args.fits_file.read_text(encoding='utf-8'))
        if not isinstance(fits, list): raise ValueError('Expected a list of saved joint fits')
        if args.models:
            missing = set(args.models)-{f.get('copula',f).get('model') for f in fits}
            if missing: raise ValueError('Models not in fit file: ' + ', '.join(sorted(missing)))
        print('Weights used without normalization: ' + ', '.join(f'{s}={w:g}' for s, w in weights.items()))
        print(f'Net exposure: {sum(weights.values()):.4g}; gross exposure: {sum(abs(w) for w in weights.values()):.4g}. No cash return or financing costs added.')
        rows, plots = [], []
        for index, fit in enumerate(fits):
            model = fit.get('copula',fit).get('model')
            stage = fit.get('stage','two-stage' if 'copula' in fit else 'joint-family')
            if args.models and model not in args.models: continue
            if args.stage != 'all' and stage != args.stage: continue
            if fit.get('copula',fit).get('status') != 'ok':
                print(f'Warning: skipping fit {index} ({model}): status {fit.get("copula",fit).get("status")}')
                continue
            symbols = [str(s).upper() for s in fit['symbols']]
            if len(set(symbols)) != len(symbols): raise ValueError('Duplicate symbols in fit file')
            missing = set(weights)-set(symbols)
            if missing: raise ValueError(f'Fit {index} lacks weight symbols: {sorted(missing)}')
            w = np.array([weights.get(s, 0.) for s in symbols])
            if 'copula' in fit or model in SDB_MODELS:
                print(f'Simulating fit {index}: {model}, {stage}; {args.simulations} draws...',flush=True)
                # Same seed across fits uses common random numbers for comparisons.
                simulator = simulate_portfolio if 'copula' in fit else simulate_joint_portfolio
                metrics, draws = simulator(fit,w,args.quantiles,args.risk_levels or [],args.simulations,
                    args.seed,args.mc_batches,args.allow_log_linear_combination)
                row = dict(fit_index=index,model=model,stage=stage,window=fit.get('window'),
                    first_date=fit.get('first_date'),last_date=fit.get('last_date'),observations=fit.get('observations'),
                    return_type=fit.get('return_type','unspecified'),weights=json.dumps(dict(zip(symbols,w))),**metrics)
                if fit.get('return_type') == 'log': print('Warning: simulated linear combination of log returns, NOT portfolio log return.')
                for level in args.risk_levels or []:
                    if metrics[f'tail_count_{level:g}'] < 100: print(f'Warning: only {metrics[f"tail_count_{level:g}"]} tail draws at {level:g}; increase --simulations.')
                    if not np.isfinite(metrics[f'es_{level:g}']): print('Warning: '+metrics[f'es_status_{level:g}'])
                if metrics['endpoint_rounding']: print('Warning: CDF roundoff endpoints occurred; extreme quantiles may be truncated and ES is suppressed.')
                rows.append(row)
                if args.show_plot: plots.append((f'{index}: {model}, {stage}, window={fit.get("window")}',draws))
                continue
            distribution = project_distribution(fit, w, allow_log=args.allow_log_linear_combination)
            if fit.get('return_type') == 'log':
                print('Warning: reporting a weighted sum of asset log returns, NOT the portfolio log return.')
            row = dict(fit_index=index, model=fit['model'], window=fit.get('window'), first_date=fit.get('first_date'),
                       last_date=fit.get('last_date'), observations=fit.get('observations'),
                       return_type=fit.get('return_type', 'unspecified'), mean=float(distribution.mean()),
                       volatility=float(distribution.std()), weights=json.dumps(dict(zip(symbols, w))),
                       method='numerical-projection' if distribution.__class__.__name__ in {'ProjectedPower','ProjectedSlash'} else 'analytic-family')
            for q in dict.fromkeys(args.quantiles): row[f'q_{q:g}'] = float(distribution.ppf(q))
            for level in dict.fromkeys(args.risk_levels or []):
                risk = portfolio_risk(distribution, level)
                for key, value in risk.items(): row[f'{key}_{level:g}'] = value
            rows.append(row)
            plots.append((f'{index}: {fit["model"]}, window={fit.get("window")}', distribution))
        if not rows: raise ValueError('No usable fits selected')
        result = pd.DataFrame(rows)
        print('\nPortfolio distribution (per input return period; quantiles are returns, not losses):')
        view = result.drop(columns='weights').copy()
        risk_columns = [f'{key}_{level:g}' for level in dict.fromkeys(args.risk_levels or []) for key in ['var', 'es']]
        mc_columns = [c for c in view if c.endswith('_mc_se') and not c.startswith('q_')]
        for col in ['mean', 'volatility', *[c for c in view if c.startswith('q_')], *risk_columns,*mc_columns]:
            view[col] = view[col].map(lambda v: f'{v:.2%}' if np.isfinite(v) else ('infinite' if np.isposinf(v) else 'n/a'))
        print(view.to_string(index=False, na_rep='n/a'))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f'Wrote {args.output}\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        print('Volatility is not annualized. Undefined moments remain unavailable. Model parameter uncertainty is not included.')
        if args.risk_levels:
            print('VaR/ES use positive-loss convention, per input period. Negative values are not clipped. Joint Student-t ES is infinite for df <= 1; copula df alone does not determine ES existence.')
        if any('copula' in fit or fit.get('model') in SDB_MODELS for fit in fits):
            print('MC SEs are approximate independent-batch errors, not parameter/model uncertainty. n/a if too few batch tail draws, endpoint rounding, or ES has no established finite variance. More draws may be needed.')
        if args.show_plot:
            import matplotlib.pyplot as plt
            for label, distribution in plots:
                if isinstance(distribution,np.ndarray):
                    if np.ptp(distribution) == 0: plt.axvline(distribution[0],label=label+' (point mass)')
                    else: plt.hist(distribution,bins=100,density=True,histtype='step',label=label)
                    continue
                lo, hi = distribution.ppf([.001, .999])
                if lo == hi:
                    plt.axvline(lo, label=label+' (point mass)')
                else:
                    grid = np.linspace(lo, hi, 300)
                    plt.plot(grid, distribution.pdf(grid), label=label)
            plt.xlabel('Linear-combination return')
            plt.ylabel('Density')
            plt.legend(fontsize='small')
            plt.tight_layout()
            plt.show()
        return 0
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
