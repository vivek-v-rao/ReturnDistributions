"""Joint return distribution fitting on common observations."""
import argparse
from .model_names import canonical_model, unique_models
import json
from pathlib import Path
import sys
import time

import pandas as pd

from .data import read_returns
from .multivariate import fit_joint, JOINT_MODELS, ALL_JOINT_MODELS


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--input-type', choices=['prices', 'returns'], default='prices')
    parser.add_argument('--return-type', choices=['simple', 'log'], default='simple')
    parser.add_argument('--days', type=int, nargs='+')
    parser.add_argument('--models', type=canonical_model, choices=ALL_JOINT_MODELS, nargs='+', default=list(JOINT_MODELS), help='Aliases nig, gh, hyperbolic, variance-gamma select skewed variants')
    parser.add_argument('--sdb-cdf-points', type=int, default=512, help='Experimental SDB fitting grid: power of two, 64--16384; audit uses 4x points')
    parser.add_argument('--sdb-seed', type=int, default=12345)
    parser.add_argument('--slash-quadrature-points',type=int,default=64,help='Slash fitting quadrature order, 16--256; audit uses 4x points')
    parser.add_argument('--location', type=float, help='Fix every location component to this value')
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=Path('joint_distribution_fits.json'))
    args = parser.parse_args(argv)
    summary_path = args.output.with_suffix('.csv')
    if args.output.suffix.lower() != '.json':
        parser.error('--output must end in .json; a matching summary CSV is also written')
    if args.file.resolve() in {args.output.resolve(), summary_path.resolve()}:
        parser.error('Output must not overwrite input')
    if args.max_iterations < 1 or args.days and min(args.days) < 8:
        parser.error('Iterations must be positive; windows must be at least 8')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions.joint_cli', *(sys.argv[1:] if argv is None else argv)]))
    try:
        frame = read_returns(args.file, args.symbols, args.input_type, args.return_type)
        if len(frame.columns) < 2:
            raise ValueError('Select at least two assets for the joint CLI')
        complete = frame.dropna()
        data_elapsed = time.perf_counter()-start
        results = []
        print('Same complete-case observations for every joint model; no pairwise deletion.')
        for window in dict.fromkeys(args.days or [None]):
            sample = complete if window is None else complete.tail(window)
            if sample.empty:
                raise ValueError('No common returns')
            first, last = str(sample.index[0].date()), str(sample.index[-1].date())
            print(f'\n{window or "All"} periods: {len(sample)} common returns; {first} to {last}')
            if window and len(sample) < window:
                print('Warning: fewer common returns than requested')
            for model in unique_models(args.models):
                print(f'Fitting joint {model} ({len(sample.columns)} assets)...', flush=True)
                try:
                    fit = fit_joint(sample.to_numpy(), model, location=args.location, max_iterations=args.max_iterations,
                                    sdb_points=args.sdb_cdf_points, sdb_seed=args.sdb_seed, slash_points=args.slash_quadrature_points)
                except (ValueError, ArithmeticError) as exc:
                    fit = dict(model=model, status='failed', error=str(exc), observations=len(sample))
                fit.update(symbols=list(sample.columns), window=window, first_date=first, last_date=last, return_type=args.return_type)
                results.append(fit)
                if 'scatter' in fit:
                    print('Location: ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(sample.columns, fit['location'])))
                    corr = fit.get('correlation') or fit['scatter_correlation']
                    if 'alpha' in fit:
                        print('Skewness alpha: ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(sample.columns, fit['alpha'])))
                    if 'delta' in fit:
                        label = 'SDB skew loadings' if model.startswith('sdb-') else 'Noncentrality delta'
                        print(label+' (return units): ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(sample.columns, fit['delta'])))
                    if 'cdf_audit_passed' in fit:
                        print(f"CDF accuracy audit: passed={fit['cdf_audit_passed']}; loglik change={fit['cdf_loglik_change']:.5g}; max row log-density change={fit['cdf_max_logpdf_change']:.5g}")
                    if 'quadrature_audit_passed' in fit:
                        print(f"Slash quadrature audit: passed={fit['quadrature_audit_passed']}; loglik change={fit['quadrature_loglik_change']:.5g}; max row log-density change={fit['quadrature_max_logpdf_change']:.5g}")
                    print('\nFitted ' + ('scatter correlation (Pearson correlation undefined)' if fit['covariance'] is None else 'return correlation') + ':')
                    print(pd.DataFrame(corr, index=sample.columns, columns=sample.columns).to_string(float_format=lambda v: f'{v:.3f}'))
        keys = ['model', 'window', 'observations', 'dimensions', 'parameters', 'first_date', 'last_date', 'loglik', 'aic', 'bic', 'df', 'q', 'vg_shape', 'power', 'lambda', 'psi', 'status', 'fit_sec', 'error']
        summary = pd.DataFrame([{k: r.get(k) for k in keys} for r in results])
        summary['rank'] = summary.aic.where(summary.status.eq('ok')).groupby(summary.window.fillna('all')).rank(method='min')
        print('\nJoint fit comparison:')
        print(summary.to_string(index=False, float_format=lambda v: f'{v:.5g}', na_rep='n/a'))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        summary.to_csv(summary_path, index=False)
        print(f'Wrote {args.output}\nWrote {summary_path}')
        print('Student-t scatter is not covariance; covariance is unavailable when df <= 2. Boundary fits are unranked.')
        print('GH mixture: chi=1; hyperbolic lambda=(dimension+1)/2; NIG lambda=-1/2. Skewed GH location is not its mean.')
        if any(r['model']=='gh-skew-t' for r in results):
            print('GH skew-t: inverse-gamma mixture; nonzero skew requires df>2 for a finite mean and df>4 for covariance. ES depends on the projected loss-tail direction. Fit bounds: 0.25<=df<=200; boundary fits excluded.')
        if any(r['model'].startswith('variance-gamma-') for r in results):
            print('VG mixture: Gamma(shape=vg_shape, rate=vg_shape); E[W]=1. Restricted MLE: dimension/2+0.05 <= vg_shape <= 100; boundary fits excluded from portfolio risk.')
        if any(r['model'] in {'gh-symmetric','gh-skewed'} for r in results):
            print('Generalized GH estimates lambda in [-20,20]; boundary fits are unranked and excluded from portfolio risk.')
        if any(r['model'].startswith('sdb-') for r in results):
            print('Experimental SDB: numerically approximated likelihood; failed CDF audits are unranked. Portfolio risk uses simulation, not an ordinary univariate skew-t projection.')
        if any('slash' in r['model'] for r in results):
            print('Slash mixtures: location is outside random scaling. Mean/ES require q>1 and df>1 (t); covariance requires q>2 and df>2. Failed quadrature audits and boundary fits are unranked.')
        print('Laplace/GED are elliptical power-exponential: exp(-q^(power/2)); Laplace fixes power=1. Marginals need not be Laplace/GED.')
        print(f'Data elapsed: {data_elapsed:.3f} seconds\nFitting/output elapsed: {time.perf_counter()-start-data_elapsed:.3f} seconds\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        return 0 if summary.status.eq('ok').all() else 1
    except (ValueError, KeyError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
