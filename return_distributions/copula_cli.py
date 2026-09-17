"""Two-stage marginal and Gaussian/Student-t/AC skew-t copula fitting on common dates."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from .data import read_returns
from .fitting import fit_many, fitted_distribution
from .copulas import fit_copula, COPULA_MODELS
from .skew_t_copula import print_details
from .copula_refinement import refine_joint
from .table_format import aligned_table


def clean_json(value):
    if isinstance(value, dict): return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--days', type=int, nargs='+')
    parser.add_argument('--input-type', choices=['prices', 'returns'], default='prices')
    parser.add_argument('--return-type', choices=['simple', 'log'], default='simple')
    parser.add_argument('--copulas', choices=COPULA_MODELS, nargs='+', default=['gaussian', 'student-t'])
    parser.add_argument('--marginal-mode', choices=['fitted', 'ranks'], default='fitted')
    parser.add_argument('--marginal-models', nargs='+', default=['student-t'], help='Candidate univariate families for each asset')
    parser.add_argument('--marginal', action='append', default=[], metavar='SYMBOL=MODEL', help='Override an asset marginal family')
    parser.add_argument('--criterion', choices=['aic', 'bic'], default='aic', help='Marginal selection criterion')
    parser.add_argument('--cdf-clip', type=float, default=1e-10)
    parser.add_argument('--max-iterations', type=int, default=1000)
    parser.add_argument('--joint-refine', action='store_true', help='Jointly refine marginal/copula parameters from two-stage estimates')
    parser.add_argument('--output', type=Path, default=Path('copula_fits.json'))
    parser.add_argument('--no-save', action='store_true', help='Screen output only; do not write fit, summary, or marginal audit files')
    args = parser.parse_args(argv)
    if args.joint_refine and 'azzalini-skew-t' in args.copulas:
        parser.error('--joint-refine currently supports only Gaussian and Student-t copulas')
    if args.joint_refine and args.marginal_mode == 'ranks': parser.error('--joint-refine requires fitted marginals, not ranks')
    if args.days and min(args.days) < 8: parser.error('Windows must be >=8')
    if args.max_iterations < 1 or not 0 < args.cdf_clip < .01: parser.error('Positive iterations and 0 < cdf-clip < .01 required')
    if not args.no_save:
        if args.output.suffix.lower() != '.json': parser.error('--output must end in .json')
        paths = [args.output, args.output.with_suffix('.csv'), args.output.with_name(args.output.stem+'_marginals.csv')]
        if args.file.resolve() in {p.resolve() for p in paths}: parser.error('Output must not overwrite input')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions.copula_cli', *(sys.argv[1:] if argv is None else argv)]))
    try:
        returns = read_returns(args.file, args.symbols, args.input_type, args.return_type)
        symbols = list(returns.columns)
        if len(symbols) < 2: raise ValueError('Need at least two assets')
        overrides = {}
        for entry in args.marginal:
            if '=' not in entry: raise ValueError('Marginal override requires SYMBOL=MODEL')
            symbol, model = entry.split('=', 1)
            if symbol not in symbols or symbol in overrides: raise ValueError('Unknown or duplicate marginal override symbol')
            overrides[symbol] = model
        if args.marginal_mode == 'ranks' and args.marginal: raise ValueError('Rank mode does not use parametric overrides')
        complete = returns.dropna()
        records, summaries, marginal_audit = [], [], []
        print('Common complete-case sample; initial two-stage estimation, NOT joint maximum likelihood.' + (' Joint refinement follows.' if args.joint_refine else ''))
        for window in dict.fromkeys(args.days or [None]):
            sample = complete if window is None else complete.tail(window)
            n = len(sample)
            if n < max(8, len(symbols)+2): raise ValueError('Insufficient common observations')
            dates = dict(first_date=str(sample.index[0].date()), last_date=str(sample.index[-1].date()))
            print(f'\n{window or "All"} periods: {n} common returns; {dates["first_date"]} to {dates["last_date"]}')
            if window and n < window: print('Warning: fewer common observations than requested')
            marginals, marginal_ll, marginal_k = [], 0., 0
            preparation_started = time.perf_counter()
            if args.marginal_mode == 'ranks':
                u = np.column_stack([stats.rankdata(sample[s], method='average')/(n+1) for s in symbols])
                ties = {s: n-sample[s].nunique() for s in symbols}
                print('Rank pseudo-observations use average ties and rank/(n+1); duplicate counts: ' + str(ties))
            else:
                columns = []
                for symbol in symbols:
                    print(f'Fitting marginal {symbol}...', flush=True)
                    candidates = [overrides[symbol]] if symbol in overrides else args.marginal_models
                    fits = fit_many(sample[symbol].to_numpy(), candidates, max_iterations=args.max_iterations)
                    marginal_audit.extend([dict(row, symbol=symbol, window=window, **dates) for row in fits.to_dict('records')])
                    good = fits.loc[fits.status.eq('ok')].sort_values(args.criterion)
                    if good.empty: raise ValueError(f'No successful marginal for {symbol}; try another family/iteration limit')
                    row = good.iloc[0].to_dict()
                    row['symbol'] = symbol
                    marginals.append(row)
                    marginal_ll += row['loglik']
                    marginal_k += int(row['k'])
                    columns.append(fitted_distribution(row).cdf(sample[symbol]))
                    print(f'  {row["name"]}: {args.criterion}={row[args.criterion]:.3f}')
                u = np.column_stack(columns)
            if not np.isfinite(u).all(): raise ValueError('Nonfinite marginal CDF values')
            clipped = int(((u < args.cdf_clip) | (u > 1-args.cdf_clip)).sum())
            u = np.clip(u, args.cdf_clip, 1-args.cdf_clip)
            print(f'CDF entries clipped: {clipped}/{u.size}; threshold {args.cdf_clip:g}')
            print(f'Copula marginal preparation elapsed: {time.perf_counter()-preparation_started:.3f} seconds', flush=True)
            for model in dict.fromkeys(args.copulas):
                print(f'Fitting {model} copula...', flush=True)
                fit_started = time.perf_counter()
                try:
                    fit = fit_copula(u, model, args.max_iterations)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    fit = dict(model=model, status='failed', error=str(exc))
                print(f'  {model} copula fitting elapsed: {time.perf_counter()-fit_started:.3f} seconds', flush=True)
                record = dict(symbols=symbols, window=window, **dates, observations=n, return_type=args.return_type,
                              marginal_mode=args.marginal_mode, cdf_clip=args.cdf_clip, clipped_entries=clipped,
                              marginals=marginals, copula=fit)
                row = dict(window=window, **dates, observations=n, copula=model, status=fit['status'],
                           copula_loglik=fit.get('loglik'), copula_aic=fit.get('aic'), copula_bic=fit.get('bic'),
                           copula_df=fit.get('df'), clipped_entries=clipped)
                if args.marginal_mode == 'fitted' and 'loglik' in fit:
                    ll, k = marginal_ll+fit['loglik'], marginal_k+fit['parameters']
                    row.update(marginal_loglik=marginal_ll, total_parameters=k, joint_loglik=ll,
                               two_stage_aic=2*k-2*ll, two_stage_bic=np.log(n)*k-2*ll)
                record['summary'] = row
                record['stage'] = row['stage'] = 'two-stage'
                records.append(record)
                summaries.append(row)
                if 'correlation' in fit:
                    print('\nLatent copula ' + ('scatter correlation' if model == 'azzalini-skew-t' else 'correlation') + ' (not raw return correlation):')
                    print(pd.DataFrame(fit['correlation'], index=symbols, columns=symbols).to_string(float_format=lambda v: f'{v:.3f}'))
                print_details(fit, symbols)
                if args.joint_refine:
                    print(f'Jointly refining {model} copula and marginals...', flush=True)
                    try:
                        refined = refine_joint(sample.to_numpy(), record, args.max_iterations)
                        records.append(refined)
                        summaries.append(refined['summary'])
                        info = refined['refinement']
                        print(f'  accepted={info["accepted"]}; converged={info["converged"]}; boundary={info["boundary"]}; likelihood gain={info["improvement"]:.6g}; {info["fit_sec"]:.3f}s')
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        record['refinement'] = dict(accepted=False, error=str(exc))
                        row['refinement_accepted'] = False
                        row['refinement_error'] = str(exc)
                        print(f'Warning: refinement skipped; original fit retained: {exc}')
        summary = pd.DataFrame(summaries)
        print('\nCopula fit comparison:')
        print(aligned_table(summary))
        if not args.no_save:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(clean_json(records), indent=2, allow_nan=False), encoding='utf-8')
            summary.to_csv(paths[1], index=False)
            if marginal_audit: pd.DataFrame(marginal_audit).to_csv(paths[2], index=False)
            print(f'Wrote {args.output} and summary CSV' + (' and marginal audit CSV' if marginal_audit else ''))
        print('Compare copula scores only on identical marginal transforms and observations. Rank scores are pseudo-likelihoods, not return likelihoods.')
        print('Two-stage AIC/BIC are descriptive plug-in criteria, not jointly maximized-likelihood criteria; marginal family selection uncertainty is not counted.')
        if args.joint_refine:
            print('Joint refinement holds marginal families fixed; accepted fits have joint AIC/BIC. Marginal dfs and copula df remain separate. Compare full joint likelihoods, not copula-only scores across stages.')
        print(f'Overall elapsed: {time.perf_counter()-start:.3f} seconds')
        return 0 if summary.status.eq('ok').all() else 1
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
