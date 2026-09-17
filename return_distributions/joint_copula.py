"""Opt-in two-stage copulas on the joint CLI's already selected sample."""
import time
import numpy as np
import pandas as pd
from .fitting import fit_many, fitted_distribution
from .copulas import fit_copula
from .skew_t_copula import print_details
from .fit_timeout import run_fit, FitTimeout
from .copula_portfolio import simulate_portfolio
from .table_format import aligned_table
from .model_names import display_model_label
from .univariate_analysis import print_risk
from .copula_portfolio import empirical_es


def fit_block(sample, args, metadata, scales=None, next_scale=None, weights=None, vol_parameters=None):
    records, summaries, audit, risks, cache = [], [], [], [], {}
    marginals, uniforms = [], []
    n = len(sample)
    preparation_started = time.perf_counter()
    try:
        for symbol in sample:
            print(f'Fitting copula marginal {symbol}...', flush=True)
            table = fit_many(sample[symbol].to_numpy(), args.marginal_models, location=args.location,
                             max_iterations=args.max_iterations, fit_timeout=args.fit_timeout)
            cache[symbol] = table.copy()
            audit.extend(dict(row, **metadata, symbol=symbol) for row in table.to_dict('records'))
            good = table.loc[table.status.eq('ok') & np.isfinite(table[args.marginal_criterion])]
            if good.empty: raise ValueError(f'No successful marginal for {symbol}')
            row = good.sort_values(args.marginal_criterion).iloc[0].to_dict()
            row['symbol'] = symbol
            marginals.append(row)
            uniforms.append(fitted_distribution(row).cdf(sample[symbol]))
            print(f'  {symbol}: {display_model_label(row["name"])}')
        u = np.column_stack(uniforms)
        if not np.isfinite(u).all(): raise ValueError('Nonfinite marginal CDF values')
        clipped = int(((u < args.cdf_clip) | (u > 1-args.cdf_clip)).sum())
        u = np.clip(u, args.cdf_clip, 1-args.cdf_clip)
    except Exception as exc:
        print(f'Copula marginal fitting failed: {exc}')
        for model in args.copulas:
            summary = dict(metadata, copula=model, status='failed', error=str(exc), stage='two-stage')
            summaries.append(summary)
            records.append(dict(metadata, symbols=list(sample), stage='two-stage', marginal_mode='fitted',
                                marginals=marginals, copula=dict(model=model, status='failed', error=str(exc)), summary=summary))
        return records, summaries, audit, risks, cache
    adjustment = 0. if scales is None else float(np.log(scales.to_numpy()).sum())
    print(f'Copula marginal fitting and CDF transforms elapsed: {time.perf_counter()-preparation_started:.3f} seconds', flush=True)
    print(f'Marginal CDF entries clipped: {clipped}/{u.size}; threshold {args.cdf_clip:g}')
    if clipped: print('Copula likelihood uses clipped marginal CDFs; extreme observations are approximated.')
    marginal_ll = sum(r['loglik'] for r in marginals)-adjustment
    for model in args.copulas:
        print(f'Fitting {model} copula...', flush=True)
        fit_started = time.perf_counter()
        try:
            fit = run_fit(fit_copula, u, model, args.max_iterations, fit_timeout=args.fit_timeout)
        except Exception as exc:
            fit = dict(model=model, status='timeout' if isinstance(exc, FitTimeout) else 'failed', error=str(exc))
        fit_elapsed = time.perf_counter()-fit_started
        print(f'  {model} copula fitting elapsed: {fit_elapsed:.3f} seconds', flush=True)
        row = dict(metadata, copula=model, status=fit['status'], stage='two-stage',
                   marginal_models='; '.join(f'{r["symbol"]}={r["name"]}' for r in marginals),
                   copula_df=fit.get('df'), copula_loglik=fit.get('loglik'), copula_aic=fit.get('aic'),
                   copula_bic=fit.get('bic'), clipped_entries=clipped, error=fit.get('error',''), copula_fit_sec=fit_elapsed)
        if fit.get('loglik') is not None:
            k = sum(int(r['k']) for r in marginals)+fit['parameters']
            k += sum(p['parameters'] for p in (vol_parameters or {}).values())
            ll = marginal_ll+fit['loglik']
            row.update(marginal_loglik=marginal_ll, joint_loglik=ll, total_parameters=k,
                       two_stage_aic=2*k-2*ll, two_stage_bic=np.log(n)*k-2*ll)
        record = dict(metadata, symbols=list(sample), marginal_mode='fitted', stage='two-stage',
                      marginals=marginals, copula=fit, summary=row, cdf_clip=args.cdf_clip)
        if next_scale is not None: record['next_volatility'] = next_scale.tolist()
        if vol_parameters: record['vol_parameters'] = vol_parameters
        records.append(record)
        summaries.append(row)
        if 'correlation' in fit:
            print('\nLatent copula ' + ('scatter correlation' if model == 'azzalini-skew-t' else 'correlation') + ' (not raw return correlation):')
            print(pd.DataFrame(fit['correlation'], index=sample.columns, columns=sample.columns).to_string(float_format=lambda v:f'{v:.3f}'))
        print_details(fit, list(sample))
        if weights is not None:
            risk = dict(name=f'{model} copula', status=fit['status'])
            if fit['status'] == 'ok':
                print(f'Simulating {model} copula portfolio risk ({args.simulations} draws)...', flush=True)
                risk_started = time.perf_counter()
                try:
                    w = np.array([weights.get(s,0.) for s in sample])/args.return_scale
                    if next_scale is not None: w *= next_scale.to_numpy()
                    metrics, _ = simulate_portfolio(record, w, [], args.risk_levels or [.95,.975,.99,.995],
                        args.simulations, args.seed, args.mc_batches, args.allow_log_linear_combination)
                    risk.update(metrics)
                    for c in args.risk_levels or [.95,.975,.99,.995]:
                        risk[f'VaR_{100*c:g}%'] = metrics[f'var_{c:g}']
                        risk[f'ES_{100*c:g}%'] = metrics[f'es_{c:g}']
                except Exception as exc: risk.update(status='failed',error=str(exc))
                risk['simulation_sec'] = time.perf_counter()-risk_started
                print(f'  Portfolio simulation elapsed: {risk["simulation_sec"]:.3f} seconds', flush=True)
            risks.append(dict(risk, **metadata))
    print('\nTwo-stage copula comparison (separate from joint-model AIC/BIC ranks):')
    print(aligned_table(pd.DataFrame(summaries)))
    print('Two-stage scores are descriptive plug-in criteria, not joint MLE; marginal selection uncertainty is not counted.')
    if risks:
        levels = args.risk_levels or [.95,.975,.99,.995]
        w = np.array([weights.get(s,0.) for s in sample])/args.return_scale
        if next_scale is not None: w *= next_scale.to_numpy()
        values = sample.to_numpy()@w
        empirical = dict(name='empirical', status='ok', **metadata)
        for c in levels:
            empirical[f'VaR_{100*c:g}%'] = -float(np.quantile(values,1-c))
            empirical[f'ES_{100*c:g}%'] = empirical_es(values,1-c)
        risks.append(empirical)
        print_risk(risks, metadata['vol_standardization'], levels, n, title='Copula portfolio VaR and expected shortfall')
        print(f'Copula risk: {args.simulations} simulations; seed {args.seed}; Monte Carlo uncertainty applies.')
    return records, summaries, audit, risks, cache
