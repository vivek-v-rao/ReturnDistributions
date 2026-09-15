"""Portfolio risk from in-memory joint fits and their common historical sample."""
import json
import numpy as np
import pandas as pd
from .projection import project_distribution
from .tail_risk import portfolio_risk
from .copula_portfolio import empirical_es
from .joint_portfolio_mc import simulate_joint_portfolio, MC_JOINT_MODELS


def weight_vector(symbols, weights):
    symbols=[str(s).upper() for s in symbols]
    if len(set(symbols))!=len(symbols): raise ValueError('Duplicate asset symbols')
    missing=set(weights)-set(symbols)
    if missing: raise ValueError('Unknown weight symbols: '+', '.join(sorted(missing)))
    return np.array([weights.get(s,0.) for s in symbols],dtype=float)


def risk_report(fits, sample, weights, levels, *, allow_log=False,
                simulations=100000, seed=12345, batches=20):
    """One row per model plus empirical; numerical failures stay visible."""
    if not fits or sample.empty: raise ValueError('No fits or common observations')
    if fits[0].get('return_type')=='log' and not allow_log:
        raise ValueError('Weighted log returns require --allow-log-linear-combination')
    if not levels or any(not np.isfinite(c) or not 0<c<1 for c in levels):
        raise ValueError('Risk levels must be strictly between 0 and 1')
    levels=list(dict.fromkeys(levels))
    w=weight_vector(sample.columns,weights)
    if not np.isfinite(w).all() or not np.isfinite(sample.to_numpy()).all():
        raise ValueError('Risk report requires finite weights and complete returns')
    meta=dict(window=fits[0].get('window'),observations=len(sample),
              first_date=str(sample.index[0].date()),last_date=str(sample.index[-1].date()),
              return_type=fits[0].get('return_type','unspecified'),
              subperiods=fits[0].get('subperiods',1), block=fits[0].get('block',1),
              weights=json.dumps(weights),net_exposure=float(w.sum()),gross_exposure=float(np.abs(w).sum()))
    rows=[]
    for fit in fits:
        row=dict(meta,model=fit['model'],status='ok',fit_status=fit.get('status'),error='')
        for key in ['location_method','fit_label','estimation_method','asset_order','selected_for_display','components']:
            if key in fit: row[key]=fit[key]
        if fit.get('status')!='ok':
            row.update(status='skipped',error='Fit status: '+str(fit.get('status')))
            rows.append(row)
            continue
        try:
            fw=weight_vector(fit['symbols'],weights)/fit.get('return_scale', 1.)
            if fit['model'] in MC_JOINT_MODELS:
                metrics,_=simulate_joint_portfolio(fit,fw,[],levels,simulations,seed,batches,allow_log)
                row.update(metrics)
            else:
                dist=project_distribution(fit,fw,allow_log=allow_log)
                row['method']='projected-family'
                for c in levels:
                    try:
                        risk=portfolio_risk(dist,c)
                        row.update({f'{key}_{c:g}':value for key,value in risk.items()})
                    except (ValueError,ArithmeticError) as exc:
                        row['status']='failed'
                        row['error']+=f'{c:g}: {exc}; '
        except (ValueError,ArithmeticError,KeyError) as exc:
            row.update(status='failed',error=str(exc))
        rows.append(row)
    conditional = fits[0].get('vol_standardization') == 'ewma'
    if conditional:
        from .vol_standardization import conditional_weights
        order = fits[0]['symbols']
        empirical_weights = conditional_weights(fits[0], weight_vector(order, weights))/fits[0].get('return_scale', 1.)
        values = sample.loc[:, order].to_numpy()@empirical_weights
    else:
        values=sample.to_numpy()@w/fits[0].get('return_scale', 1.)
    row=dict(meta,model='empirical',status='ok',fit_status='not applicable',error='',method='historical')
    if any('selected_for_display' in f for f in fits): row['selected_for_display'] = True
    for c in levels:
        row[f'var_{c:g}']=-float(np.quantile(values,1-c))
        row[f'es_{c:g}']=empirical_es(values,1-c)
        row[f'es_status_{c:g}']='finite empirical estimate'
        row[f'es_method_{c:g}']='empirical fractional-tail mean'
        row[f'tail_mass_{c:g}']=(1-c)*len(values)
    rows.append(row)
    result=pd.DataFrame(rows)
    result['return_scale'] = fits[0].get('return_scale', 1.)
    if 'vol_standardization' in fits[0]:
        result['vol_standardization'] = fits[0]['vol_standardization']
        result['risk_basis'] = 'unconditional historical'
    if conditional:
        result['risk_basis'] = 'next-period conditional; empirical = filtered historical simulation'
        result['vol_lambda'] = fits[0]['vol_lambda'] if 'vol_lambda' in fits[0] else np.nan
    for c in levels:
        for key in ['var','es']:
            if f'{key}_{c:g}' not in result: result[f'{key}_{c:g}']=np.nan
    return result


def print_risk_report(report, levels):
    from .model_names import display_model_label
    first=report.iloc[0]
    print(f'\nPortfolio VaR and expected shortfall: {first.first_date} to {first.last_date}; {first.observations} common returns')
    print('Positive-loss percentages per input return period (one day for daily data).')
    columns=['model','status']+[f'{key}_{c:g}' for c in levels for key in ['var','es']]
    if 'vol_standardization' in report: columns.insert(1, 'vol_standardization')
    if 'vol_lambda' in report: columns.insert(2, 'vol_lambda')
    if 'location_method' in report: columns.insert(1,'location_method')
    if 'asset_order' in report: columns.insert(1,'asset_order')
    view=report[columns].copy()
    if 'components' in report:
        view['model'] = [f'{name} [{int(k)} components]' if pd.notna(k) and k > 1 else name
                         for name, k in zip(view.model, report.components)]
    if 'location_method' in view: view['location_method']=view.location_method.fillna('')
    if 'asset_order' in view: view['asset_order']=view.asset_order.fillna('')
    for c in levels:
        for key in ['var','es']:
            column=f'{key}_{c:g}'
            view[column]=view[column].map(lambda v: f'{v:.2%}' if np.isfinite(v) else ('infinite' if np.isposinf(v) else 'n/a'))
    view=view.rename(columns={f'{key}_{c:g}':f'{key.upper()}_{100*c:g}%' for c in levels for key in ['var','es']})
    view['model'] = view['model'].map(display_model_label)
    print(view.to_string(index=False))
    if 'asset_order' in report:
        ordered = report.loc[report.asset_order.notna() & report.asset_order.ne('order-invariant')]
        for model, group in ordered.groupby('model', sort=False):
            if len(group) < 2: continue
            accepted = group.loc[group.status.eq('ok')]
            print(f'\n{model}: risk range across {len(accepted)}/{len(group)} successful orderings (includes Monte Carlo noise):')
            ranges = []
            for c in levels:
                for measure in ['var', 'es']:
                    values = accepted[f'{measure}_{c:g}'].dropna()
                    if len(values):
                        ranges.append(dict(measure=f'{measure.upper()}_{100*c:g}%',
                                           minimum=values.min(), maximum=values.max()))
            if ranges:
                print(pd.DataFrame(ranges).to_string(index=False, float_format=lambda v: f'{v:.2%}'))
    for _,row in report.iterrows():
        label = row.get('fit_label')
        if pd.isna(label): label = row.model
        label = display_model_label(label)
        if row.error: print(f'Warning: {label}: {row.error}')
        if row.get('method') in {'sdb-monte-carlo', 'generalized-t-monte-carlo', 'skew-power-monte-carlo'}:
            print(f'{label}: simulated risk estimates; {int(row.simulations)} draws, seed {int(row.seed)}.')
            for c in levels:
                def show(value):
                    return f'{value:.4%}' if np.isfinite(value) else 'n/a'
                print(f'  {100*c:g}% Monte Carlo SE: VaR {show(row.get(f"var_{c:g}_mc_se", np.nan))}; '
                      f'ES {show(row.get(f"es_{c:g}_mc_se", np.nan))}.')
            print('  Monte Carlo SE excludes parameter uncertainty; n/a means insufficient tail draws or undefined variance.')
    print('Empirical tail mass: '+', '.join(f'{100*c:g}%: {(1-c)*first.observations:.2f} observations' for c in levels)+'.')
    print('Empirical VaR uses linear quantile interpolation; ES uses a fractional-weight tail mean.')
    if 'risk_basis' in report and str(first.risk_basis).startswith('next-period'):
        print('Next-period conditional risk; empirical row is filtered historical simulation at next-period EWMA scales. Parameter/filter uncertainty is not included.')
    else:
        print('Unconditional historical risk, not a forecast conditioned on current volatility; parameter uncertainty is not included.')
