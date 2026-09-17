"""Sample alignment and tail reporting for standalone univariate fits."""
import numpy as np
import pandas as pd
from .vol_standardization import ewma_standardize
from .fitting import fitted_distribution, model_catalog
from .model_names import display_model_name, unique_models, display_model_label
from .tail_risk import portfolio_risk
from .copula_portfolio import empirical_es
from .sample_blocks import blocks, date_mask, endpoint_scale


def all_project_models():
    return unique_models(sorted({display_model_name(m) for m in model_catalog()['Project models and aliases']}))


def prepare_samples(frame, modes, decays, warmup, floor, common):
    cache = {v: ewma_standardize(frame, v, warmup, floor) for v in decays} if 'ewma' in modes else {}
    eligible = frame.notna()
    if any(m in modes for m in ('garch', 'nagarch')):
        eligible &= ewma_standardize(frame, .94, warmup, floor)[0].notna()
        cache['_garch'] = {}
    for standardized, _, _ in (v for k, v in cache.items() if k != '_garch'):
        eligible &= standardized.notna()
    if common:
        eligible.loc[~eligible.all(axis=1), :] = False
    return cache, eligible


def samples(frame, symbol, windows, modes, decays, cache, eligible, *, partitions=None, date_min=None, date_max=None, floor=1e-8, warmup=63):
    index = frame.index[eligible[symbol]]
    index = index[date_mask(index, date_min, date_max)]
    for window, count, number, selected in blocks(index, windows, partitions or [1]):
        extra = (count, number) if partitions is not None else ()
        for mode in modes:
            for decay in (decays if mode == 'ewma' else [None]):
                if mode == 'none':
                    yield (window, mode, decay, frame.loc[selected, symbol], 0., 1., *extra)
                elif mode in ('garch', 'nagarch'):
                    from .garch_standardization import fit_selected
                    standardized, scales, next_scales, parameters = fit_selected(
                        frame[[symbol]], selected, mode, warmup, floor)
                    cache['_garch'][(symbol, mode, selected[0], selected[-1])] = parameters
                    yield (window, mode, decay, standardized.loc[selected, symbol],
                           float(np.log(scales.loc[selected, symbol]).sum()), float(next_scales[symbol]), *extra)
                else:
                    standardized, scales, next_scales = cache[decay]
                    next_scale = endpoint_scale(frame, scales, selected[-1], decay, floor)[symbol]
                    yield (window, mode, decay, standardized.loc[selected, symbol], float(np.log(scales.loc[selected, symbol]).sum()), float(next_scale), *extra)


def risk_rows(fits, series, levels, scale):
    rows = []
    for _, fit in fits.iterrows():
        row = dict(name=fit['name'], status=fit['status'])
        if fit['status'] == 'ok':
            try:
                law = fitted_distribution(fit)
                for c in levels:
                    result = law.risk(c) if isinstance(fit.get('mixture_fit'), str) else portfolio_risk(law, c)
                    row[f'VaR_{100*c:g}%'] = scale*result['var']
                    row[f'ES_{100*c:g}%'] = scale*result['es']
                    row[f'ES_status_{100*c:g}%'] = result['es_status']
            except Exception as exc:
                row.update(status='failed', error=str(exc))
        rows.append(row)
    values = np.asarray(series)*scale
    if len(values):
        empirical = dict(name='empirical', status='ok')
        for c in levels:
            empirical[f'VaR_{100*c:g}%'] = -float(np.quantile(values, 1-c))
            empirical[f'ES_{100*c:g}%'] = empirical_es(values, 1-c)
        rows.append(empirical)
    return rows


def print_risk(rows, mode, levels, n, title='Per-asset VaR and expected shortfall'):
    print(f'\n{title} (positive-loss percentages per input period):')
    print('Next-period conditional risk; empirical = filtered historical simulation.' if mode != 'none'
          else 'Unconditional historical risk; empirical = observed returns.')
    columns = ['name', 'status'] + [key for c in levels for key in (f'VaR_{100*c:g}%', f'ES_{100*c:g}%')]
    table = pd.DataFrame(rows).reindex(columns=columns)
    table['name'] = table['name'].map(display_model_label)
    for key in columns[2:]:
        table[key] = table[key].map(lambda v: 'n/a' if pd.isna(v) else f'{100*v:.2f}%')
    print(table.to_string(index=False))
    for row in rows:
        if row.get('error'): print(f'{row["name"]}: {row["error"]}')
    print('Empirical tail mass: '+', '.join(f'{100*c:g}%: {n*(1-c):.2f} observations' for c in levels))
    print('Parameter/filter uncertainty is not included; no annualization. Empirical ES uses fractional tail weights.')
