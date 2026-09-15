"""Independent univariate comparisons for joint CLI samples; no new estimators."""
import pandas as pd

from .model_names import unique_models, display_model_label
from .fitting import fit_many
from .table_format import aligned_table


UNIVARIATE_MODELS = {
    name: name for name in (
        'normal', 'student-t', 'laplace', 'ged', 'ged-skewed', 'fs-skew-normal', 'generalized-t', 'generalized-t-skewed', 'nts-symmetric', 'nts-skewed',
        'hyperbolic-symmetric', 'hyperbolic-skewed', 'nig-symmetric', 'nig-skewed',
        'gh-symmetric', 'gh-skewed', 'gh-skew-t',
        'variance-gamma-symmetric', 'variance-gamma-skewed', 'azzalini-skew-t')
}
UNIVARIATE_MODELS.update({
    'noncentral-t': 'nct',
    'sdb-skew-normal': 'skewnorm',
    'sdb-skew-t': 'azzalini-skew-t',
    'laplace-mixture-symmetric': 'laplace',
    'asymmetric-laplace': 'laplace_asymmetric',
})


def model_mapping(models):
    """Group shared one-dimensional counterparts so each is fit only once."""
    mapped, skipped = {}, []
    for model in unique_models(models):
        name = UNIVARIATE_MODELS.get(model)
        if name is None:
            skipped.append(model)
        else:
            mapped.setdefault(name, []).append(model)
    return mapped, skipped


def format_univariate_comparison(table):
    """Compact screen view; preserve all fields in returned/saved fit tables."""
    columns = list(table.columns)
    if 'joint_models' in columns:
        columns = columns[:columns.index('joint_models')]
    view = table[columns].drop(columns=['scipy_distribution', 'n', 'ks_p'], errors='ignore')
    if not table.empty and 'converged' in view and view['converged'].eq(True).fillna(False).all():
        view = view.drop(columns='converged')
    lines = []
    if not table.empty and 'status' in view and view['status'].eq('ok').fillna(False).all():
        lines.append('Status: all fits ok')
        view = view.drop(columns='status')
    def clean(value):
        return '' if pd.isna(value) else str(value).strip()

    if 'warnings' not in view or not view['warnings'].map(clean).any():
        view = view.drop(columns='warnings', errors='ignore')
        lines.append('Warnings: none')
    diagnostics = []
    for column, label in [('optimizer_message', 'Optimizer messages'), ('fit_restriction', 'Fit restrictions')]:
        groups = {}
        if column in view:
            for _, row in table.iterrows():
                message = clean(row.get(column))
                if not message:
                    continue
                if column == 'optimizer_message':
                    parts = [part.strip() for part in message.split(';') if part.strip()]
                    converged = row.get('converged')
                    if not pd.isna(converged) and bool(converged):
                        parts = [part for part in parts if not (
                            part.rstrip('.').lower() == 'optimization terminated successfully'
                            or part.upper().startswith('CONVERGENCE:')
                            or part.startswith('Analytic '))]
                    message = '; '.join(parts)
                if message:
                    groups.setdefault(message, []).append(display_model_label(str(row['name'])))
            view = view.drop(columns=column)
        if groups:
            diagnostics.append(label + ':')
            diagnostics.extend('  '+', '.join(names)+': '+message for message, names in groups.items())
    lines.append(aligned_table(view))
    if diagnostics:
        lines.extend(['', *diagnostics])
    return '\n'.join(lines)


def fit_univariate_sample(sample, mapped, window, *, location=None, max_iterations=2000, return_type='simple', fit_timeout=None, vol_scales=None, cached_fits=None):
    """Use the already-selected complete-case sample, with no new data filtering."""
    tables = []
    for symbol in sample:
        print(f'\nSeparately estimated univariate fits - {symbol}: {len(sample)} common returns; '
              f'{sample.index[0].date()} to {sample.index[-1].date()}', flush=True)
        cached = (cached_fits or {}).get(symbol)
        if cached is None:
            table = fit_many(sample[symbol].to_numpy(), list(mapped), location=location,
                             max_iterations=max_iterations, fit_timeout=fit_timeout)
        else:
            table = cached.loc[cached.name.isin(mapped)].copy()
            missing = [m for m in mapped if m not in set(table.name)]
            if missing:
                table = pd.concat([table, fit_many(sample[symbol].to_numpy(), missing, location=location,
                                  max_iterations=max_iterations, fit_timeout=fit_timeout)], ignore_index=True)
            from .fit_ranks import criterion_ranks
            table = criterion_ranks(table).sort_values('aic_rank')
        if vol_scales is not None:
            import numpy as np
            adjustment = float(np.log(vol_scales.loc[sample.index, symbol]).sum())
            table['standardized_loglik'] = table['loglik']
            table['loglik'] -= adjustment
            for key in ('aic', 'bic'):
                table[key] += 2*adjustment
            print('Parameters/moments/KS: standardized units; loglik/AIC/BIC: original return units, conditional on fixed EWMA.')
        table['joint_models'] = table['name'].map(lambda name: ' '.join(mapped[name]))
        table['symbol'] = symbol
        table['window'] = window if window is not None else 'all'
        table['first_date'] = str(sample.index[0].date())
        table['last_date'] = str(sample.index[-1].date())
        table['return_type'] = return_type
        table['sample_policy'] = 'joint-complete-case'
        if vol_scales is not None:
            table['parameter_units'] = 'standardized returns'
            table['likelihood_units'] = 'original returns; conditional on fixed EWMA filter'
        print(format_univariate_comparison(table))
        tables.append(table)
    return tables
