"""Descriptive fitted-distribution distances, not hypothesis tests.

JS uses base 2; KL uses natural logs (nats). Numerical failures are explicit.
Univariate integration is in standardized return coordinates, not a histogram.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special

from .fitting import fitted_distribution
from .multivariate import joint_distribution


def add_distance_options(parser):
    parser.add_argument('--js-distance', action='store_true', help='Pairwise Jensen-Shannon distances (base 2, square root)')
    parser.add_argument('--ks-distance', action='store_true', help='Univariate fitted-CDF Kolmogorov-Smirnov distances, not p-values')
    parser.add_argument('--kl-divergence', action='store_true', help='Directional univariate KL divergence in nats; rows P, columns Q')


def selected_metrics(args):
    return [name for name, enabled in [('js', args.js_distance), ('ks', args.ks_distance),
                                     ('kl', args.kl_divergence)] if enabled]


def js_integrand(logp, logq):
    # Conditional label entropy under the equal mixture; nonnegative and <=1.
    u = special.expit(np.asarray(logp)-np.asarray(logq))
    return np.maximum(0., 1+(special.xlogy(u, u)+special.xlog1py(1-u, -u))/np.log(2.))


def univariate_logpdf(dist, x):
    # Some SciPy versions calculate Laplace logpdf by first exponentiating;
    # that underflows in otherwise harmless quadrature tails.
    if getattr(getattr(dist, 'dist', None), 'name', None) == 'laplace':
        loc = dist.kwds.get('loc', dist.args[0] if len(dist.args) else 0.)
        scale = dist.kwds.get('scale', dist.args[1] if len(dist.args)>1 else 1.)
        return -abs((x-loc)/scale)-np.log(2.)-np.log(scale)
    return float(dist.logpdf(x))


def univariate_distance(p, q, metric):
    if metric not in {'js', 'ks', 'kl'}: raise ValueError('Unknown distance metric')
    if metric == 'kl' and getattr(getattr(p, 'dist', None), 'name', None) == 't' \
            and getattr(getattr(q, 'dist', None), 'name', None) == 'norm':
        df = p.kwds.get('df', p.args[0] if p.args else np.nan)
        if df <= 2:
            return np.inf, 0., 'analytic: Student-t has no finite second moment; KL to normal is infinite'
    probabilities = np.unique(np.r_[np.geomspace(1e-9, .1, 35), np.linspace(.1, .9, 41),
                                   1-np.geomspace(1e-9, .1, 35)])
    quantiles = np.r_[p.ppf(probabilities), q.ppf(probabilities)]
    if not np.isfinite(quantiles).all():
        raise ValueError('Nonfinite comparison quantiles')
    center = float((p.ppf(.5)+q.ppf(.5))/2)
    scale = float(max(p.ppf(.75)-p.ppf(.25), q.ppf(.75)-q.ppf(.25),
                      abs(p.ppf(.5)-q.ppf(.5))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid comparison scale')
    support = np.asarray([*p.support(), *q.support()], dtype=float)
    support = support[np.isfinite(support)]
    grid = np.unique((np.r_[quantiles, support]-center)/scale)
    if metric == 'ks':
        def difference(z):
            x = center+scale*z
            result = max(abs(float(p.cdf(x)-q.cdf(x))), abs(float(p.sf(x)-q.sf(x))))
            if not np.isfinite(result): raise ValueError('Nonfinite CDF')
            return result
        values = np.array([difference(z) for z in grid])
        best = values.max()
        for i in range(1, len(grid)-1):
            if values[i] >= values[i-1] and values[i] >= values[i+1]:
                result = optimize.minimize_scalar(lambda z: -difference(z),
                    bounds=(grid[i-1], grid[i+1]), method='bounded', options={'xatol': 1e-11})
                if not result.success: raise ValueError('KS refinement failed')
                best = max(best, -result.fun)
        return float(best), np.nan, 'quantile-grid/local refinement; tail CDF bound 2e-9; no global error guarantee'

    def integrand(z):
        x = center+scale*z
        lp, lq = univariate_logpdf(p, x), univariate_logpdf(q, x)
        if np.isnan(lp) or np.isnan(lq) or np.isposinf(lp) or np.isposinf(lq):
            raise ValueError('Nonfinite log density')
        if lp == -np.inf and lq == -np.inf: return 0.
        if metric == 'js':
            value = np.exp(np.logaddexp(lp, lq)-np.log(2.)+np.log(scale))*js_integrand(lp, lq)
        elif metric == 'kl':
            # Nonnegative form p log(p/q)-p+q avoids cancellation.
            if lp == -np.inf: return float(np.exp(lq+np.log(scale)))
            if lq == -np.inf: raise ValueError('Q density underflow or support mismatch; KL may be infinite')
            delta = lq-lp
            if delta <= 0:
                value = np.exp(lp+np.log(scale))*(-delta+np.expm1(delta))
            else:
                value = np.exp(lq+np.log(scale))*(1-np.exp(-delta)*(1+delta))
        else:
            raise ValueError('Unknown distance metric')
        if not np.isfinite(value): raise ValueError('Nonfinite divergence integrand')
        return float(max(0., value))

    # Split at quantiles to resolve displaced modes and central density peaks.
    integration_probabilities = [1e-6, .001, .01, .1, .25, .5, .75, .9, .99, .999, 1-1e-6]
    knots = np.r_[-np.inf, np.unique((np.r_[p.ppf(integration_probabilities),
                                          q.ppf(integration_probabilities), support]-center)/scale), np.inf]
    value = error = 0.
    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        for a, b in zip(knots[:-1], knots[1:]):
            v, e = integrate.quad(integrand, a, b, epsabs=1e-9/len(knots), epsrel=1e-7, limit=150)
            value += v
            error += e
    if error > max(1e-7, abs(value)*1e-5): raise ValueError('Quadrature tolerance not met; divergence may be infinite')
    if metric == 'js':
        if value > 1+1e-7: raise ValueError('JS integral exceeds theoretical bound')
        distance = np.sqrt(min(1., value))
        uncertainty = max(np.sqrt(value+error)-np.sqrt(value), np.sqrt(value)-np.sqrt(max(0., value-error)))
        return float(distance), float(uncertainty), 'adaptive quadrature; error estimate, not a statistical SE'
    return float(value), float(error), 'adaptive quadrature; error estimate, not a statistical SE'


def print_comparisons(rows, labels, metrics):
    from .model_names import display_model_label
    for metric in metrics:
        subset = [r for r in rows if r['metric'] == metric]
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        errors = matrix.copy()
        for r in subset:
            matrix.loc[r['model_p'], r['model_q']] = r['value']
            errors.loc[r['model_p'], r['model_q']] = r['numerical_error']
        title = {'js': 'Jensen-Shannon distance (base 2; range 0--1)',
                 'ks': 'Kolmogorov-Smirnov distance (CDFs; no p-values)',
                 'kl': 'KL divergence (nats; row P to column Q)'}[metric]
        print('\n'+title+':')
        print(matrix.rename(index=display_model_label, columns=display_model_label).to_string(float_format=lambda x: f'{x:.6f}', na_rep='n/a'))
        if metric != 'ks' and subset:
            print('Numerical uncertainty (quadrature estimates or Monte Carlo SE; excludes fit uncertainty):')
            print(errors.rename(index=display_model_label, columns=display_model_label).to_string(float_format=lambda x: f'{x:.2e}', na_rep='n/a'))
        for r in subset:
            if r['status'] != 'ok': print(f"Warning: {r['model_p']} -> {r['model_q']}: {r['error']}")
    if 'ks' in metrics:
        print('KS uses quantile-grid/local refinement; tail bound 2e-9, not a certified global maximization.')


def compare_univariate(table, metrics):
    from .model_names import display_model_label
    good = table.loc[table.status.eq('ok')]
    for _, row in table.loc[~table.status.eq('ok')].iterrows():
        print(f"Distance comparison excluded {display_model_label(row['name'])}: {row['status']}")
    labels = good['name'].tolist()
    distributions = {}
    failures = {}
    for _, row in good.iterrows():
        try: distributions[row['name']] = fitted_distribution(row)
        except Exception as exc: failures[row['name']] = str(exc)
    rows = []
    for metric in metrics:
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if metric != 'kl' and j < i: continue
                row = dict(metric=metric, model_p=a, model_q=b, value=np.nan,
                           numerical_error=np.nan, method='', status='ok', error='')
                try:
                    if a in failures or b in failures: raise ValueError(failures.get(a, failures.get(b)))
                    if i == j: row.update(value=0., numerical_error=0., method='identity')
                    else:
                        print(f'Comparing {metric.upper()}: {display_model_label(a)} -> {display_model_label(b)}...', flush=True)
                        value, error, method = univariate_distance(distributions[a], distributions[b], metric)
                        row.update(value=value, numerical_error=error, method=method)
                except Exception as exc:
                    row.update(status='failed', error=str(exc))
                rows.append(row)
                if metric != 'kl' and i != j: rows.append(dict(row, model_p=b, model_q=a))
    print_comparisons(rows, labels, metrics)
    return rows


def compare_joint(fits, simulations=100000, seed=12345, batches=20, *, display_selected=False, return_units=False):
    from .model_names import display_model_label
    good = [f for f in fits if f.get('status') == 'ok']
    for f in fits:
        if f.get('status') != 'ok': print(f"Distance comparison excluded {f.get('fit_label', f['model'])}: {f.get('status')}")
    if simulations < 1000 or batches < 2 or simulations//batches < 50 or seed < 0:
        raise ValueError('Invalid JS simulation settings')
    labels = []
    for f in good:
        base = f.get('fit_label', f['model'])
        label = base
        while label in labels: label += ' [duplicate]'
        labels.append(label)
    samples, distributions, failures = {}, {}, {}
    for i, (fit, child) in enumerate(zip(good, np.random.SeedSequence(seed).spawn(len(good)))):
        try:
            dist = joint_distribution(fit)
            if return_units:
                from .vol_standardization import ReturnUnitDistribution
                dist = ReturnUnitDistribution(dist, fit)
            sample = np.asarray(dist.rvs(size=simulations, random_state=np.random.default_rng(child)))
            if sample.shape != (simulations, len(fit['symbols'])) or not np.isfinite(sample).all():
                raise ValueError('Invalid simulated samples; no draws discarded')
            distributions[i], samples[i] = dist, sample
        except Exception as exc: failures[i] = str(exc)
    cache = {}
    def logdensity(model, source):
        key = model, source
        if key not in cache:
            a, b = good[model]['symbols'], good[source]['symbols']
            if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
                raise ValueError('Joint distributions must have the same unique asset symbols')
            values = np.asarray(distributions[model].logpdf(samples[source][:, [b.index(s) for s in a]]))
            if values.shape != (simulations,) or np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError('Invalid joint log density')
            if model == source and not np.isfinite(values).all(): raise ValueError('Nonfinite own-sample density')
            cache[key] = values
        return cache[key]
    rows = []
    for i, a in enumerate(labels):
        for j in range(i, len(labels)):
            b = labels[j]
            row = dict(metric='js', model_p=a, model_q=b, value=np.nan, numerical_error=np.nan,
                       method='equal-mixture Monte Carlo; delta-method distance SE', status='ok', error='',
                       simulations=simulations, seed=seed, mc_batches=batches)
            if return_units:
                row['comparison_units'] = 'original returns; volatility filters at next-period scales'
            if display_selected:
                row['selected_p'] = good[i].get('selected_for_display', True)
                row['selected_q'] = good[j].get('selected_for_display', True)
                row['selected_for_display'] = row['selected_p'] and row['selected_q']
            try:
                if i in failures or j in failures: raise ValueError(failures.get(i, failures.get(j)))
                if i == j: row.update(value=0., numerical_error=0., method='identity')
                else:
                    if row.get('selected_for_display', True):
                        print(f'Comparing joint JS: {display_model_label(a)} <-> {display_model_label(b)}...', flush=True)
                    vp = js_integrand(logdensity(i, i), logdensity(j, i))
                    vq = js_integrand(logdensity(i, j), logdensity(j, j))
                    values = (vp+vq)/2
                    divergence = float(values.mean())
                    distance = np.sqrt(divergence)
                    bp = np.array([v.mean() for v in np.array_split(vp, batches)])
                    bq = np.array([v.mean() for v in np.array_split(vq, batches)])
                    se_div = float(np.sqrt((bp.var(ddof=1)+bq.var(ddof=1))/(4*batches)))
                    row.update(value=float(distance), numerical_error=se_div/(2*distance) if distance > 0 else 0.)
            except Exception as exc: row.update(status='failed', error=str(exc))
            rows.append(row)
            if i != j:
                reverse = dict(row, model_p=b, model_q=a)
                if display_selected: reverse.update(selected_p=row['selected_q'], selected_q=row['selected_p'])
                rows.append(reverse)
    shown_labels = [label for label, f in zip(labels, good) if not display_selected or f.get('selected_for_display', True)]
    print_comparisons([r for r in rows if r.get('selected_for_display', True)], shown_labels, ['js'])
    for row in rows:
        if not row.get('selected_for_display', True) and row['status'] != 'ok':
            print(f"Warning: undisplayed JS comparison {row['model_p']} -> {row['model_q']}: {row['error']}")
    print(f'Joint JS: {simulations} draws per fit, seed {seed}; SE is approximate, especially near zero. No fit uncertainty included.')
    return rows
