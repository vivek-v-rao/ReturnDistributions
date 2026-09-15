"""SciPy maximum-likelihood fits adapted from xscipy_dist_returns.py.

Parameters and likelihoods are reported in input units. KS is a descriptive
statistic only: fitting on the test sample invalidates ordinary KS p-values.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from .skew_t import CUSTOM_DISTS
from .variance_gamma import VG_MODELS, variance_gamma
from .model_names import MODEL_ALIASES, canonical_model, unique_models
from .gh_skew_t import gh_skew_t, fit_gh_skew_t
from .champernowne import champernowne, fit_champernowne
from .meixner import MEIXNER_MODELS, meixner, fit_meixner
from .egb2 import EGB2_MODELS, egb2, fit_egb2
from .nts import NTS_MODELS, nts, fit_nts
from .skew_ged import skew_ged, fit_skew_ged
from .generalized_t import GT_MODELS, generalized_t, fit_generalized_t
from .scipy_extra import EXTRA_MODELS, crystal_ball, fit_extra

DEFAULT_MODELS = ('normal', 'student-t', 'laplace', 'ged',
                  'hyperbolic-symmetric', 'hyperbolic-skewed',
                  'nig-symmetric', 'nig-skewed')
ALIASES = {'normal': 'norm', 'student-t': 't', 'ged': 'gennorm',
           'hyperbolic-secant': 'hypsecant',
           'hyperbolic-symmetric': 'genhyperbolic', 'hyperbolic-skewed': 'genhyperbolic',
           'nig-symmetric': 'norminvgauss', 'nig-skewed': 'norminvgauss',
           'gh-symmetric': 'genhyperbolic', 'gh-skewed': 'genhyperbolic'}


def model_catalog():
    """Names accepted by the univariate fitter, including installed SciPy families."""
    project = set(DEFAULT_MODELS) | set(ALIASES) | set(MODEL_ALIASES) | set(CUSTOM_DISTS)
    project.update((*EXTRA_MODELS, *GT_MODELS, *NTS_MODELS, *EGB2_MODELS, *MEIXNER_MODELS, *VG_MODELS,
                    'ged-skewed', 'fs-skew-normal', 'champernowne', 'gh-skew-t'))
    project.update(MODEL_ALIASES.values())
    scipy_names = {name for name in dir(stats) if not name.startswith('_')
                   and isinstance(getattr(stats, name), stats.rv_continuous)}
    return {'Project models and aliases': sorted(project),
            'Additional SciPy continuous models': sorted(scipy_names-project)}


def specification(name):
    name = canonical_model(name)
    if name in {'johnsonsu', 'johnson-su-symmetric'}:
        return stats.johnsonsu, ['a', 'b', 'loc', 'scale'], {'a': 0.} if name == 'johnson-su-symmetric' else {}
    if name == 'crystalball':
        return crystal_ball, ['beta', 'm', 'loc', 'scale'], {}
    if name in GT_MODELS:
        return generalized_t, ['power', 'q', 'skewness', 'loc', 'scale'], {'skewness': 0.} if name == 'generalized-t' else {}
    if name in {'ged-skewed', 'fs-skew-normal'}:
        return skew_ged, ['power', 'skewness', 'loc', 'scale'], {'power': 2.} if name == 'fs-skew-normal' else {}
    if name in NTS_MODELS:
        return nts, ['alpha','lam','b','loc','scale'], {'b':0.} if name.endswith('-symmetric') else {}
    if name in EGB2_MODELS:
        # The specialized fitter ties b=a; this counts one shape constraint.
        return egb2, ['a','b','loc','scale'], {'b':'a'} if name.endswith('-symmetric') else {}
    if name in MEIXNER_MODELS:
        return meixner, ['delta','b','loc','scale'], {'b':0.} if name.endswith('-symmetric') else {}
    if name == 'champernowne':
        return champernowne, ['lam','loc','scale'], {}
    if name == 'gh-skew-t':
        return gh_skew_t, ['df','b','loc','scale'], {}
    dist = variance_gamma if name in VG_MODELS else CUSTOM_DISTS.get(name) or getattr(stats, ALIASES.get(name, name), None)
    if not isinstance(dist, stats.rv_continuous):
        raise ValueError(f'Unknown continuous distribution: {name}')
    names = ([s.strip() for s in dist.shapes.split(',')] if dist.shapes else []) + ['loc', 'scale']
    fixed = {}
    if name in {'hyperbolic-symmetric','hyperbolic-skewed'}:
        fixed['p'] = 1.0
    if name.endswith('-symmetric'):
        fixed['b'] = 0.0
    return dist, names, fixed


def fit_one(x, name, param_names=None, *, location=None, max_iterations=10000):
    """Fit one model; optional location is fixed in original input units.

    Nonfinite observations are rejected, not silently removed. ``param_names``
    is accepted for compatibility with the original script's catalog.
    """
    started = time.perf_counter()
    name = canonical_model(name)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError('Need at least eight finite observations in a one-dimensional array')
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if location is not None and not np.isfinite(location):
        raise ValueError('Fixed location must be finite')
    center, unit = float(np.mean(x)), float(np.std(x))
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Cannot fit a constant or numerically invalid sample')
    if name == 'gh-skew-t':
        fit=fit_gh_skew_t(x[:,None],location,max_iterations)
        scale=float(np.sqrt(fit['scatter'][0][0])); b=fit['gamma'][0]/scale
        frozen=gh_skew_t(fit['df'],b,loc=fit['location'][0],scale=scale)
        mean,variance,skew,kurt=frozen.stats(moments='mvsk')
        return dict(name=name,scipy_distribution='gh_skew_t',n=len(x),k=fit['parameters'],loglik=fit['loglik'],aic=fit['aic'],bic=fit['bic'],df=fit['df'],b=b,loc=fit['location'][0],scale=scale,mean=float(mean),variance=float(variance),skew=float(skew),kurt=float(kurt),ks=float(stats.kstest(x,frozen.cdf).statistic),ks_p=np.nan,converged=fit['converged'],status=fit['status'],optimizer_message=fit['message'],warnings='',fit_sec=time.perf_counter()-started)
    z = (x-center)/unit
    dist, names, fixed = specification(name)
    kwargs = {f'f{key}': value for key, value in fixed.items()}
    if location is not None:
        kwargs['floc'] = (location-center)/unit
    runs = []

    def optimizer(func, initial, args=(), disp=0):
        result = optimize.minimize(func, initial, args=args, method='Nelder-Mead',
                                   options={'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-8})
        runs.append(result)
        return result.x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name in EXTRA_MODELS:
            params, result = fit_extra(z, name, kwargs.get('floc'), max_iterations)
            params = list(params)
            runs.append(result)
        elif name in GT_MODELS:
            params, result = fit_generalized_t(z, name == 'generalized-t', kwargs.get('floc'), max_iterations)
            params = list(params)
            runs.append(result)
        elif name in {'ged-skewed', 'fs-skew-normal'}:
            params, result = fit_skew_ged(z, kwargs.get('floc'), max_iterations,
                                        fixed_power=2. if name == 'fs-skew-normal' else None)
            params = list(params)
            runs.append(result)
        elif name == 'champernowne':
            params, result = fit_champernowne(z,kwargs.get('floc'),max_iterations)
            params=list(params)
            runs.append(result)
        elif name in NTS_MODELS:
            params,result=fit_nts(z,name.endswith('-symmetric'),kwargs.get('floc'),max_iterations)
            params=list(params)
            runs.append(result)
        elif name in EGB2_MODELS:
            params,result=fit_egb2(z,name.endswith('-symmetric'),kwargs.get('floc'),max_iterations)
            params=list(params)
            runs.append(result)
        elif name in MEIXNER_MODELS:
            params,result=fit_meixner(z,name.endswith('-symmetric'),kwargs.get('floc'),max_iterations)
            params=list(params)
            runs.append(result)
        elif dist.name == 'logistic':
            # Its specialized SciPy fit ignores our optimizer callback, hiding
            # convergence status and iteration limits. Use the generic MLE path.
            params = list(stats.rv_continuous.fit(dist,z,optimizer=optimizer,**kwargs))
        else:
            params = list(dist.fit(z, optimizer=optimizer, **kwargs))
        params[-2] = center+unit*params[-2]
        params[-1] *= unit
        ll = float(np.sum(dist.logpdf(x, *params)))
        if not np.isfinite(ll) or params[-1] <= 0:
            raise ValueError('Fit produced an invalid density or likelihood')
        # Do not replace undefined moments with truncated quantile approximations.
        mean, variance, skew, kurt = dist.stats(*params, moments='mvsk')
        ks = stats.kstest(x, dist.cdf, args=tuple(params)).statistic
    k = len(names)-len(fixed)-(location is not None)
    converged = all(r.success for r in runs) if runs else None
    if not runs and dist.name in {'norm', 'laplace'}:
        converged = True
    row = dict(name=name, scipy_distribution=dist.name, n=len(x), k=int(k),
               loglik=ll, aic=2*k-2*ll, bic=np.log(len(x))*k-2*ll,
               ks=float(ks), ks_p=np.nan, mean=float(mean), variance=float(variance),
               skew=float(skew), kurt=float(kurt), converged=converged,
               status='ok' if converged else ('not_converged' if runs else 'optimizer_status_unavailable'),
               optimizer_message='; '.join(str(r.message) for r in runs),
               warnings='; '.join(dict.fromkeys(str(w.message) for w in caught)),
               fit_sec=time.perf_counter()-started)
    # Shape "k" in some legacy families must not overwrite parameter count.
    row.update({('shape_k' if key == 'k' else key): float(value) for key, value in zip(names, params)})
    if name in EXTRA_MODELS:
        row['fit_restriction'] = ('beta in [0.1,10]; m in [1.05,100]' if name == 'crystalball'
            else ('a fixed at 0' if name == 'johnson-su-symmetric' else 'a in [-10,10]')+'; b in [0.25,20]') + '; standardized log(scale) in [-12,12]'
        if name == 'crystalball': row['tail_index'] = row['m']-1
        if result.boundary: row['status'] = 'boundary' if converged else 'not_converged'
    if name in GT_MODELS:
        row['tail_index'] = row['power']*row['q']
        row['fit_restriction'] = 'power in [0.2,10]; q in [0.1,1000]; skewness in [-3,3]; standardized log(scale) in [-12,12]'
        if result.boundary: row['status'] = 'boundary' if converged else 'not_converged'
    if name in {'ged-skewed', 'fs-skew-normal'}:
        row['fit_restriction'] = ('power fixed at 2' if name == 'fs-skew-normal' else 'power in [0.1,10]') + '; skewness in [-3,3]; standardized log(scale) in [-12,12]'
        if result.boundary: row['status'] = 'boundary' if converged else 'not_converged'
    if name == 'champernowne':
        row['fit_restriction']='log(1+lam), standardized log(scale) in [-12,12]'
        if result.boundary: row['status']='boundary' if converged else 'not_converged'
    if name in MEIXNER_MODELS:
        row['fit_restriction']='delta in [0.01,200]; abs(b)<=pi-0.001; standardized log(scale) in [-12,12]'
        if result.boundary: row['status']='boundary' if converged else 'not_converged'
    if name in EGB2_MODELS:
        row['fit_restriction']='a,b in [0.01,200]; standardized log(scale) in [-12,12]; b=a if symmetric'
        if result.boundary: row['status']='boundary' if converged else 'not_converged'
    if name in NTS_MODELS:
        row['fit_restriction']='alpha in [0.1,0.95]; lam in [0.1,50]; b in [-5,5]; standardized log(scale) in [-6,6]'
        row['fit_method']='multi-start MLE; shifted Fourier quadrature with analytic scores'
        row['projected_score_per_observation']=result.projected_score
        if result.boundary: row['status']='boundary' if converged else 'not_converged'
    if name in VG_MODELS:
        row['fit_restriction'] = 'bounded density: shape > 0.55; shape <= 100'
        if row['vg_shape'] < .5501 or row['vg_shape'] > 99.99:
            row['status'] = 'boundary'
    return row


def fit_many(x, models=DEFAULT_MODELS, *, fit_timeout=None, **kwargs):
    """Return a ranked table; unsuccessful fits remain visible but unranked."""
    rows = []
    from .fit_timeout import run_fit, FitTimeout, positive_seconds
    if fit_timeout is not None: positive_seconds(fit_timeout)
    for name in unique_models(models):
        started = time.perf_counter()
        try:
            rows.append(run_fit(fit_one, x, name, fit_timeout=fit_timeout, **kwargs))
        except Exception as exc:
            rows.append(dict(name=name, status='timeout' if isinstance(exc, FitTimeout) else 'failed', converged=False, error=str(exc),
                             aic=np.nan, bic=np.nan, fit_sec=time.perf_counter()-started))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    from .fit_ranks import criterion_ranks
    table = criterion_ranks(table)
    return table.sort_values(['aic_rank', 'aic'], na_position='last').reset_index(drop=True)


def fitted_distribution(row):
    """Reconstruct a frozen SciPy distribution from a fit table/CSV row."""
    if isinstance(row.get('mixture_fit'), str):
        import json
        from .univariate_mixture import UnivariateMixture
        return UnivariateMixture(json.loads(row['mixture_fit']))
    dist, names, _ = specification(row['name'])
    return dist(*[row['shape_k' if key == 'k' else key] for key in names])
