"""Joint normal, Student-t and GH maximum likelihood for small asset panels.

Student-t has a single shared degrees-of-freedom parameter and a full positive
definite scatter matrix. Scatter is NOT covariance: covariance=df/(df-2)*scatter
when df>2. No pairwise deletion or implicit covariance regularization is used.
"""
import time

import numpy as np
from scipy import linalg, optimize, special, stats
from .joint_gh import GH_MODELS, fit_gh, JointGH
from .joint_power import POWER_MODELS, SKEW_POWER_MODELS, fit_power, JointPower
from .joint_skew_t import fit_skew_t, JointSkewT
from .joint_nct import JointNCT
from .gh_skew_t import JointGHSkewT, fit_gh_skew_t
from .model_names import canonical_model
from .joint_sdb import SDB_MODELS, JointSDB, fit_sdb
from .joint_slash import SLASH_MODELS, JointSlash, fit_slash
from .joint_laplace_mixture import LAPLACE_MIXTURE_MODELS, JointLaplaceMixture, fit_laplace_mixture
from .joint_generalized_t import JointGeneralizedT, fit_generalized_t
from .joint_nts import JointNTS, fit_nts_joint, NTS_MODELS

JOINT_MODELS = ('normal', 'student-t', *POWER_MODELS, *GH_MODELS, 'azzalini-skew-t', 'noncentral-t')
from .variance_gamma import VG_MODELS, JointVG
ALL_JOINT_MODELS = (*JOINT_MODELS, *SDB_MODELS, *SLASH_MODELS, *VG_MODELS, 'gh-skew-t', *LAPLACE_MIXTURE_MODELS, 'generalized-t', 'generalized-t-skewed', *NTS_MODELS, *SKEW_POWER_MODELS)


def fit_joint(data, model='student-t', *, location=None, max_iterations=2000,
              df_starts=(3., 8., 30.), sdb_points=512, sdb_seed=12345, slash_points=64,
              laplace_location=None, laplace_mode_bandwidth=None, nts_points=256):
    """Return a JSON-serializable joint fit in original return units.

    Data must be a finite (observations, assets) array. Fixed location may be
    scalar or a vector. Estimated df is bounded to [0.1, 100000]; bound solutions
    are reported explicitly. Multiple starts mitigate, not eliminate, local optima.
    """
    started = time.perf_counter()
    model = canonical_model(model)
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Joint input must be a finite two-dimensional array')
    n, d = x.shape
    if d < 1 or n < max(8, d+2):
        raise ValueError('Need at least max(8, assets+2) complete observations')
    if model not in ALL_JOINT_MODELS:
        raise ValueError('Unknown joint model: ' + model)
    if model not in LAPLACE_MIXTURE_MODELS and (laplace_location is not None or laplace_mode_bandwidth is not None):
        raise ValueError('Laplace location options require a Laplace-mixture model')
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    center, unit = x.mean(axis=0), x.std(axis=0)
    if not np.isfinite(unit).all() or (unit <= 0).any():
        raise ValueError('Constant or numerically invalid asset')
    z = (x-center)/unit
    if np.linalg.matrix_rank(z) < d:
        raise ValueError('Rank-deficient returns: remove duplicate or linearly dependent assets')
    fixed = None
    if location is not None:
        fixed = np.broadcast_to(np.asarray(location, dtype=float), (d,)).copy()
        if not np.isfinite(fixed).all():
            raise ValueError('Fixed location must be finite')
        fixed = (fixed-center)/unit
    mu0 = np.zeros(d) if fixed is None else fixed
    if model in NTS_MODELS:
        if not isinstance(nts_points,int) or not 64 <= nts_points <= 1024:
            raise ValueError('NTS quadrature points must be an integer from 64 to 1024')
        return fit_nts_joint(x,model,location,max_iterations,nts_points)
    if model in {'generalized-t', 'generalized-t-skewed'}:
        return fit_generalized_t(x, location, max_iterations, skewed=model.endswith('-skewed'))
    if model in LAPLACE_MIXTURE_MODELS:
        return fit_laplace_mixture(x,model,location,max_iterations,laplace_location,laplace_mode_bandwidth)
    if model == 'gh-skew-t':
        return fit_gh_skew_t(x,location,max_iterations)
    if model in SLASH_MODELS:
        return fit_slash(x,model,location,max_iterations,slash_points)
    if model in SDB_MODELS:
        return fit_sdb(x, model, location, max_iterations, sdb_points, sdb_seed)
    if model in {'azzalini-skew-t', 'noncentral-t'}:
        return fit_skew_t(x, location, max_iterations, model=model)
    if model in (*GH_MODELS, *VG_MODELS):
        return fit_gh(x, model, location, max_iterations)
    if model in (*POWER_MODELS, *SKEW_POWER_MODELS):
        return fit_power(x, model, location, max_iterations)
    scatter0 = (z-mu0).T@(z-mu0)/n
    attempts = []
    boundary = False
    if model == 'normal':
        mu, scatter, df = mu0, scatter0, None
        loglik_z = float(stats.multivariate_normal.logpdf(z, mean=mu, cov=scatter).sum())
        converged, message = True, 'Analytic normal MLE'
    else:
        indices = np.tril_indices(d)
        diagonal = np.flatnonzero(indices[0] == indices[1])
        nloc = d if fixed is None else 0

        def pack(mu, scatter, df):
            chol = np.linalg.cholesky(scatter)
            values = chol[indices].copy()
            values[diagonal] = np.log(values[diagonal])
            return np.r_[mu if fixed is None else [], values, np.log(df)]

        def unpack(theta):
            mu = theta[:d] if fixed is None else fixed
            values = theta[nloc:-1].copy()
            values[diagonal] = np.exp(values[diagonal])
            chol = np.zeros((d, d))
            chol[indices] = values
            return mu, chol, np.exp(theta[-1])

        def objective(theta):
            mu, chol, df = unpack(theta)
            residual = linalg.solve_triangular(chol, (z-mu).T, lower=True, check_finite=False)
            q = np.sum(residual**2, axis=0)
            constant = special.gammaln((df+d)/2)-special.gammaln(df/2)-d/2*np.log(df*np.pi)-np.log(chol.diagonal()).sum()
            value = -n*constant+(df+d)/2*np.log1p(q/df).sum()
            return float(value) if np.isfinite(value) else 1e100

        bounds = [(None, None)]*(nloc+len(indices[0])) + [(np.log(.1), np.log(100000.))]
        for j in diagonal:
            bounds[nloc+j] = (-12, 12)
        candidates = []
        for initial_df in df_starts:
            if not .1 < initial_df < 100000:
                raise ValueError('Initial df must be between 0.1 and 100000')
            start_scatter = scatter0*max(.2, (initial_df-2)/initial_df)
            result = optimize.minimize(objective, pack(mu0, start_scatter, initial_df),
                                       method='L-BFGS-B', bounds=bounds,
                                       options={'maxiter': max_iterations, 'ftol': 1e-11,
                                                'gtol': 1e-6, 'maxfun': 200000})
            attempts.append(dict(initial_df=float(initial_df), converged=bool(result.success),
                                 negative_loglik=float(result.fun), iterations=int(result.nit),
                                 message=str(result.message)))
            if np.isfinite(result.fun):
                candidates.append(result)
        if not candidates:
            raise ValueError('No finite joint Student-t fit')
        successful = [r for r in candidates if r.success]
        result = min(successful or candidates, key=lambda r: r.fun)
        mu, chol, df = unpack(result.x)
        scatter = chol@chol.T
        loglik_z = -float(result.fun)
        converged, message = bool(result.success), str(result.message)
        boundary = any((lo is not None and abs(v-lo) < 1e-4) or
                       (hi is not None and abs(v-hi) < 1e-4)
                       for v, (lo, hi) in zip(result.x, bounds))
    mu = center+unit*mu
    scatter = scatter*np.outer(unit, unit)
    loglik = loglik_z-n*np.log(unit).sum()
    k = d*(d+1)//2+(d if fixed is None else 0)+(model == 'student-t')
    covariance = scatter if model == 'normal' else (scatter*df/(df-2) if df > 2 else None)
    mean = mu if model == 'normal' or df > 1 else None
    correlation = scatter/np.sqrt(np.outer(scatter.diagonal(), scatter.diagonal()))
    return dict(model=model, observations=n, dimensions=d, parameters=k,
                loglik=float(loglik), aic=float(2*k-2*loglik), bic=float(np.log(n)*k-2*loglik),
                location=mu.tolist(), scatter=scatter.tolist(),
                covariance=None if covariance is None else covariance.tolist(),
                mean=None if mean is None else mean.tolist(),
                scatter_correlation=correlation.tolist(), df=None if df is None else float(df),
                converged=converged, boundary=boundary,
                status='boundary' if converged and boundary else ('ok' if converged else 'not_converged'),
                message=message, attempts=attempts, fit_sec=time.perf_counter()-started)


def joint_distribution(fit):
    """Reconstruct a SciPy frozen joint distribution from a saved JSON fit."""
    if fit.get('components', 1) > 1:
        from .joint_finite_mixture import JointFiniteMixture
        return JointFiniteMixture(fit)
    if fit['model'] in NTS_MODELS:
        return JointNTS(fit)
    if fit['model'] in {'generalized-t', 'generalized-t-skewed'}:
        return JointGeneralizedT(fit)
    if fit['model'] in LAPLACE_MIXTURE_MODELS:
        return JointLaplaceMixture(fit)
    if fit['model'] == 'gh-skew-t':
        return JointGHSkewT(fit)
    if fit['model'] in VG_MODELS:
        return JointVG(fit)
    if fit['model'] in SLASH_MODELS:
        return JointSlash(fit)
    if fit['model'] in SDB_MODELS:
        return JointSDB(fit)
    if fit['model'] == 'noncentral-t':
        return JointNCT(fit)
    if fit['model'] == 'azzalini-skew-t':
        return JointSkewT(fit)
    if fit['model'] == 'normal':
        return stats.multivariate_normal(mean=fit['location'], cov=fit['scatter'])
    if fit['model'] == 'student-t':
        return stats.multivariate_t(loc=fit['location'], shape=fit['scatter'], df=fit['df'])
    if fit['model'] in GH_MODELS:
        return JointGH(fit)
    if fit['model'] in (*POWER_MODELS, *SKEW_POWER_MODELS):
        return JointPower(fit)
    raise ValueError('Unknown joint model')
