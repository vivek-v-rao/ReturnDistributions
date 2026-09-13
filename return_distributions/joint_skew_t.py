"""Azzalini--Capitanio skew-t: 2 t_d(x) T_{nu+d}(alpha'z sqrt((nu+d)/(nu+Q))).

z uses marginal scatter scales, not standard deviations. See Azzalini and
Capitanio (2003), JRSS B, doi:10.1111/1467-9868.00391.
"""
import time
import numpy as np
from scipy import linalg, optimize, special, stats


def logdensity(x, mu, chol, a, df):
    y = linalg.solve_triangular(chol, (np.atleast_2d(x)-mu).T, lower=True)
    d = len(mu)
    q = np.sum(y*y, axis=0)
    base = (special.gammaln((df+d)/2)-special.gammaln(df/2)
            -d/2*np.log(df*np.pi)-np.log(chol.diagonal()).sum()
            -(df+d)/2*np.log1p(q/df))
    return np.log(2)+base+stats.t.logcdf(a@y*np.sqrt((df+d)/(df+q)), df+d)


class JointSkewT:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'], dtype=float)
        self.scatter = np.asarray(fit['scatter'], dtype=float)
        self.alpha = np.asarray(fit['alpha'], dtype=float)
        self.df = float(fit['df'])
        d = len(self.location)
        if (self.scatter.shape != (d, d) or self.alpha.shape != (d,)
                or not np.isfinite(self.location).all() or not np.isfinite(self.scatter).all()
                or not np.isfinite(self.alpha).all() or not np.isfinite(self.df) or self.df <= 0
                or not np.allclose(self.scatter, self.scatter.T)):
            raise ValueError('Invalid multivariate skew-t parameters')
        self.chol = np.linalg.cholesky(self.scatter)
        self.a = self.chol.T@(self.alpha/np.sqrt(self.scatter.diagonal()))
        self.delta = self.chol@self.a/np.sqrt(1+self.a@self.a)

    def logpdf(self, x):
        value = logdensity(x, self.location, self.chol, self.a, self.df)
        return value[0] if np.asarray(x).ndim == 1 else value

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        if self.df <= 1: return np.full_like(self.location, np.nan)
        b = np.sqrt(self.df/np.pi)/special.poch((self.df-1)/2, .5)
        return self.location+b*self.delta

    def cov(self):
        if self.df <= 2: return np.full_like(self.scatter, np.nan)
        shift = self.mean()-self.location
        return self.df/(self.df-2)*self.scatter-np.outer(shift, shift)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        residual = self.scatter-np.outer(self.delta, self.delta)
        z = rng.normal(size=(size, len(self.location)))@np.linalg.cholesky(residual).T
        z += np.abs(rng.normal(size=size))[:, None]*self.delta
        return self.location+z/np.sqrt(rng.chisquare(self.df, size=size)/self.df)[:, None]


def fit_skew_t(x, location=None, max_iterations=2000, *, model='azzalini-skew-t'):
    """Full MLE with positive-definite scatter and multiple skewness starts."""
    from .multivariate import fit_joint
    density, frozen_class = logdensity, JointSkewT
    if model == 'noncentral-t':
        from .joint_nct import logdensity as density, JointNCT as frozen_class
    elif model != 'azzalini-skew-t':
        raise ValueError('Unknown skew-t model')
    started = time.perf_counter()
    n, d = x.shape
    center, unit = x.mean(axis=0), x.std(axis=0)
    z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    nloc = d if fixed is None else 0
    ix = np.tril_indices(d)
    diagonal = np.flatnonzero(ix[0] == ix[1])
    nc = len(ix[0])
    nested = fit_joint(z, 'student-t', location=fixed, max_iterations=max_iterations)
    values = np.linalg.cholesky(nested['scatter'])[ix]
    values[diagonal] = np.log(values[diagonal])
    initial = np.r_[nested['location'] if fixed is None else [], values, np.zeros(d), np.log(nested['df'])]
    bounds = [(None, None)]*(nloc+nc)+[(-30., 30.)]*d+[(np.log(.1), np.log(100000.))]
    for j in diagonal: bounds[nloc+j] = (-12., 12.)

    def unpack(theta):
        mu = theta[:d] if fixed is None else fixed
        v = theta[nloc:nloc+nc].copy()
        v[diagonal] = np.exp(v[diagonal])
        chol = np.zeros((d, d)); chol[ix] = v
        return mu, chol, theta[nloc+nc:-1], np.exp(theta[-1])

    def objective(theta):
        with np.errstate(all='ignore'):
            value = -density(z, *unpack(theta)).sum()
        return float(value) if np.isfinite(value) else 1e100

    attempts, candidates = [], []
    direction = np.where(stats.skew(z, axis=0) >= 0, 1., -1.)
    for skew in (np.zeros(d), direction, -direction):
        seed = initial.copy(); seed[nloc+nc:-1] = skew
        result = optimize.minimize(objective, seed, method='L-BFGS-B', bounds=bounds,
                                   options=dict(maxiter=max_iterations, maxfun=200000, ftol=1e-11, gtol=1e-6))
        attempts.append(dict(start=skew.tolist(), converged=bool(result.success),
                             message=str(result.message), negative_loglik=float(result.fun)))
        if np.isfinite(result.fun) and result.fun < 1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite multivariate skew-t fit')
    result = min([r for r in candidates if r.success] or candidates, key=lambda r: r.fun)
    mu, chol, a, df = unpack(result.x)
    scatter = chol@chol.T
    alpha = np.sqrt(scatter.diagonal())*linalg.solve_triangular(chol.T, a, lower=False)
    ll = -float(result.fun)-n*np.log(unit).sum()
    k = nloc+nc+d+1
    boundary = any((lo is not None and abs(v-lo) < 1e-4) or (hi is not None and abs(v-hi) < 1e-4)
                   for v, (lo, hi) in zip(result.x, bounds))
    fit = dict(model=model, location=(center+unit*mu).tolist(),
               scatter=(scatter*np.outer(unit, unit)).tolist(), alpha=alpha.tolist(), df=float(df),
               observations=n, dimensions=d, parameters=k, loglik=ll,
               aic=2*k-2*ll, bic=np.log(n)*k-2*ll, converged=bool(result.success), boundary=boundary,
               status=('boundary' if boundary else 'ok') if result.success else 'not_converged',
               message=str(result.message), attempts=attempts, fit_sec=time.perf_counter()-started)
    if model == 'noncentral-t':
        del fit['alpha']
        fit['delta'] = (unit*(chol@a)).tolist()
    frozen = frozen_class(fit)
    covariance = frozen.cov() if df > 2 else None
    fit.update(mean=frozen.mean().tolist() if df > 1 else None,
               covariance=None if covariance is None else covariance.tolist(),
               scatter_correlation=(scatter/np.sqrt(np.outer(scatter.diagonal(), scatter.diagonal()))).tolist(),
               correlation=None if covariance is None else (covariance/np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))).tolist())
    return fit
