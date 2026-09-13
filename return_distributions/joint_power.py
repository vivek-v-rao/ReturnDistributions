"""Elliptical power exponential: density proportional to exp(-q**(p/2)).

q=(x-mu)' scatter^-1 (x-mu). p=1 is radial/elliptical Laplace;
p=2 is normal with covariance scatter/2. Marginals in d>1 generally are
not univariate GED/Laplace. This is not the exponential normal-mixture Laplace.
"""
import time
import numpy as np
from scipy import linalg, optimize, special

POWER_MODELS = ('laplace', 'ged')


def covariance_factor(d, power):
    return float(np.exp(special.gammaln((d+2)/power)-special.gammaln(d/power))/d)


def power_logpdf(x, location, chol, power):
    residual = linalg.solve_triangular(chol, (np.atleast_2d(x)-location).T, lower=True, check_finite=False)
    d = len(location)
    constant = (np.log(power)+special.gammaln(d/2)-np.log(2)
                -d/2*np.log(np.pi)-special.gammaln(d/power)-np.log(chol.diagonal()).sum())
    return constant-np.sum(residual**2, axis=0)**(power/2)


class JointPower:
    """Frozen elliptical GED/Laplace with density, sampling and moments."""
    def __init__(self, fit):
        self.location = np.asarray(fit['location'])
        self.scatter = np.asarray(fit['scatter'])
        self.chol = np.linalg.cholesky(self.scatter)
        self.power = float(fit['power'])

    def logpdf(self, x):
        value = power_logpdf(x, self.location, self.chol, self.power)
        return value[0] if np.asarray(x).ndim == 1 else value

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        return self.location.copy()

    def cov(self):
        return covariance_factor(len(self.location), self.power)*self.scatter

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        d = len(self.location)
        direction = rng.normal(size=(size, d))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        radius = rng.gamma(d/self.power, size=size)**(1/self.power)
        return self.location+(radius[:, None]*direction)@self.chol.T


def fit_power(x, model, location=None, max_iterations=2000):
    """Common input validation is performed by fit_joint."""
    start = time.perf_counter()
    n, d = x.shape
    center, unit = x.mean(axis=0), x.std(axis=0)
    z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    nloc = d if fixed is None else 0
    estimate_power = model == 'ged'
    indices = np.tril_indices(d)
    diagonal = np.flatnonzero(indices[0] == indices[1])
    nc = len(indices[0])

    def pack(mu, scatter, power):
        values = np.linalg.cholesky(scatter)[indices]
        values[diagonal] = np.log(values[diagonal])
        return np.r_[mu if fixed is None else [], values, [np.log(power)] if estimate_power else []]

    def unpack(theta):
        mu = theta[:d] if fixed is None else fixed
        values = theta[nloc:nloc+nc].copy()
        values[diagonal] = np.exp(values[diagonal])
        chol = np.zeros((d, d))
        chol[indices] = values
        power = np.exp(theta[-1]) if estimate_power else 1.
        return mu, chol, power

    def objective(theta):
        with np.errstate(all='ignore'):
            value = -float(power_logpdf(z, *unpack(theta)).sum())
        return value if np.isfinite(value) else 1e100

    bounds = [(None, None)]*(nloc+nc)
    for j in diagonal:
        bounds[nloc+j] = (-12, 12)
    if estimate_power:
        bounds.append((np.log(.25), np.log(10.)))
    mu0 = np.zeros(d) if fixed is None else fixed
    cov0 = (z-mu0).T@(z-mu0)/n
    starts = [(f'power={p}', pack(mu0, cov0/covariance_factor(d, p), p))
              for p in ((.75, 1., 2., 3.) if estimate_power else (1.,))]
    if estimate_power:
        nested = fit_power(x, 'laplace', location, max_iterations)
        starts.append(('laplace-fit', pack((np.array(nested['location'])-center)/unit,
                      np.array(nested['scatter'])/np.outer(unit, unit), 1.)))
    elif fixed is None:
        starts.append(('median-location', pack(np.median(z, axis=0), cov0/covariance_factor(d, 1), 1)))
    attempts, candidates = [], []
    for label, initial in starts:
        result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': max_iterations, 'maxfun': 200000, 'ftol': 1e-11, 'gtol': 1e-6})
        attempts.append(dict(start=label, converged=bool(result.success), iterations=int(result.nit),
                             negative_loglik=float(result.fun), message=str(result.message)))
        if np.isfinite(result.fun) and result.fun < 1e99:
            candidates.append(result)
    if not candidates:
        raise ValueError('No finite power-exponential fit')
    successful = [r for r in candidates if r.success]
    result = min(successful or candidates, key=lambda r: r.fun)
    mu, chol, power = unpack(result.x)
    mu = center+unit*mu
    scatter = (chol@chol.T)*np.outer(unit, unit)
    covariance = covariance_factor(d, power)*scatter
    if not np.isfinite(covariance).all():
        raise ValueError('Nonfinite fitted covariance')
    correlation = scatter/np.sqrt(np.outer(scatter.diagonal(), scatter.diagonal()))
    ll = -float(result.fun)-n*np.log(unit).sum()
    k = nloc+nc+int(estimate_power)
    boundary = any((lo is not None and abs(v-lo) < 1e-4) or (hi is not None and abs(v-hi) < 1e-4)
                   for v, (lo, hi) in zip(result.x, bounds))
    return dict(model=model, family='elliptical-power-exponential', power=float(power),
                observations=n, dimensions=d, parameters=k, loglik=float(ll),
                aic=float(2*k-2*ll), bic=float(np.log(n)*k-2*ll),
                location=mu.tolist(), mean=mu.tolist(), scatter=scatter.tolist(),
                covariance=covariance.tolist(), correlation=correlation.tolist(),
                scatter_correlation=correlation.tolist(), converged=bool(result.success),
                boundary=boundary, status='boundary' if result.success and boundary else ('ok' if result.success else 'not_converged'),
                message=str(result.message), attempts=attempts, fit_sec=time.perf_counter()-start)
