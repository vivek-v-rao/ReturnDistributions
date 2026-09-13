"""Identified GH mixtures: X=mu+W*gamma+sqrt(W)*L*Z, W~GIG(lambda,1,psi).

Fixing chi=1 removes the mixture/scatter scale indeterminacy. See QRM GHYP
documentation for lambda=(d+1)/2 hyperbolic and lambda=-1/2 NIG conventions.
"""
import time
import numpy as np
from scipy import linalg, optimize, special, stats

GH_MODELS = ('hyperbolic-symmetric', 'hyperbolic-skewed', 'nig-symmetric', 'nig-skewed',
             'gh-symmetric', 'gh-skewed')


def log_bessel(order, x):
    return np.log(special.kve(order, x))-x


def mixture_moments(lam, psi):
    root = np.sqrt(psi)
    denominator = special.kve(lam, root)
    mean = special.kve(lam+1, root)/denominator/root
    second = special.kve(lam+2, root)/denominator/psi
    variance = max(0., float(second-mean*mean))
    return float(mean), variance


def gh_logpdf(x, mu, chol, gamma, lam, psi):
    x = np.atleast_2d(x)
    d = len(mu)
    residual = linalg.solve_triangular(chol, (x-mu).T, lower=True, check_finite=False)
    skew = linalg.solve_triangular(chol, gamma, lower=True, check_finite=False)
    a = 1+np.sum(residual**2, axis=0)
    b = psi+skew@skew
    order = lam-d/2
    return (lam/2*np.log(psi)-log_bessel(lam, np.sqrt(psi))
            -d/2*np.log(2*np.pi)-np.log(np.diag(chol)).sum()
            +skew@residual+order/2*(np.log(a)-np.log(b))
            +log_bessel(order, np.sqrt(a*b)))


class JointGH:
    """Frozen fitted GH distribution; logpdf/pdf/rvs and population moments."""
    def __init__(self, fit):
        self.location = np.asarray(fit['location'])
        self.scatter = np.asarray(fit['scatter'])
        self.chol = np.linalg.cholesky(self.scatter)
        self.gamma = np.asarray(fit['gamma'])
        self.lam, self.psi = fit['lambda'], fit['psi']

    def logpdf(self, x):
        values = gh_logpdf(x, self.location, self.chol, self.gamma, self.lam, self.psi)
        return values[0] if np.asarray(x).ndim == 1 else values

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        ew, _ = mixture_moments(self.lam, self.psi)
        return self.location+ew*self.gamma

    def cov(self):
        ew, vw = mixture_moments(self.lam, self.psi)
        return ew*self.scatter+vw*np.outer(self.gamma, self.gamma)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        w = stats.geninvgauss.rvs(self.lam, np.sqrt(self.psi), scale=1/np.sqrt(self.psi), size=size, random_state=rng)
        noise = rng.normal(size=(size, len(self.location)))@self.chol.T
        return self.location+w[:, None]*self.gamma+np.sqrt(w[:, None])*noise


def fit_gh(x, model, location=None, max_iterations=2000):
    """Validated input supplied by fit_joint; constrained multi-start MLE."""
    started = time.perf_counter()
    n, d = x.shape
    from .variance_gamma import VG_MODELS, vg_logpdf
    vg = model in VG_MODELS
    if vg and d/2+.05 >= 100:
        raise ValueError('Too many dimensions for the bounded VG shape range')
    center, unit = x.mean(axis=0), x.std(axis=0)
    z = (x-center)/unit
    lam = (d+1)/2 if model.startswith('hyperbolic') else -.5
    estimate_lambda = model.startswith('gh-')
    symmetric = model.endswith('symmetric')
    moments = (lambda shape, psi: (1., 1/psi)) if vg else mixture_moments
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    nloc = d if fixed is None else 0
    indices = np.tril_indices(d)
    diagonal = np.flatnonzero(indices[0] == indices[1])
    nc = len(indices[0])

    def pack(mu, scatter, gamma, psi, shape=lam):
        values = np.linalg.cholesky(scatter)[indices]
        values[diagonal] = np.log(values[diagonal])
        return np.r_[mu if fixed is None else [], values, [] if symmetric else gamma,
                     np.log(psi), [shape] if estimate_lambda else []]

    def unpack(theta):
        mu = theta[:d] if fixed is None else fixed
        values = theta[nloc:nloc+nc].copy()
        values[diagonal] = np.exp(values[diagonal])
        chol = np.zeros((d, d))
        chol[indices] = values
        gamma = np.zeros(d) if symmetric else theta[nloc+nc:nloc+nc+d]
        psi_index = nloc+nc+(0 if symmetric else d)
        return mu, chol, gamma, np.exp(theta[psi_index]), theta[-1] if estimate_lambda else lam

    def objective(theta):
        with np.errstate(all='ignore'):
            mu, chol, gamma, psi, shape = unpack(theta)
            values = vg_logpdf(z, mu, chol, gamma, psi) if vg else gh_logpdf(z, mu, chol, gamma, shape, psi)
            result = -float(values.sum())
        return result if np.isfinite(result) else 1e100

    bounds = [(None, None)]*(nloc+nc+(0 if symmetric else d))+[(-12, 12)]
    if vg:
        bounds[-1] = (np.log(d/2+.05), np.log(100.))
    if estimate_lambda:
        bounds.append((-20., 20.))
    for j in diagonal:
        bounds[nloc+j] = (-12, 12)
    mu0 = np.zeros(d) if fixed is None else fixed
    cov0 = (z-mu0).T@(z-mu0)/n
    starts = []
    for psi in ((min(99.99,d/2+.5), min(99.99,d/2+2.), max(d/2+.1,20.)) if vg else (.2, 2., 20.)):
        ew, _ = moments(lam, psi)
        starts.append((f'psi={psi}', pack(mu0, cov0/ew, np.zeros(d), psi)))
    # The restricted-family optima give feasible starts for the extra parameter.
    if estimate_lambda:
        for family in ('nig', 'hyperbolic'):
            nested = fit_gh(x, family+('-symmetric' if symmetric else '-skewed'), location, max_iterations)
            starts.append((family+'-fit', pack((np.array(nested['location'])-center)/unit,
                          np.array(nested['scatter'])/np.outer(unit, unit),
                          np.array(nested['gamma'])/unit, nested['psi'], nested['lambda'])))
    # Nested symmetric MLE supplies a feasible skewed start with equal likelihood.
    if not symmetric:
        nested = fit_gh(x, model.replace('skewed', 'symmetric'), location, max_iterations)
        starts.append(('symmetric-fit', pack((np.array(nested['location'])-center)/unit,
                      np.array(nested['scatter'])/np.outer(unit, unit), np.zeros(d), nested['vg_shape'] if vg else nested['psi'], nested['lambda'])))
    candidates, attempts = [], []
    for label, initial in starts:
        result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': max_iterations, 'maxfun': 200000, 'ftol': 1e-11, 'gtol': 1e-6})
        attempts.append(dict(start=label, converged=bool(result.success), iterations=int(result.nit),
                             negative_loglik=float(result.fun), message=str(result.message)))
        if np.isfinite(result.fun) and result.fun < 1e99:
            candidates.append(result)
    if not candidates:
        raise ValueError('No finite GH fit')
    successful = [r for r in candidates if r.success]
    result = min(successful or candidates, key=lambda r: r.fun)
    mu, chol, gamma, psi, fitted_lambda = unpack(result.x)
    mu, gamma = center+unit*mu, unit*gamma
    scatter = (chol@chol.T)*np.outer(unit, unit)
    ew, vw = moments(fitted_lambda, psi)
    covariance = ew*scatter+vw*np.outer(gamma, gamma)
    mean = mu+ew*gamma
    if not np.isfinite(covariance).all() or not np.isfinite(mean).all():
        raise ValueError('Nonfinite fitted GH moments')
    ll = -float(result.fun)-n*np.log(unit).sum()
    k = nloc+nc+(0 if symmetric else d)+1+int(estimate_lambda)
    boundary = any((lo is not None and abs(v-lo) < 1e-4) or (hi is not None and abs(v-hi) < 1e-4)
                   for v, (lo, hi) in zip(result.x, bounds))
    correlation = covariance/np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))
    scatter_corr = scatter/np.sqrt(np.outer(scatter.diagonal(), scatter.diagonal()))
    record = dict(model=model, observations=n, dimensions=d, parameters=k, loglik=float(ll),
                aic=float(2*k-2*ll), bic=float(np.log(n)*k-2*ll), location=mu.tolist(),
                scatter=scatter.tolist(), gamma=gamma.tolist(), chi=1., psi=float(psi),
                **{'lambda': float(fitted_lambda)}, mean=mean.tolist(), covariance=covariance.tolist(),
                correlation=correlation.tolist(), scatter_correlation=scatter_corr.tolist(),
                converged=bool(result.success), boundary=boundary,
                status='boundary' if result.success and boundary else ('ok' if result.success else 'not_converged'),
                message=str(result.message), attempts=attempts, fit_sec=time.perf_counter()-started)
    if vg:
        record.update(vg_shape=float(psi), shape_lower_bound=d/2+.05,
                      fit_restriction='bounded density: shape >= dimension/2 + 0.05', chi=0.)
        record['lambda'] = float(psi)
        record['psi'] = float(2*psi)
    return record
