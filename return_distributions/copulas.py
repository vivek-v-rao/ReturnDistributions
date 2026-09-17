"""Gaussian, Student-t and AC skew-t copula likelihoods and two-stage models."""
import time
from functools import lru_cache
import numpy as np
from scipy import optimize, stats
from .fitting import fitted_distribution
from .copula_quantiles import marginal_ppf


COPULA_MODELS = ['gaussian', 'student-t', 'azzalini-skew-t']


def copula_logpdf(u, model, correlation, df=None, alpha=None):
    """Copula density on strictly interior uniform scores, not return density."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or not np.isfinite(u).all() or (u <= 0).any() or (u >= 1).any():
        raise ValueError('Copula scores must be a finite matrix strictly inside (0,1)')
    if model == 'gaussian':
        z = stats.norm.ppf(u)
        return stats.multivariate_normal.logpdf(z, cov=correlation)-stats.norm.logpdf(z).sum(axis=1)
    if model == 'student-t':
        z = stats.t.ppf(u, df)
        return stats.multivariate_t.logpdf(z, shape=correlation, df=df)-stats.t.logpdf(z, df).sum(axis=1)
    if model == 'azzalini-skew-t':
        from .skew_t_copula import logpdf
        return logpdf(u, correlation, df, alpha)
    raise ValueError('Unknown copula model')


def fit_copula(u, model='gaussian', max_iterations=1000):
    start = time.perf_counter()
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] < 2 or len(u) < max(8, u.shape[1]+2):
        raise ValueError('Need >=2 assets and >=max(8,assets+2) complete observations')
    if not np.isfinite(u).all() or (u <= 0).any() or (u >= 1).any():
        raise ValueError('Uniform scores must lie strictly inside (0,1)')
    if model not in COPULA_MODELS or max_iterations < 1:
        raise ValueError('Invalid copula model or iteration limit')
    n, d = u.shape
    z = stats.norm.ppf(u)
    if np.linalg.matrix_rank(z-z.mean(axis=0)) < d:
        raise ValueError('Rank-deficient transformed returns')
    if model == 'azzalini-skew-t':
        from .skew_t_copula import fit
        return fit(u, max_iterations)
    initial = np.corrcoef(z.T)
    chol = np.linalg.cholesky(initial)
    indices = np.tril_indices(d, -1)
    v0 = (chol/chol.diagonal()[:, None])[indices]
    k = len(v0)+(model == 'student-t')

    @lru_cache(maxsize=16)
    def transforms(df):
        # Correlation-only finite-difference steps reuse these exact scores.
        # Cache is local to this fit, so samples cannot contaminate each other.
        if model == 'gaussian': return z, stats.norm.logpdf(z).sum(axis=1)
        scores = stats.t.ppf(u, df)
        return scores, stats.t.logpdf(scores, df).sum(axis=1)

    def unpack(theta):
        lower = np.eye(d)
        lower[indices] = theta[:len(v0)]
        lower /= np.linalg.norm(lower, axis=1)[:, None]
        correlation = lower@lower.T
        return correlation, (np.exp(theta[-1]) if model == 'student-t' else None)

    def objective(theta):
        try:
            corr, df = unpack(theta)
            scores, marginal = transforms(df)
            joint = (stats.multivariate_normal.logpdf(scores, cov=corr) if model == 'gaussian'
                     else stats.multivariate_t.logpdf(scores, shape=corr, df=df))
            value = -float((joint-marginal).sum())
            return value if np.isfinite(value) else 1e100
        except (ValueError, np.linalg.LinAlgError):
            return 1e100

    bounds = [(-20., 20.)]*len(v0)
    if model == 'student-t': bounds.append((np.log(.25), np.log(200.)))
    runs, attempts = [], []
    for df0 in ([4., 10., 30.] if model == 'student-t' else [None]):
        theta = np.r_[v0, np.log(df0)] if df0 else v0.copy()
        r = optimize.minimize(objective, theta, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': max_iterations, 'ftol': 1e-10, 'maxfun': 100000})
        attempts.append(dict(initial_df=df0, converged=bool(r.success), message=str(r.message), iterations=int(r.nit)))
        if np.isfinite(r.fun) and r.fun < 1e99: runs.append(r)
    if not runs: raise ValueError('No finite copula fit')
    best = min([r for r in runs if r.success] or runs, key=lambda r: r.fun)
    corr, df = unpack(best.x)
    boundary = any(min(abs(v-a), abs(v-b)) < 1e-4 for v, (a, b) in zip(best.x, bounds))
    ll = -float(best.fun)
    return dict(model=model, correlation=corr.tolist(), df=None if df is None else float(df),
                observations=n, dimensions=d, parameters=k, loglik=ll, aic=2*k-2*ll, bic=np.log(n)*k-2*ll,
                converged=bool(best.success), boundary=boundary,
                status='boundary' if best.success and boundary else ('ok' if best.success else 'not_converged'),
                attempts=attempts, fit_sec=time.perf_counter()-start)


def sample_copula(fit, size=1, random_state=None):
    if fit['model'] == 'azzalini-skew-t':
        from .skew_t_copula import sample
        return sample(fit, size, random_state)
    rng = np.random.default_rng(random_state)
    d = len(fit['correlation'])
    z = rng.multivariate_normal(np.zeros(d), fit['correlation'], size=size)
    if fit['model'] == 'gaussian': return stats.norm.cdf(z)
    if fit['model'] != 'student-t': raise ValueError('Unknown copula model')
    df = fit['df']
    z /= np.sqrt(rng.chisquare(df, size=size)/df)[:, None]
    return stats.t.cdf(z, df)


class CopulaJoint:
    """Reconstruct parametric marginals plus copula from runner JSON record."""
    def __init__(self, record):
        if record.get('marginal_mode') != 'fitted':
            raise ValueError('Rank-only fits do not define parametric return marginals')
        self.copula = record['copula']
        if self.copula.get('status', 'ok') != 'ok':
            raise ValueError('Joint reconstruction requires a successful interior copula fit')
        self.marginals = [fitted_distribution(row) for row in record['marginals']]
        self.clip = record.get('cdf_clip', 1e-10)

    def rvs(self, size=1, random_state=None):
        u = sample_copula(self.copula, size, random_state)
        # Only roundoff endpoints are moved to the nearest representable interior.
        u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
        return np.column_stack([marginal_ppf(dist, u[:, j]) for j, dist in enumerate(self.marginals)])

    def logpdf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != len(self.marginals): raise ValueError('Wrong number of assets')
        u = np.column_stack([dist.cdf(x[:, j]) for j, dist in enumerate(self.marginals)])
        if (u <= 0).any() or (u >= 1).any():
            raise ValueError('Marginal CDF rounded to an endpoint; tail log density cannot be evaluated safely')
        marginal_ll = np.column_stack([dist.logpdf(x[:, j]) for j, dist in enumerate(self.marginals)]).sum(axis=1)
        return copula_logpdf(u, self.copula['model'], self.copula['correlation'], self.copula['df'], self.copula.get('alpha'))+marginal_ll
