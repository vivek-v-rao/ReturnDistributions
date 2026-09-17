"""Azzalini--Capitanio skew-t copula, two-stage maximum likelihood.

References: Yoshiba (2014), ISM Research Memorandum 1183; Azzalini and
Capitanio (2003). Original implementation; no reference code copied.
Location and marginal scatter scales are fixed at zero and one. Omega is
latent scatter correlation, NOT the Pearson correlation of skew-t draws.
"""
from functools import lru_cache
import time
import numpy as np
from scipy import optimize, special, stats
from scipy.interpolate import PchipInterpolator, PPoly
from .skew_t import azzalini_skew_t
from .joint_skew_t import JointSkewT


@lru_cache(maxsize=16)
def _quadrature(df):
    nodes, weights = special.roots_jacobi(64, 0., df-1)
    return (nodes+1)/2, weights/weights.sum()


def marginal_cdf(x, df, shape):
    """Deterministic Gauss-Jacobi integration of the skew-t angle density.

    The Jacobi weight removes the endpoint singularity even for df < 1.
    Reflection evaluates only the negative half-line. Chunk to bound memory.
    """
    x = np.asarray(x, dtype=float)
    if shape == 0: return special.stdtr(df, x)
    flat = x.reshape(-1)
    result = np.empty_like(flat)
    nodes, weights = _quadrature(float(df))
    logc = special.gammaln((df+1)/2)-special.gammaln(df/2)-.5*np.log(np.pi)
    for start in range(0, len(flat), 2048):
        z = flat[start:start+2048]
        a = np.where(z > 0, -shape, shape)
        upper = np.arctan2(np.sqrt(df), np.abs(z))
        angle = upper[:, None]*nodes
        integrand = np.exp((df-1)*np.log(np.sinc(angle/np.pi)))
        integrand *= special.stdtr(df+1, -a[:, None]*np.sqrt(df+1)*np.cos(angle))
        with np.errstate(divide='ignore'):
            lower = 2*np.exp(logc+df*np.log(upper))/df*(integrand@weights)
        result[start:start+len(z)] = np.where(z > 0, 1-lower, lower)
    return np.clip(result.reshape(x.shape), 0., 1.)


def marginal_ppf(u, df, shape):
    """PCHIP initialization followed by bracketed vector Newton refinement."""
    u = np.asarray(u, dtype=float)
    if shape == 0: return stats.t.ppf(u, df)
    # Widen the density<=2*t bounds to avoid roundoff placing an endpoint
    # just outside the bracket when skewness saturates the t-CDF factor.
    lo, hi = stats.t.ppf(u/4, df), stats.t.isf((1-u)/4, df)
    grid = np.linspace(np.arcsinh(lo.min()), np.arcsinh(hi.max()), 513)
    cdf = marginal_cdf(np.sinh(grid), df, shape)
    keep = (cdf > 0) & (cdf < 1)
    grid, cdf = grid[keep], cdf[keep]
    keep = np.r_[True, np.diff(cdf) > 0]
    guess = PchipInterpolator(special.logit(cdf[keep]), grid[keep], extrapolate=False)(special.logit(u))
    x = np.sinh(guess)
    x = np.where(np.isfinite(x), np.clip(x, lo, hi), (lo+hi)/2)
    tolerance = 2e-11*np.minimum(u, 1-u)+2e-15
    for _ in range(45):
        error = marginal_cdf(x, df, shape)-u
        done = np.abs(error) <= tolerance
        if done.all(): return x
        lo = np.where(error < 0, x, lo)
        hi = np.where(error > 0, x, hi)
        pdf = np.exp(azzalini_skew_t.logpdf(x, df, shape))
        with np.errstate(divide='ignore', invalid='ignore'):
            candidate = x-error/pdf
        safe = np.isfinite(candidate) & (candidate > lo) & (candidate < hi)
        x = np.where(done, x, np.where(safe, candidate, (lo+hi)/2))
    raise ValueError('Skew-t copula quantile refinement did not meet tolerance')


def latent_model(correlation, df, alpha):
    corr, alpha = np.asarray(correlation), np.asarray(alpha, dtype=float)
    if corr.ndim != 2 or not np.allclose(np.diag(corr), 1):
        raise ValueError('Skew-t copula requires unit scatter diagonal')
    model = JointSkewT(dict(location=np.zeros(len(corr)), scatter=corr, alpha=alpha, df=df))
    shape = model.delta/np.sqrt(1-model.delta**2)
    return model, shape


def _quintic_inverse(grid, df, shape):
    """Inverse logit-CDF -> asinh(x), with analytic first/second derivatives."""
    x = np.sinh(grid)
    cdf = marginal_cdf(x, df, shape)
    keep = (cdf > 0) & (cdf < 1)
    grid, x, cdf = grid[keep], x[keep], cdf[keep]
    keep = np.r_[True, np.diff(cdf) > 0]
    grid, x, cdf = grid[keep], x[keep], cdf[keep]
    v = special.logit(cdf)
    h = np.hypot(1., x)
    radius = np.hypot(np.sqrt(df), x)
    arg = shape*np.sqrt(df+1)*x/radius
    logpdf = azzalini_skew_t.logpdf(x, df, shape)
    first = cdf*(1-cdf)*np.exp(-logpdf)/h
    slope = (-(df+1)*x/radius**2 +
             np.exp(stats.t.logpdf(arg, df+1)-stats.t.logcdf(arg, df+1))*
             shape*np.sqrt(df+1)*df/radius**3)
    second = first*((1-2*cdf)-first*(slope*h+x/h))
    step = np.diff(v)
    c0, c1, c2 = grid[:-1], step*first[:-1], .5*step**2*second[:-1]
    r0 = grid[1:]-c0-c1-c2
    r1 = step*first[1:]-c1-2*c2
    r2 = step**2*second[1:]-2*c2
    c3, c4, c5 = 10*r0-4*r1+.5*r2, -15*r0+7*r1-r2, 6*r0-3*r1+.5*r2
    coefficients = np.vstack([c5/step**5, c4/step**4, c3/step**3, c2/step**2, c1/step, c0])
    return PPoly(coefficients, v, extrapolate=False), v


def _fast_marginal_ppf(u, df, shape, bounds=None):
    """Audited quintic interpolation for large likelihood samples only.

    Integration work scales with table size rather than observation count.
    Bad tables fall back to the original per-observation Newton inversion.
    """
    if shape == 0: return special.stdtrit(df, u)
    if bounds is None:
        bounds = (np.arcsinh(special.stdtrit(df, float(np.min(u))/4)),
                  np.arcsinh(-special.stdtrit(df, (1-float(np.max(u)))/4)))
    scores = special.logit(u)
    for size in (513, 1025, 2049):
        try:
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                inverse, v = _quintic_inverse(np.linspace(*bounds, size), df, shape)
                # Check every interval midpoint against the integrated CDF.
                mid = (v[1:]+v[:-1])/2
                target = special.expit(mid)
                probe_x = np.sinh(inverse(mid))
                error = np.abs(marginal_cdf(probe_x, df, shape)-target)
                good = (np.isfinite(error).all() and
                        np.all(error <= 2e-13+2e-11*np.minimum(target, 1-target)) and
                        np.all(inverse.derivative()(mid) > 0))
                if good:
                    x = np.sinh(inverse(scores))
                    if np.isfinite(x).all(): return x
        except (ValueError, FloatingPointError):
            pass
    return marginal_ppf(u, df, shape)


def logpdf(u, correlation, df, alpha):
    model, shapes = latent_model(correlation, df, alpha)
    z = np.column_stack([marginal_ppf(u[:, j], df, a) for j, a in enumerate(shapes)])
    marginal = sum(azzalini_skew_t.logpdf(z[:, j], df, a) for j, a in enumerate(shapes))
    return model.logpdf(z)-marginal


def sample(fit, size, random_state):
    model, shapes = latent_model(fit['correlation'], fit['df'], fit['alpha'])
    z = model.rvs(size, random_state)
    return np.column_stack([marginal_cdf(z[:, j], model.df, a) for j, a in enumerate(shapes)])


def fit(u, max_iterations):
    from .copulas import fit_copula
    started = time.perf_counter()
    n, d = u.shape
    nested = fit_copula(u, 'student-t', max_iterations)
    chol = np.linalg.cholesky(nested['correlation'])
    ix = np.tril_indices(d, -1)
    v = (chol/chol.diagonal()[:, None])[ix]
    nc = len(v)
    # Normalized triangular rows keep Omega PD. Unrestricted alpha implies
    # delta'Omega^-1 delta < 1, hence the augmented selection matrix is PD.
    bounds = [(-20., 20.)]*nc+[(-12., 12.)]*d+[(np.log(.25), np.log(200.))]

    @lru_cache(maxsize=16)
    def quantile_bounds(df):
        return [(np.arcsinh(special.stdtrit(df, float(u[:, j].min())/4)),
                 np.arcsinh(-special.stdtrit(df, (1-float(u[:, j].max()))/4))) for j in range(d)]

    def fast_loglik(corr, df, alpha):
        model, shapes = latent_model(corr, df, alpha)
        limits = quantile_bounds(df)
        z = np.column_stack([_fast_marginal_ppf(u[:, j], df, a, limits[j])
                             for j, a in enumerate(shapes)])
        marginal = sum(azzalini_skew_t.logpdf(z[:, j], df, a) for j, a in enumerate(shapes))
        return float(np.sum(model.logpdf(z)-marginal))

    def unpack(theta):
        lower = np.eye(d)
        lower[ix] = theta[:nc]
        lower /= np.linalg.norm(lower, axis=1)[:, None]
        return lower@lower.T, np.exp(theta[-1]), theta[nc:nc+d]

    def objective(theta):
        try:
            value = -fast_loglik(*unpack(theta)) if n >= 256 else -float(logpdf(u, *unpack(theta)).sum())
            return value if np.isfinite(value) else 1e100
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e100

    # A zero-shape start alone can stall at the symmetric stationary point.
    direction = np.where(np.arange(d) % 2, -1., 1.)
    starts = [np.zeros(d), np.ones(d), -np.ones(d), direction, -direction]
    nested_theta = np.r_[v, np.zeros(d), np.log(nested['df'])]
    attempts = []
    runs = [optimize.OptimizeResult(x=nested_theta, fun=objective(nested_theta),
                                   success=nested['converged'])]
    for a in {tuple(a): a for a in starts}.values():
        initial = np.r_[v, a, np.log(nested['df'])]
        r = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
            options=dict(maxiter=max_iterations, maxfun=100000, ftol=1e-10, gtol=1e-5, eps=1e-5))
        attempts.append(dict(initial_alpha=a.tolist(), converged=bool(r.success),
                             loglik=float(-r.fun), message=str(r.message), iterations=int(r.nit)))
        if np.isfinite(r.fun) and r.fun < 1e99: runs.append(r)
    if not runs: raise ValueError('No finite skew-t copula fit')
    best = min(runs, key=lambda r: r.fun)
    corr, df, alpha = unpack(best.x)
    exact_ll = float(logpdf(u, corr, df, alpha).sum())
    likelihood_error = abs(exact_ll+float(best.fun))
    model, shapes = latent_model(corr, df, alpha)
    # Independent adaptive-quadrature audit at sample quantiles and tails.
    audit_error = 0.
    for j, a in enumerate(shapes):
        probes = np.unique(np.quantile(u[:, j], np.linspace(0, 1, 21)))
        z = marginal_ppf(probes, df, a)
        reference = azzalini_skew_t.cdf(z, df, a)
        if not np.isfinite(reference).all():
            raise ValueError('Nonfinite adaptive-quadrature CDF audit')
        audit_error = max(audit_error, float(np.max(np.abs(reference-probes))))
    boundary = any(min(abs(x-lo), abs(x-hi)) < 1e-4 for x, (lo, hi) in zip(best.x, bounds))
    status = 'ok' if best.success else 'not_converged'
    if best.success and boundary: status = 'boundary'
    if audit_error > 2e-7: status = 'numerical_failure'
    if likelihood_error > 1e-5: status = 'numerical_failure'
    k, ll = nc+d+1, exact_ll
    return dict(model='azzalini-skew-t', correlation=corr.tolist(), df=float(df), alpha=alpha.tolist(),
        delta=model.delta.tolist(), marginal_shapes=shapes.tolist(), correlation_kind='latent-scatter',
        observations=n, dimensions=d, parameters=k, loglik=ll, aic=2*k-2*ll, bic=np.log(n)*k-2*ll,
        converged=bool(best.success), boundary=boundary, status=status, attempts=attempts,
        cdf_audit_max_error=audit_error, likelihood_audit_error=likelihood_error,
        quantile_method='audited-quintic' if n >= 256 else 'newton',
        nested_t_loglik=nested['loglik'], fit_sec=time.perf_counter()-started)


def print_details(fit, symbols):
    if fit.get('model') != 'azzalini-skew-t' or 'alpha' not in fit: return
    print('Joint copula shape alpha: '+', '.join(f'{s}={v:.3f}' for s, v in zip(symbols, fit['alpha'])))
    print(f'Copula df: {fit["df"]:.3f}; CDF audit max error: {fit["cdf_audit_max_error"]:.2e}')
    if 'likelihood_audit_error' in fit:
        print(f'Final exact likelihood audit difference: {fit["likelihood_audit_error"]:.2e}')
    print('Matrix is latent scatter correlation, not Pearson return correlation. Zero alpha gives the Student-t copula.')
