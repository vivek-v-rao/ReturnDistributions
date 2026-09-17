"""Cached, audited NIG quantiles for copula marginal transformations.

PINV interpolates the inverse CDF; it does not replace the copula uniforms by
independent marginal draws. Density fitting and likelihoods are unchanged.
"""
from functools import lru_cache
import warnings
import numpy as np
from scipy import integrate, optimize, special
from scipy.stats.sampling import NumericalInversePolynomial


TAIL_CUTOFF = 1e-7


class _NIGDensity:
    def __init__(self, a, b):
        if not np.isfinite([a, b]).all() or a <= abs(b):
            raise ValueError('NIG requires finite a > abs(b)')
        self.a, self.b = a, b
        self.gamma = np.sqrt((a-b)*(a+b))

    def logpdf(self, x):
        if not np.isfinite(x): return -np.inf
        h = np.hypot(1., x)
        # Stable b*x-a*h, including strongly skewed tails.
        exponent = (self.b*np.sign(x)-self.a)*abs(x)-self.a/(h+abs(x))
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return np.log(self.a/np.pi)+self.gamma+exponent+np.log(special.k1e(self.a*h))-np.log(h)

    def pdf(self, x):
        return np.exp(self.logpdf(x))


def _lower_cdf(x, a, b):
    density = _NIGDensity(a, b)
    # Relative error control also applies to probabilities far below 1e-8.
    value, error = integrate.quad(density.pdf, -np.inf, x, epsabs=0., epsrel=2e-10, limit=250)
    if not np.isfinite(value) or not 0 <= value <= 1+1e-9 or error > max(1e-300, 1e-7*value):
        raise ValueError('NIG CDF quadrature failed accuracy check')
    return value


def _tail_ppf(u, a, b):
    # Invert a lower-tail probability using reflection, avoiding 1-CDF loss
    # of precision and SciPy's occasionally failing generic tail bracket.
    reflected = u > .5
    p, shape = (1-u, -b) if reflected else (u, b)
    left, right = -1., 1.
    for _ in range(80):
        if _lower_cdf(left, a, shape) <= p: break
        left *= 2
    else: raise ValueError('Could not bracket NIG lower quantile')
    for _ in range(80):
        if _lower_cdf(right, a, shape) >= p: break
        right *= 2
    else: raise ValueError('Could not bracket NIG upper quantile')
    x = optimize.brentq(lambda x: _lower_cdf(x, a, shape)-p, left, right, xtol=1e-11, rtol=1e-12)
    return -x if reflected else x


@lru_cache(maxsize=16)
def _inverse(a, b):
    """Reuse a standardized interpolator across batches, scales and copulas."""
    density = _NIGDensity(a, b)
    try:
        interpolator = NumericalInversePolynomial(density, center=b/density.gamma,
                                                  u_resolution=1e-12, random_state=0)
        lower = np.geomspace(TAIL_CUTOFF, .01, 9)
        probes = np.unique(np.r_[lower, np.linspace(.01, .99, 25), 1-lower])
        x = interpolator.ppf(probes)
        if not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
            raise ValueError('Nonfinite or nonmonotone NIG inverse table')
        errors = []
        for u, value in zip(probes, x):
            actual = _lower_cdf(-value if u > .5 else value, a, -b if u > .5 else b)
            errors.append(abs(actual-min(u, 1-u)))
        if max(errors) > 5e-11:
            raise ValueError(f'NIG inverse audit error {max(errors):.3g}')
        return interpolator
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        warnings.warn(f'Fast NIG quantiles unavailable; using slower numerical inversion: {exc}', RuntimeWarning)
        return None


def nig_ppf(q, a, b, loc=0., scale=1.):
    """Vector quantiles with checked interpolation and untruncated tail fallback."""
    _NIGDensity(a, b)
    if not np.isfinite([loc, scale]).all() or scale <= 0:
        raise ValueError('Invalid NIG location or scale')
    q = np.asarray(q, dtype=float)
    flat = q.ravel()
    result = np.full(flat.shape, np.nan)
    result[flat == 0], result[flat == 1] = -np.inf, np.inf
    valid = (flat > 0) & (flat < 1)
    fast = valid & (flat >= TAIL_CUTOFF) & (flat <= 1-TAIL_CUTOFF)
    if fast.any():
        inverse = _inverse(float(a), float(b))
        if inverse is None:
            fast[:] = False
        else:
            result[fast] = inverse.ppf(flat[fast])
            fast &= np.isfinite(result)
    for i in np.flatnonzero(valid & ~fast):
        result[i] = _tail_ppf(flat[i], a, b)
    result = (loc+scale*result).reshape(q.shape)
    return result.item() if result.ndim == 0 else result


def marginal_ppf(distribution, q):
    """Dispatch only NIG frozen marginals; other distributions retain their PPF."""
    if getattr(getattr(distribution, 'dist', None), 'name', None) != 'norminvgauss':
        return distribution.ppf(q)
    args, kwds = distribution.args, distribution.kwds
    a = args[0] if len(args) > 0 else kwds['a']
    b = args[1] if len(args) > 1 else kwds['b']
    loc = args[2] if len(args) > 2 else kwds.get('loc', 0.)
    scale = args[3] if len(args) > 3 else kwds.get('scale', 1.)
    return nig_ppf(q, a, b, loc, scale)
