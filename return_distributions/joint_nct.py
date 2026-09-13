"""Noncentral multivariate t: X=location+(Z+delta)/sqrt(W/df).

Z~N(0, scatter), W~chi-square(df), independently. Delta is in return units.
This is not merely a location-shifted central t.
"""
import warnings
import numpy as np
from scipy import integrate, linalg, special, stats


def log_tilt(m, h):
    """Log E exp(h*chi_m), using mode-centered adaptive quadrature."""
    root = np.hypot(h, 2*np.sqrt(m-1))
    mode = (h+root)/2 if h >= 0 else 2*(m-1)/(root-h)
    width = 1/np.sqrt(1+(m-1)/mode**2)
    peak = (m-1)*np.log(mode)-mode*mode/2+h*mode
    def integrand(v):
        r = mode+width*v
        if r <= 0: return 0.
        return np.exp((m-1)*np.log(r)-r*r/2+h*r-peak)
    # Split at the peak so even very narrow densities cannot be missed.
    left, e1 = integrate.quad(integrand, -mode/width, 0, epsabs=1e-10, epsrel=1e-10, limit=200)
    right, e2 = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    value = left+right
    if not np.isfinite(value) or value <= 0 or e1+e2 > 1e-7*value:
        raise ValueError('Noncentral-t density quadrature failed')
    return peak+np.log(width*value)-(m/2-1)*np.log(2)-special.gammaln(m/2)


def logdensity(x, mu, chol, a, df):
    """Reduce the mixing integral to a univariate nct density ratio.

With whitened y and a, Q=y'y, B=a'y, C=a'a, m=df+d:
f/f_central=exp(-C/2) E[exp(B*chi_m/sqrt(df+Q))].
Matching this tilted-chi integral to a scalar nct avoids per-row quadrature
in ordinary cases. Ill-conditioned ratios use direct positive integration.
"""
    y = linalg.solve_triangular(chol, (np.atleast_2d(x)-mu).T, lower=True)
    d = len(mu); q = np.sum(y*y, axis=0); b = a@y; c = float(a@a)
    m = df+d
    central = (special.gammaln(m/2)-special.gammaln(df/2)-d/2*np.log(df*np.pi)
               -np.log(chol.diagonal()).sum()-m/2*np.log1p(q/df))
    if c == 0: return central
    nc = np.divide(b, np.sqrt(q), out=np.zeros_like(b), where=q > 0)
    t = np.sqrt(q*(m-1)/df)
    # Avoid calling SciPy in known problematic regions (it can throw, not just
    # return NaN). Warnings and exceptions also trigger positive quadrature.
    fallback = (nc < -5) | (np.abs(nc) > 15) | (df > 300)
    ratio = np.full_like(q, np.nan)
    safe = ~fallback
    if safe.any():
        try:
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                ratio[safe] = stats.nct.logpdf(t[safe], m-1, nc[safe])-stats.t.logpdf(t[safe], m-1)-(c-nc[safe]**2)/2
        except (RuntimeWarning, ArithmeticError):
            pass
    fallback |= ~np.isfinite(ratio)
    for i in np.flatnonzero(fallback):
        ratio[i] = -c/2+log_tilt(m, b[i]/np.sqrt(df+q[i]))
    ratio[q == 0] = -c/2
    return central+ratio


class JointNCT:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'], dtype=float)
        self.scatter = np.asarray(fit['scatter'], dtype=float)
        self.delta = np.asarray(fit['delta'], dtype=float)
        self.df = float(fit['df'])
        d = len(self.location)
        if (self.scatter.shape != (d, d) or self.delta.shape != (d,)
                or not np.isfinite(self.location).all() or not np.isfinite(self.scatter).all()
                or not np.isfinite(self.delta).all() or not np.isfinite(self.df) or self.df <= 0
                or not np.allclose(self.scatter, self.scatter.T)):
            raise ValueError('Invalid multivariate noncentral-t parameters')
        self.chol = np.linalg.cholesky(self.scatter)
        self.a = linalg.solve_triangular(self.chol, self.delta, lower=True)

    def logpdf(self, x):
        value = logdensity(x, self.location, self.chol, self.a, self.df)
        return value[0] if np.asarray(x).ndim == 1 else value

    def pdf(self, x): return np.exp(self.logpdf(x))

    def mean(self):
        if self.df <= 1: return np.full_like(self.location, np.nan)
        b = np.sqrt(self.df/2)/special.poch((self.df-1)/2, .5)
        return self.location+b*self.delta

    def cov(self):
        if self.df <= 2: return np.full_like(self.scatter, np.nan)
        b = np.sqrt(self.df/2)/special.poch((self.df-1)/2, .5)
        factor = self.df/(self.df-2)
        return factor*self.scatter+(factor-b*b)*np.outer(self.delta, self.delta)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        z = rng.normal(size=(size, len(self.location)))@self.chol.T+self.delta
        return self.location+z/np.sqrt(rng.chisquare(self.df, size=size)/self.df)[:, None]


class StableNCT(stats.nct.__class__):
    """SciPy CDF/quantiles/moments with the safeguarded mixture density."""
    def _logpdf(self, x, df, nc):
        x, df, nc = np.broadcast_arrays(x, df, nc)
        result = np.empty(x.shape)
        for index in np.ndindex(x.shape):
            if not np.isfinite(x[index]):
                result[index] = np.nan if np.isnan(x[index]) else -np.inf
            else:
                result[index] = logdensity([x[index]], np.zeros(1), np.eye(1), np.array([nc[index]]), df[index])[0]
        return result

    def _pdf(self, x, df, nc):
        return np.exp(self._logpdf(x, df, nc))


stable_nct = StableNCT(name='nct', shapes='df, nc')
