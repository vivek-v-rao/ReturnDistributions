"""Variance gamma with identified Gamma(shape, rate=shape) mixing.

Fitting is restricted to shape > dimension/2 + .05: densities are bounded
at the location. This is a restricted VG family, not unrestricted VG MLE.
"""
import numpy as np
from scipy import integrate, linalg, special, stats

VG_MODELS = ('variance-gamma-symmetric', 'variance-gamma-skewed')


def vg_logpdf(x, mu, chol, gamma, shape):
    y = linalg.solve_triangular(chol, (np.atleast_2d(x)-mu).T, lower=True)
    g = linalg.solve_triangular(chol, gamma, lower=True)
    q = np.sum(y*y, axis=0)
    b = 2*shape + g@g
    order = shape-len(mu)/2
    constant = (shape*np.log(shape)-special.gammaln(shape)
                -len(mu)/2*np.log(2*np.pi)-np.log(chol.diagonal()).sum()+g@y)
    result = np.empty_like(q)
    positive = q > 0
    z = np.sqrt(q[positive]*b)
    result[positive] = (np.log(2)+order/2*np.log(q[positive]/b)
                        +np.log(special.kve(order, z))-z)
    result[~positive] = (special.gammaln(order)+order*np.log(2/b)
                         if order > 0 else np.inf)
    return constant+result


class JointVG:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'], dtype=float)
        self.scatter = np.asarray(fit['scatter'], dtype=float)
        self.gamma = np.asarray(fit['gamma'], dtype=float)
        self.shape = float(fit['vg_shape'])
        if self.shape <= 0 or not np.isfinite(self.shape):
            raise ValueError('VG shape must be finite and positive')
        self.chol = np.linalg.cholesky(self.scatter)

    def logpdf(self, x):
        value = vg_logpdf(x, self.location, self.chol, self.gamma, self.shape)
        return value[0] if np.asarray(x).ndim == 1 else value

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def mean(self):
        return self.location+self.gamma

    def cov(self):
        return self.scatter+np.outer(self.gamma, self.gamma)/self.shape

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        w = rng.gamma(self.shape, 1/self.shape, size=size)
        z = rng.normal(size=(size, len(self.location)))@self.chol.T
        return self.location+w[:, None]*self.gamma+np.sqrt(w[:, None])*z


class VarianceGamma(stats.rv_continuous):
    def _argcheck(self, vg_shape, b):
        return (vg_shape > .55) & (vg_shape <= 100) & np.isfinite(b)

    def _fitstart(self, data):
        # A zero shape seed creates a very narrow Nelder-Mead simplex in b.
        # Use sample skewness to give the skewed fit a meaningful initial step.
        b = float(np.clip(stats.skew(data)*.5, -2., 2.))
        scale = float(np.std(data)/np.sqrt(1+b*b/2))
        return (2., b, float(np.mean(data))-scale*b, scale)

    def _logpdf(self, x, vg_shape, b):
        x, k, b = np.broadcast_arrays(x, vg_shape, b)
        order = k-.5
        rate = 2*k+b*b
        q = x*x
        with np.errstate(all='ignore'):
            z = np.sqrt(q*rate)
            kernel = np.log(2)+order/2*np.log(q/rate)+np.log(special.kve(order,z))-z
            kernel = np.where(q == 0, special.gammaln(order)+order*np.log(2/rate), kernel)
            value = k*np.log(k)-special.gammaln(k)-.5*np.log(2*np.pi)+b*x+kernel
        return np.where(np.isinf(x), -np.inf, value)

    def _pdf(self, x, vg_shape, b):
        return np.exp(self._logpdf(x, vg_shape, b))

    def _cdf(self, x, vg_shape, b):
        return self._tail(x, vg_shape, b, False)

    def _sf(self, x, vg_shape, b):
        return self._tail(x, vg_shape, b, True)

    def _tail(self, x, k, b, survival):
        def scalar(v, s, g):
            if np.isinf(v): return float(v < 0) if survival else float(v > 0)
            def integrand(w):
                if w == 0: return 0.
                z = (v-g*w)/np.sqrt(w)
                return stats.gamma.pdf(w, s, scale=1/s)*special.ndtr(-z if survival else z)
            value, error = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=250)
            if error > max(1e-8, abs(value)*1e-6):
                raise ValueError('VG CDF integration failed')
            return value
        return np.vectorize(scalar, otypes=[float])(x, k, b)

    def _stats(self, k, b):
        variance = 1+b*b/k
        third = 3*b/k+2*b**3/k**2
        fourth_cumulant = 3/k+12*b*b/k**2+6*b**4/k**3
        return b, variance, third/variance**1.5, fourth_cumulant/variance**2

    def _rvs(self, k, b, size=None, random_state=None):
        w = random_state.gamma(k, 1/k, size=size)
        return b*w+np.sqrt(w)*random_state.normal(size=size)


variance_gamma = VarianceGamma(name='variance_gamma', shapes='vg_shape, b')
