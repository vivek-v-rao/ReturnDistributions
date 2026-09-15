"""Univariate distributions of linear combinations of fitted joint returns."""
import numpy as np
from scipy import integrate, optimize, special, stats

from .joint_gh import GH_MODELS
from .joint_power import POWER_MODELS, covariance_factor


class PointMass:
    """Zero-weight portfolio. No ordinary Lebesgue density exists."""
    def mean(self): return 0.
    def var(self): return 0.
    def std(self): return 0.
    def pdf(self, x): return np.full_like(np.asarray(x, dtype=float), np.nan)
    def cdf(self, x): return np.asarray(x) >= 0
    def sf(self, x): return np.asarray(x) < 0
    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return np.where((q >= 0) & (q <= 1), 0., np.nan)
    def rvs(self, size=1, random_state=None): return np.zeros(size)


class ProjectedPower:
    """Elliptical projection retaining ORIGINAL joint dimension and power.

    Survival integration uses X=R*U1, R**power~Gamma(d/power,1),
    U uniform on the unit sphere. Numerical error target is 1e-9 absolute.
    """
    def __init__(self, location, scale, dimension, power):
        self.location, self.scale, self.dimension, self.power = location, scale, dimension, power
        self.factor = covariance_factor(dimension, power)

    def mean(self): return self.location
    def var(self): return self.scale**2*self.factor
    def std(self): return np.sqrt(self.var())

    def _tail(self, z):
        if z == 0: return .5
        if np.isinf(z): return 0.
        d, p = self.dimension, self.power
        constant = np.exp(special.gammaln(d/2)-.5*np.log(np.pi)-special.gammaln((d-1)/2))
        def integrand(theta):
            with np.errstate(over='ignore'):
                cutoff = (z/np.cos(theta))**p
            return constant*np.sin(theta)**(d-2)*special.gammaincc(d/p, cutoff)
        value, error = integrate.quad(integrand, 0, np.pi/2, epsabs=1e-10, epsrel=1e-8, limit=200)
        if error > 1e-8:
            raise ValueError('Projected-power CDF integration did not meet tolerance')
        return value

    def sf(self, x):
        def one(v):
            if np.isnan(v): return np.nan
            z = (v-self.location)/self.scale
            return self._tail(z) if z >= 0 else 1-self._tail(-z)
        return np.vectorize(one, otypes=[float])(x)

    def cdf(self, x):
        return self.sf(2*self.location-np.asarray(x))

    def pdf(self, x):
        d, p = self.dimension, self.power
        unit = np.sqrt(self.factor)
        logconstant = (np.log(p)+special.gammaln(d/2)-.5*np.log(np.pi)
                       -special.gammaln(d/p)-special.gammaln((d-1)/2))
        def one(v):
            z = abs((v-self.location)/self.scale)
            if np.isnan(z): return np.nan
            if np.isinf(z): return 0.
            def integrand(t):
                if t == 0: return 0.
                return np.exp(logconstant+(d-1)*np.log(unit)+(d-2)*np.log(t)-(z*z+(unit*t)**2)**(p/2))
            value, error = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=200)
            if error > 1e-8: raise ValueError('Projected-power density integration did not meet tolerance')
            return value/self.scale
        return np.vectorize(one, otypes=[float])(x)

    def ppf(self, q):
        def one(prob):
            if not 0 <= prob <= 1: return np.nan
            if prob == 0: return -np.inf
            if prob == 1: return np.inf
            if prob == .5: return self.location
            tail = min(prob, 1-prob)
            high = np.sqrt(self.factor)
            for _ in range(100):
                if self._tail(high) <= tail: break
                high *= 2
            else: raise ValueError('Could not bracket projected quantile')
            z = optimize.brentq(lambda v: self._tail(v)-tail, 0, high, xtol=1e-10)
            return self.location+self.scale*z*(-1 if prob < .5 else 1)
        return np.vectorize(one, otypes=[float])(q)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = np.sqrt(rng.beta(.5, (self.dimension-1)/2, size=size))*rng.choice([-1., 1.], size=size)
        r = rng.gamma(self.dimension/self.power, size=size)**(1/self.power)
        return self.location+self.scale*r*u


def project_distribution(fit, weights, *, allow_log=False):
    """Project ordered weights exactly, without normalization or cash assumptions.

    Log inputs require opt-in: the result is a weighted log-return combination,
    NOT the log return of a constant-weight portfolio.
    """
    if fit.get('status', 'ok') != 'ok':
        raise ValueError('Projection requires an interior, successfully converged fit')
    if fit.get('return_type') == 'log' and not allow_log:
        raise ValueError('Log-return fits require allow_log=True; the projection is not a portfolio log return')
    from .vol_standardization import conditional_weights
    w = conditional_weights(fit, weights)
    if fit.get('components', 1) > 1:
        from .joint_finite_mixture import ProjectedFiniteMixture, JointFiniteMixture
        JointFiniteMixture(fit)  # Validate the saved component specification.
        records = fit['component_fits']
        if w.shape != (len(records[0]['location']),) or not np.isfinite(w).all():
            raise ValueError('Weights must be finite and match the joint symbol order')
        if not np.any(w): return PointMass()
        locations = [float(w@np.asarray(r['location'])) for r in records]
        scales = [float(np.sqrt(w@np.asarray(r['scatter'])@w)) for r in records]
        if not np.isfinite(scales).all() or np.any(np.asarray(scales) <= 0) or not np.isfinite(locations).all():
            raise ValueError('Numerically invalid mixture projection')
        distributions = [project_distribution(r, w, allow_log=allow_log) for r in records] if fit['model'] in ('ged', 'nig-skewed') else None
        return ProjectedFiniteMixture(fit['mixture_weights'], locations, scales, fit.get('df'), distributions=distributions)
    mu, scatter = np.asarray(fit['location']), np.asarray(fit['scatter'])
    d = len(mu)
    if w.shape != (d,) or not np.isfinite(w).all():
        raise ValueError('Weights must be finite and match the joint symbol order')
    if scatter.shape != (d, d) or not np.isfinite(scatter).all() or not np.isfinite(mu).all():
        raise ValueError('Invalid location/scatter')
    if not np.allclose(scatter, scatter.T, rtol=1e-10, atol=1e-14):
        raise ValueError('Scatter is not symmetric')
    np.linalg.cholesky(scatter)
    if not np.any(w): return PointMass()
    loc = float(w@mu)
    scale = np.sqrt(float(w@scatter@w))
    if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Weights produce a numerically invalid projected location/scale')
    model = fit['model']
    if model in {'ged-skewed', 'fs-skew-normal'}:
        raise ValueError('Skewed power portfolios require simulate_joint_portfolio; no univariate family closure')
    if model in {'nts-symmetric','nts-skewed'}:
        from .joint_nts import JointNTS
        return JointNTS(fit).project(w)
    if model == 'generalized-t-skewed':
        raise ValueError('Skewed generalized t portfolios require simulate_joint_portfolio; no univariate generalized-t closure')
    if model == 'generalized-t':
        from .generalized_t import generalized_t
        from .joint_generalized_t import ProjectedGeneralizedT
        p, q = fit['power'], fit['q']
        if d == 1: return generalized_t(p, q, 0., loc=loc, scale=scale)
        if p == 2: return stats.t(2*q, loc=loc, scale=scale/np.sqrt(2))
        return ProjectedGeneralizedT(loc, scale, d, p, q)
    from .joint_laplace_mixture import LAPLACE_MIXTURE_MODELS
    if model in LAPLACE_MIXTURE_MODELS:
        gamma=float(w@np.asarray(fit['gamma']))
        kappa=np.exp(-np.arcsinh(gamma/(np.sqrt(2)*scale)))
        return stats.laplace_asymmetric(kappa,loc=loc,scale=scale/np.sqrt(2))
    if model == 'gh-skew-t':
        from .gh_skew_t import gh_skew_t
        return gh_skew_t(fit['df'],float(w@np.asarray(fit['gamma']))/scale,loc=loc,scale=scale)
    from .variance_gamma import VG_MODELS, variance_gamma
    if model in VG_MODELS:
        return variance_gamma(fit['vg_shape'], float(w@np.asarray(fit['gamma']))/scale, loc=loc, scale=scale)
    from .joint_slash import SLASH_MODELS, JointSlash
    if model in SLASH_MODELS:
        return JointSlash(fit).project(w)
    if model.startswith('sdb-'):
        raise ValueError('SDB portfolio sums are not ordinary univariate skew-t; use simulate_joint_portfolio or the portfolio CLI')
    if model == 'normal': return stats.norm(loc=loc, scale=scale)
    if model == 'student-t': return stats.t(fit['df'], loc=loc, scale=scale)
    if model == 'noncentral-t':
        from .joint_nct import JointNCT, stable_nct
        return stable_nct(fit['df'], float(w@JointNCT(fit).delta)/scale, loc=loc, scale=scale)
    if model == 'azzalini-skew-t':
        from .joint_skew_t import JointSkewT
        from .skew_t import azzalini_skew_t
        delta = float(w@JointSkewT(fit).delta)/scale
        shape = delta/np.sqrt(1-delta*delta)
        return azzalini_skew_t(fit['df'], shape, loc=loc, scale=scale)
    if model in GH_MODELS:
        if fit.get('chi', 1) != 1: raise ValueError('GH chi must be 1')
        b = float(w@np.asarray(fit['gamma']))/scale
        a = np.sqrt(fit['psi']+b*b)
        return stats.genhyperbolic(fit['lambda'], a, b, loc=loc, scale=scale)
    if model in POWER_MODELS:
        p = fit['power']
        if d == 1: return stats.gennorm(p, loc=loc, scale=scale)
        if p == 2: return stats.norm(loc=loc, scale=scale/np.sqrt(2))
        return ProjectedPower(loc, scale, d, p)
    raise ValueError('Unknown joint family')
