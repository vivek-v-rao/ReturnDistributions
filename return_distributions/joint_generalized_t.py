"""Elliptical generalized t with R**power/q ~ BetaPrime(d/power,q).

This explicitly specified radial extension reduces to McDonald--Newey in d=1,
Student-t at power=2 (t scatter = scatter/2), and elliptical GED as q->infinity.
General projections retain the original dimension; they are not univariate GT.
The opt-in skewed model two-piece scales the spherical Cholesky coordinates;
the shared radius is retained (coordinates are not independent). Its density
is normalized by product(cosh(skewness)); asset order is part of the model.
"""
import time
import warnings

import numpy as np
from scipy import integrate, linalg, optimize, special


def radial_moment(d, power, q, order):
    if power*q <= order:
        return np.inf
    return float(np.exp(order*np.log(q)/power+special.betaln((d+order)/power, q-order/power)-special.betaln(d/power, q)))


def covariance_factor(d, power, q):
    return radial_moment(d, power, q, 2)/d


def logpdf(x, location, chol, power, q, skewness=None):
    z = linalg.solve_triangular(chol, (np.atleast_2d(x)-location).T, lower=True, check_finite=False)
    d = len(location)
    correction = 0.
    if skewness is not None:
        s = np.asarray(skewness)[:, None]
        z = z*np.exp(np.where(z < 0, s, -s))
        correction = np.sum(np.logaddexp(s, -s)-np.log(2.))
    with np.errstate(divide='ignore'):
        log_radius = .5*np.log(np.sum(z*z, axis=0))
    constant = (np.log(power)+special.gammaln(d/2)-np.log(2)-d/2*np.log(np.pi)
                -d/power*np.log(q)-special.betaln(d/power, q)-np.log(chol.diagonal()).sum())
    return constant-correction-(q+d/power)*np.logaddexp(0., power*log_radius-np.log(q))


class JointGeneralizedT:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'], dtype=float)
        self.scatter = np.asarray(fit['scatter'], dtype=float)
        self.power, self.q = float(fit['power']), float(fit['q'])
        if self.power <= 0 or self.q <= 0 or not np.isfinite([self.power, self.q]).all():
            raise ValueError('Generalized t shapes must be finite and positive')
        self.chol = np.linalg.cholesky(self.scatter)
        self.skewness = np.asarray(fit.get('skewness', np.zeros(len(self.location))), dtype=float)
        if self.skewness.shape != self.location.shape or not np.isfinite(self.skewness).all():
            raise ValueError('Invalid generalized t skewness vector')

    def logpdf(self, x):
        values = logpdf(x, self.location, self.chol, self.power, self.q, self.skewness)
        return values[0] if np.asarray(x).ndim == 1 else values

    def pdf(self, x): return np.exp(self.logpdf(x))
    def mean(self):
        if self.power*self.q <= 1:
            return np.full_like(self.location, np.nan)
        return self.location+self.chol@self._latent_mean()

    def _latent_mean(self):
        d = len(self.location)
        absolute = radial_moment(d, self.power, self.q, 1)*np.exp(
            special.gammaln(d/2)-.5*np.log(np.pi)-special.gammaln((d+1)/2))
        return 2*np.sinh(self.skewness)*absolute

    def cov(self):
        if self.power*self.q <= 2:
            return None
        factor = covariance_factor(len(self.location), self.power, self.q)
        shift = 2*np.sinh(self.skewness)
        second = factor*(2/np.pi)*np.outer(shift, shift)
        np.fill_diagonal(second, factor*(1+4*np.sinh(self.skewness)**2))
        latent = second-np.outer(self._latent_mean(), self._latent_mean())
        return self.chol@latent@self.chol.T

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        d = len(self.location)
        direction = rng.normal(size=(size, d))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        if np.any(self.skewness):
            negative = rng.random((size, d)) < special.expit(-2*self.skewness)
            direction = np.abs(direction)*np.where(negative, -np.exp(-self.skewness), np.exp(self.skewness))
        radius = (self.q*rng.gamma(d/self.power, size=size)/rng.gamma(self.q, size=size))**(1/self.power)
        return self.location+(radius[:, None]*direction)@self.chol.T


def fit_generalized_t(x, location=None, max_iterations=2000, *, skewed=False):
    started = time.perf_counter()
    n, d = x.shape
    center, unit = x.mean(axis=0), x.std(axis=0)
    z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    nloc = d if fixed is None else 0
    indices = np.tril_indices(d)
    diagonal = np.flatnonzero(indices[0] == indices[1])
    nc = len(indices[0])

    def unpack(v):
        mu = v[:d] if fixed is None else fixed
        entries = v[nloc:nloc+nc].copy()
        entries[diagonal] = np.exp(entries[diagonal])
        chol = np.zeros((d, d))
        chol[indices] = entries
        shapes = nloc+nc
        return mu, chol, np.exp(v[shapes]), np.exp(v[shapes+1]), v[shapes+2:] if skewed else None

    def objective(v):
        with np.errstate(all='ignore'):
            value = -float(logpdf(z, *unpack(v)).sum())
        return value if np.isfinite(value) else 1e100

    bounds = [(None, None)]*(nloc+nc)
    for j in diagonal:
        bounds[nloc+j] = (-12., 12.)
    bounds += [(np.log(.25), np.log(10.)), (np.log(.1), np.log(1000.))]
    if skewed:
        bounds += [(-3., 3.)]*d
    mu0 = np.zeros(d) if fixed is None else fixed
    cov0 = (z-mu0).T@(z-mu0)/n
    attempts, candidates = [], []
    starts = [(p, q, s) for p, q in [(1., 4.), (2., 2.), (1., 30.), (2., 30.), (3., 2.)]
              for s in ([-.25, 0., .25] if skewed else [0.])]
    for p, q, s in starts:
        chol = np.linalg.cholesky(cov0/covariance_factor(d, p, q))
        entries = chol[indices]
        entries[diagonal] = np.log(entries[diagonal])
        initial = np.r_[mu0 if fixed is None else [], entries, np.log(p), np.log(q)]
        if skewed:
            initial = np.r_[initial, np.full(d, s)]
        result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iterations, 'maxfun': 200000, 'ftol': 1e-11, 'gtol': 1e-6})
        attempts.append(dict(start=f'power={p},q={q},skew={s}', converged=bool(result.success),
            iterations=int(result.nit), negative_loglik=float(result.fun), message=str(result.message)))
        if np.isfinite(result.fun) and result.fun < 1e99:
            candidates.append(result)
    if not candidates:
        raise ValueError('No finite joint generalized t fit')
    good = [r for r in candidates if r.success]
    result = min(good or candidates, key=lambda r: r.fun)
    mu, chol, p, q, skewness = unpack(result.x)
    mu = center+unit*mu
    scatter = (chol@chol.T)*np.outer(unit, unit)
    record = dict(location=mu.tolist(), scatter=scatter.tolist(), power=float(p), q=float(q))
    if skewed:
        record['skewness'] = skewness.tolist()
    dist = JointGeneralizedT(record)
    covariance = dist.cov()
    if p*q > 1 and not np.isfinite(dist.mean()).all():
        raise ValueError('Numerically nonfinite generalized t mean')
    if covariance is not None and not np.isfinite(covariance).all():
        raise ValueError('Numerically nonfinite generalized t covariance')
    corr = scatter/np.sqrt(np.outer(scatter.diagonal(), scatter.diagonal()))
    ll = -float(result.fun)-n*np.log(unit).sum()
    k = nloc+nc+2+(d if skewed else 0)
    boundary = any(lo is not None and (abs(v-lo) < 1e-4 or abs(v-hi) < 1e-4)
                   for v, (lo, hi) in zip(result.x, bounds))
    output = dict(model='generalized-t-skewed' if skewed else 'generalized-t',
        family='cholesky-two-piece-generalized-t' if skewed else 'elliptical-generalized-t', power=float(p), q=float(q),
        tail_index=float(p*q), observations=n, dimensions=d, parameters=k, loglik=float(ll),
        aic=float(2*k-2*ll), bic=float(np.log(n)*k-2*ll), location=mu.tolist(),
        mean=dist.mean().tolist() if p*q > 1 else None, scatter=scatter.tolist(),
        covariance=covariance.tolist() if covariance is not None else None,
        correlation=(covariance/np.sqrt(np.outer(covariance.diagonal(), covariance.diagonal()))).tolist() if covariance is not None else None, scatter_correlation=corr.tolist(),
        converged=bool(result.success), boundary=boundary,
        status='boundary' if result.success and boundary else ('ok' if result.success else 'not_converged'),
        message=str(result.message), attempts=attempts, fit_sec=time.perf_counter()-started)
    if skewed:
        output['skewness'] = skewness.tolist()
        output['skew_coordinates'] = 'Cholesky coordinates in supplied asset order; not asset marginal skewness'
    return output


def checked_integral(function, lower, upper, tail=.5):
    tolerance = max(1e-14, min(1e-10, tail*1e-7))
    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        try:
            value, error = integrate.quad(function, lower, upper, epsabs=tolerance, epsrel=1e-8, limit=250)
        except integrate.IntegrationWarning as exc:
            raise ValueError('Generalized t projection quadrature failed') from exc
    if not np.isfinite(value) or error > max(tolerance*10, abs(value)*1e-6):
        raise ValueError('Generalized t projection quadrature missed tolerance')
    return value


class ProjectedGeneralizedT:
    """One-dimensional projection, retaining the original radial dimension."""
    def __init__(self, location, scale, dimension, power, q):
        self.location, self.scale, self.dimension = location, scale, dimension
        self.power, self.q = power, q
        self.factor = covariance_factor(dimension, power, q)
        self.angular = np.exp(special.gammaln(dimension/2)-.5*np.log(np.pi)-special.gammaln((dimension-1)/2))

    def mean(self): return self.location if self.power*self.q > 1 else np.nan
    def var(self): return self.scale**2*self.factor if self.power*self.q > 1 else np.nan
    def std(self): return np.sqrt(self.var())

    def _tail(self, z, tolerance_tail=.5, moment=False):
        if np.isinf(z): return 0.
        if z == 0 and not moment: return .5
        d, p, q = self.dimension, self.power, self.q
        order = 1 if moment else 0
        factor = radial_moment(d, p, q, 1) if moment else 1.
        if not np.isfinite(factor): return np.inf
        def integrand(theta):
            cosine = np.cos(theta)
            with np.errstate(divide='ignore'):
                logratio = p*(np.log(z)-np.log(cosine))-np.log(q)
            # Small arguments on either side avoid beta-CDF cancellation.
            tail = (special.betainc(q-order/p, (d+order)/p, special.expit(-logratio)) if logratio > 0
                    else 1-special.betainc((d+order)/p, q-order/p, special.expit(logratio)))
            return self.angular*np.sin(theta)**(d-2)*cosine**order*tail
        return factor*checked_integral(integrand, 0, np.pi/2, tolerance_tail)

    def sf(self, x):
        def one(v):
            if np.isnan(v): return np.nan
            z = (v-self.location)/self.scale
            return self._tail(z) if z >= 0 else 1-self._tail(-z)
        return np.vectorize(one, otypes=[float])(x)

    def cdf(self, x): return self.sf(2*self.location-np.asarray(x))

    def ppf(self, probabilities):
        def one(prob):
            if not 0 <= prob <= 1: return np.nan
            if prob == 0: return -np.inf
            if prob == 1: return np.inf
            if prob == .5: return self.location
            tail = min(prob, 1-prob)
            high = 1.
            for _ in range(1024):
                if self._tail(high, tail) <= tail: break
                high *= 2
                if not np.isfinite(high): raise ValueError('Nonfinite generalized t quantile bracket')
            else: raise ValueError('Could not bracket generalized t quantile')
            z = optimize.brentq(lambda v: self._tail(v, tail)-tail, 0, high, xtol=1e-11, rtol=1e-11)
            return self.location+self.scale*z*(-1 if prob < .5 else 1)
        return np.vectorize(one, otypes=[float])(probabilities)

    def isf(self, p): return 2*self.location-self.ppf(p)

    def pdf(self, x):
        d, p, q = self.dimension, self.power, self.q
        constant = (np.log(p)+special.gammaln(d/2)-.5*np.log(np.pi)-special.gammaln((d-1)/2)
                    -d/p*np.log(q)-special.betaln(d/p, q))
        shift = (np.log(q)+np.log((d-1)/(p*q+1)))/p
        def one(v):
            z = abs((v-self.location)/self.scale)
            if np.isnan(z): return np.nan
            if np.isinf(z): return 0.
            if z == 0:
                return np.exp(constant+(d-1)/p*np.log(q)-np.log(p)+special.betaln((d-1)/p, q+1/p))/self.scale
            logz = np.log(z)
            offset = max(shift, logz)
            def integrand(t):
                logr = t+offset
                lognorm = .5*np.logaddexp(2*logz, 2*logr)
                value = constant+(d-1)*logr-(q+d/p)*np.logaddexp(0., p*lognorm-np.log(q))
                return np.exp(value)
            return checked_integral(integrand, -np.inf, np.inf)/self.scale
        return np.vectorize(one, otypes=[float])(x)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = np.sqrt(rng.beta(.5, (self.dimension-1)/2, size=size))*rng.choice([-1., 1.], size=size)
        radius = (self.q*rng.gamma(self.dimension/self.power, size=size)/rng.gamma(self.q, size=size))**(1/self.power)
        return self.location+self.scale*radius*u

    def risk(self, confidence):
        probability = 1-confidence
        quantile = float(self.ppf(probability))
        if self.power*self.q <= 1:
            return dict(var=-quantile, es=np.inf, es_status='infinite (power*q <= 1)', es_method='generalized-t-radial-partial-moment')
        moment = self._tail(abs((quantile-self.location)/self.scale), probability, moment=True)
        es = -self.location+self.scale*moment/probability
        if not np.isfinite(es): raise ValueError('Numerically nonfinite generalized t ES')
        return dict(var=-quantile, es=float(es), es_status='finite', es_method='generalized-t-radial-partial-moment')
