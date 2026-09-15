"""McDonald--Newey generalized t and its Fernandez--Steel two-piece extension.

Base density: p/[2*q**(1/p)*B(1/p,q)] * (1+abs(x)**p/q)**(-q-1/p).
Original reference: McDonald and Newey (1988), Econometric Theory 4, 428--457.
All routines are implemented with NumPy/SciPy, not copied from another package.
"""
import numpy as np
from scipy import optimize, special, stats

GT_MODELS = ('generalized-t', 'generalized-t-skewed')


def log_ratio(x, power, q, skewness):
    """Log of |two-piece standardized x|**power/q, including x=0."""
    with np.errstate(divide='ignore'):
        return power*(np.log(np.abs(x))+np.where(x < 0, skewness, -skewness))-np.log(q)


class GeneralizedT(stats.rv_continuous):
    def _argcheck(self, power, q, skewness):
        return (power > 0) & (q > 0) & np.isfinite(power) & np.isfinite(q) & np.isfinite(skewness) & (np.abs(skewness) < 100)

    def _logpdf(self, x, power, q, skewness):
        return (np.log(power)-np.log(q)/power-special.betaln(1/power, q)
                -np.logaddexp(skewness, -skewness)
                -(q+1/power)*np.logaddexp(0., log_ratio(x, power, q, skewness)))

    def _pdf(self, x, power, q, skewness):
        return np.exp(self._logpdf(x, power, q, skewness))

    def _cdf(self, x, power, q, skewness):
        lr = log_ratio(x, power, q, skewness)
        left, right = special.expit(-2*skewness), special.expit(2*skewness)
        central = special.betainc(1/power, q, special.expit(lr))
        tail = special.betainc(q, 1/power, special.expit(-lr))
        # Use the small beta argument directly near both the mode and tails.
        return np.where(x < 0, left*np.where(lr <= 0, 1-central, tail),
                        left+right*np.where(lr <= 0, central, 1-tail))

    def _sf(self, x, power, q, skewness):
        return self._cdf(-x, power, q, -skewness)

    def _ppf(self, prob, power, q, skewness):
        prob, power, q, skewness = np.broadcast_arrays(prob, power, q, skewness)
        left, right = special.expit(-2*skewness), special.expit(2*skewness)
        negative = prob < left
        tail = np.where(negative, prob/left, (1-prob)/right)
        tail = np.clip(tail, 0., 1.)
        v = special.betaincinv(q, 1/power, tail)
        u = special.betaincinv(1/power, q, 1-tail)
        with np.errstate(divide='ignore', over='ignore'):
            log_odds = np.where(tail <= .5, np.log1p(-v)-np.log(v), np.log(u)-np.log1p(-u))
            magnitude = np.exp((np.log(q)+log_odds)/power)
        return np.where(negative, -np.exp(-skewness)*magnitude, np.exp(skewness)*magnitude)

    def _isf(self, prob, power, q, skewness):
        return -self._ppf(prob, power, q, -skewness)

    def _stats(self, power, q, skewness):
        tail_index = power*q
        raw = []
        with np.errstate(all='ignore'):
            for r in range(1, 5):
                absolute = np.exp(r*np.log(q)/power+special.betaln((r+1)/power, q-r/power)-special.betaln(1/power, q))
                factor = (np.exp((r+1)*skewness)+(-1)**r*np.exp(-(r+1)*skewness))/(2*np.cosh(skewness))
                raw.append(np.where(tail_index > r, absolute*factor, np.nan))
            m1, m2, m3, m4 = raw
            variance = m2-m1*m1
            skew = (m3-3*m1*m2+2*m1**3)/variance**1.5
            kurt = (m4-4*m1*m3+6*m1*m1*m2-3*m1**4)/variance**2-3
        return (m1, np.where(tail_index > 2, variance, np.where(tail_index > 1, np.inf, np.nan)),
                skew, np.where(tail_index > 4, kurt, np.where(tail_index > 2, np.inf, np.nan)))

    def _rvs(self, power, q, skewness, size=None, random_state=None):
        positive = random_state.uniform(size=size) < special.expit(2*skewness)
        magnitude = (q*random_state.gamma(1/power, size=size)/random_state.gamma(q, size=size))**(1/power)
        return np.where(positive, np.exp(skewness)*magnitude, -np.exp(-skewness)*magnitude)


generalized_t = GeneralizedT(name='generalized_t', shapes='power, q, skewness')


def lower_first_moment(x, power, q, skewness):
    """E[X 1(X<=x)] in base location/scale units; requires power*q>1."""
    if power*q <= 1:
        return -np.inf
    log_m = np.log(q)/power+special.betaln(2/power, q-1/power)-special.betaln(1/power, q)
    lr = log_ratio(x, power, q, skewness)
    normalizer = np.logaddexp(skewness, -skewness)
    if x < 0:
        tail = (1-special.betainc(2/power, q-1/power, special.expit(lr)) if lr <= 0
                else special.betainc(q-1/power, 2/power, special.expit(-lr)))
        return -np.exp(log_m-2*skewness-normalizer)*tail
    return (np.exp(log_m+2*skewness-normalizer)*special.betainc(2/power, q-1/power, special.expit(lr))
            -np.exp(log_m-2*skewness-normalizer))


def fit_generalized_t(x, symmetric=True, location=None, max_iterations=10000):
    """Bounded multistart MLE; choose best converged finite attempt."""
    def unpack(v):
        return np.exp(v[0]), np.exp(v[1]), 0. if symmetric else v[2], v[-2] if location is None else location, np.exp(v[-1])

    def objective(v):
        p, q, skew, loc, scale = unpack(v)
        value = -float(np.sum(generalized_t._logpdf((x-loc)/scale, p, q, skew)-np.log(scale)))
        return value if np.isfinite(value) else 1e100

    bounds = [(np.log(.2), np.log(10.)), (np.log(.1), np.log(1000.))]
    bounds += ([] if symmetric else [(-3., 3.)])
    bounds += ([] if location is not None else [(None, None)]) + [(-12., 12.)]
    candidates = []
    for p, q in [(1., 4.), (2., 2.), (1., 30.), (2., 30.), (3., 2.)]:
        for skew in ([0.] if symmetric else [-.4, 0., .4]):
            mean, variance = generalized_t.stats(p, q, skew, moments='mv')
            scale = np.std(x)/np.sqrt(variance)
            loc = float(np.mean(x)-scale*mean)
            initial = [np.log(p), np.log(q)] + ([] if symmetric else [skew])
            initial += ([] if location is not None else [loc]) + [np.clip(np.log(scale), -12., 12.)]
            result = optimize.minimize(objective, initial, method='Nelder-Mead', bounds=bounds,
                options={'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-8})
            if np.isfinite(result.fun) and result.fun < 1e99:
                candidates.append(result)
    if not candidates:
        raise ValueError('No finite generalized t fit')
    good = [r for r in candidates if r.success]
    result = min(good or candidates, key=lambda r: r.fun)
    result.boundary = any(lo is not None and (abs(v-lo) < 1e-4 or abs(v-hi) < 1e-4)
                          for v, (lo, hi) in zip(result.x, bounds))
    return unpack(result.x), result
