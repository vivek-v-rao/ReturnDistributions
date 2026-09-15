"""Fernandez--Steel two-piece GED, using the project's FS-t convention.

For z=(x-loc)/scale, density is 2*g(z*exp(sign(-z)*skewness)) /
(exp(skewness)+exp(-skewness))/scale, where g is scipy's gennorm.
This is a location/scale parameterization, NOT mean/standard deviation.
See https://search.r-project.org/CRAN/refmans/tsdistributions/html/sged.html
for the same family under a standardized parameterization. No external code copied.
"""
import numpy as np
from scipy import optimize, special, stats


class FSSkewGED(stats.rv_continuous):
    def _argcheck(self, power, skewness):
        return (power > 0) & np.isfinite(power) & np.isfinite(skewness) & (np.abs(skewness) < 100)

    def _logpdf(self, x, power, skewness):
        z = x*np.exp(np.where(x < 0, skewness, -skewness))
        with np.errstate(over='ignore'):
            return np.log(power)-special.gammaln(1/power)-np.logaddexp(skewness, -skewness)-np.abs(z)**power

    def _pdf(self, x, power, skewness):
        return np.exp(self._logpdf(x, power, skewness))

    def _cdf(self, x, power, skewness):
        left, right = special.expit(-2*skewness), special.expit(2*skewness)
        z = x*np.exp(np.where(x < 0, skewness, -skewness))
        with np.errstate(over='ignore'):
            cutoff = np.abs(z)**power
        tail = special.gammaincc(1/power, cutoff)
        return np.where(x < 0, left*tail, left+right*special.gammainc(1/power, cutoff))

    def _sf(self, x, power, skewness):
        return self._cdf(-x, power, -skewness)

    def _ppf(self, q, power, skewness):
        q, power, skewness = np.broadcast_arrays(q, power, skewness)
        left, right = special.expit(-2*skewness), special.expit(2*skewness)
        mask = q < left
        result = np.empty_like(q)
        result[mask] = -np.exp(-skewness[mask])*special.gammainccinv(1/power[mask], q[mask]/left[mask])**(1/power[mask])
        result[~mask] = np.exp(skewness[~mask])*special.gammainccinv(1/power[~mask], np.clip((1-q[~mask])/right[~mask], 0, 1))**(1/power[~mask])
        return result

    def _isf(self, q, power, skewness):
        return -self._ppf(q, power, -skewness)

    def _stats(self, power, skewness):
        raw = []
        for r in range(1, 5):
            absolute = np.exp(special.gammaln((r+1)/power)-special.gammaln(1/power))
            factor = (np.exp((r+1)*skewness)+(-1)**r*np.exp(-(r+1)*skewness))/(2*np.cosh(skewness))
            raw.append(absolute*factor)
        m1, m2, m3, m4 = raw
        variance = m2-m1*m1
        return m1, variance, (m3-3*m1*m2+2*m1**3)/variance**1.5, (m4-4*m1*m3+6*m1*m1*m2-3*m1**4)/variance**2-3

    def _rvs(self, power, skewness, size=None, random_state=None):
        positive = random_state.uniform(size=size) < special.expit(2*skewness)
        magnitude = random_state.gamma(1/power, size=size)**(1/power)
        return np.where(positive, np.exp(skewness)*magnitude, -np.exp(-skewness)*magnitude)


skew_ged = FSSkewGED(name='fs_skew_ged', shapes='power, skewness')


def fit_skew_ged(x, location=None, max_iterations=10000, *, fixed_power=None):
    """Bounded multistart MLE on standardized input; NM handles the mode cusp."""
    if fixed_power is not None and (not np.isfinite(fixed_power) or fixed_power <= 0):
        raise ValueError('Fixed power must be finite and positive')
    def unpack(v):
        return (np.exp(v[0]) if fixed_power is None else fixed_power,
                v[1] if fixed_power is None else v[0],
                v[-2] if location is None else location, np.exp(v[-1]))

    def objective(v):
        power, skew, loc, scale = unpack(v)
        value = -float(np.sum(skew_ged._logpdf((x-loc)/scale, power, skew)-np.log(scale)))
        return value if np.isfinite(value) else 1e100

    bounds = ([] if fixed_power is not None else [(np.log(.1), np.log(10.))]) + [(-3., 3.)]
    bounds += ([] if location is not None else [(None, None)]) + [(-12., 12.)]
    candidates = []
    for power in ((.75, 1.5, 3.) if fixed_power is None else [fixed_power]):
        for skew in (-.5, 0., .5):
            mean, variance = skew_ged.stats(power, skew, moments='mv')
            scale = np.std(x)/np.sqrt(variance)
            loc = float(np.mean(x)-scale*mean)
            start = ([] if fixed_power is not None else [np.log(power)]) + [skew]
            start += ([] if location is not None else [loc]) + [np.clip(np.log(scale), -12., 12.)]
            result = optimize.minimize(objective, start, method='Nelder-Mead', bounds=bounds,
                options={'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-8})
            if np.isfinite(result.fun) and result.fun < 1e99:
                candidates.append(result)
    if not candidates:
        raise ValueError('No finite skewed GED fit')
    good = [r for r in candidates if r.success]
    result = min(good or candidates, key=lambda r: r.fun)
    result.boundary = any(lo is not None and (abs(v-lo) < 1e-4 or abs(v-hi) < 1e-4)
                          for v, (lo, hi) in zip(result.x, bounds))
    return unpack(result.x), result
