"""Audited fitting/risk support for SciPy Johnson SU and Crystal Ball.

Densities/CDFs/quantiles/sampling are supplied by SciPy. Crystal Ball moments
are evaluated separately to handle divergent moments and avoid large powers.
Definitions: scipy.stats.johnsonsu and scipy.stats.crystalball documentation.
"""
import numpy as np
from scipy import optimize, special, stats

EXTRA_MODELS = ('johnsonsu', 'johnson-su-symmetric', 'crystalball')


class CrystalBall(stats.rv_continuous):
    def _argcheck(self, beta, m):
        return (beta > 0) & (m > 1) & np.isfinite(beta) & np.isfinite(m)

    def _logpdf(self, x, beta, m): return stats.crystalball.logpdf(x, beta, m)
    def _pdf(self, x, beta, m): return stats.crystalball.pdf(x, beta, m)
    def _cdf(self, x, beta, m): return stats.crystalball.cdf(x, beta, m)
    def _sf(self, x, beta, m): return stats.crystalball.sf(x, beta, m)
    def _ppf(self, p, beta, m): return stats.crystalball.ppf(p, beta, m)
    def _isf(self, p, beta, m): return stats.crystalball.isf(p, beta, m)
    def _rvs(self, beta, m, size=None, random_state=None):
        return stats.crystalball.rvs(beta, m, size=size, random_state=random_state)

    def _stats(self, beta, m):
        e = np.exp(-beta**2/2)
        core = np.sqrt(2*np.pi)*special.ndtr(beta)
        tail = m/beta/(m-1)*e
        norm = 1/(core+tail)
        gaussian = [core, e]
        for r in range(2, 5):
            gaussian.append((-beta)**(r-1)*e+(r-1)*gaussian[r-2])
        raw = []
        with np.errstate(all='ignore'):
            for r in range(1, 5):
                # Conditional excess below -beta is Lomax(shape=m-1, scale=m/beta).
                total = beta**r
                excess_moment = np.ones_like(beta)
                for k in range(1, r+1):
                    excess_moment = excess_moment*k*(m/beta)/(m-1-k)
                    total = total+special.comb(r, k)*beta**(r-k)*excess_moment
                raw.append(np.where(m > r+1, norm*(gaussian[r]+(-1)**r*tail*total), np.nan))
            m1, m2, m3, m4 = raw
            var = m2-m1*m1
            skew = (m3-3*m1*m2+2*m1**3)/var**1.5
            kurt = (m4-4*m1*m3+6*m1*m1*m2-3*m1**4)/var**2-3
        return (np.where(m > 2, m1, -np.inf),
                np.where(m > 3, var, np.where(m > 2, np.inf, np.nan)),
                skew, np.where(m > 5, kurt, np.where(m > 3, np.inf, np.nan)))


crystal_ball = CrystalBall(name='crystalball', shapes='beta, m')


def fit_extra(x, model, location=None, max_iterations=10000):
    """Bounded multistart likelihood optimization on standardized observations."""
    johnson = model != 'crystalball'
    symmetric = model == 'johnson-su-symmetric'
    dist = stats.johnsonsu if johnson else stats.crystalball

    def unpack(v):
        if johnson:
            shapes = [0. if symmetric else v[0], np.exp(v[0] if symmetric else v[1])]
        else:
            shapes = [np.exp(v[0]), 1+np.exp(v[1])]
        return (*shapes, v[-2] if location is None else location, np.exp(v[-1]))

    def objective(v):
        value = -float(np.sum(dist.logpdf(x, *unpack(v))))
        return value if np.isfinite(value) else 1e100

    if johnson:
        bounds = ([] if symmetric else [(-10., 10.)])+[(np.log(.25), np.log(20.))]
        starts = [(a, b) for b in [.7, 1.5, 3.] for a in ([0.] if symmetric else [-1., 0., 1.])]
    else:
        bounds = [(np.log(.1), np.log(10.)), (np.log(.05), np.log(99.))]
        starts = [(1., 3.), (2., 3.), (1., 8.), (2., 8.), (3., 20.)]
    bounds += ([] if location is not None else [(None, None)])+[(-12., 12.)]
    sample_iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    candidates = []
    for a, b in starts:
        unit_iqr = dist.ppf(.75, a, b)-dist.ppf(.25, a, b)
        scale = (sample_iqr if sample_iqr > 0 else np.std(x))/unit_iqr
        loc = float(np.median(x)-scale*dist.median(a, b))
        initial = (([] if symmetric else [a])+[np.log(b)] if johnson
                   else [np.log(a), np.log(b-1)])
        initial += ([] if location is not None else [loc])+[np.clip(np.log(scale), -12, 12)]
        result = optimize.minimize(objective, initial, method='Nelder-Mead', bounds=bounds,
            options={'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-8})
        if np.isfinite(result.fun) and result.fun < 1e99:
            candidates.append(result)
    if not candidates:
        raise ValueError('No finite '+model+' fit')
    good = [r for r in candidates if r.success]
    result = min(good or candidates, key=lambda r: r.fun)
    result.boundary = any(lo is not None and (abs(v-lo) < 1e-4 or abs(v-hi) < 1e-4)
                          for v, (lo, hi) in zip(result.x, bounds))
    return unpack(result.x), result


def extra_es(distribution, probability, quantile):
    """Lower-tail ES; analytic partial moments, no finite-variance requirement."""
    johnson = distribution.dist.name == 'johnsonsu'
    keys = ['a', 'b', 'loc', 'scale'] if johnson else ['beta', 'm', 'loc', 'scale']
    values = dict(zip(keys, distribution.args))
    values.update(distribution.kwds)
    loc, scale = values.get('loc', 0.), values.get('scale', 1.)
    if johnson:
        a, b = values['a'], values['b']
        z = special.ndtri(probability)
        plus = -a/b+.5/b**2+special.log_ndtr(z-1/b)
        minus = a/b+.5/b**2+special.log_ndtr(z+1/b)
        # Divide by tail probability in log space before subtracting.
        plus -= np.log(probability)
        minus -= np.log(probability)
        conditional = (.5*np.exp(plus)*(-np.expm1(minus-plus)) if plus >= minus
                       else .5*np.exp(minus)*np.expm1(plus-minus))
        return -loc-scale*conditional
    beta, m = values['beta'], values['m']
    if m <= 2:
        return np.inf
    z = (quantile-loc)/scale
    if z <= -beta:
        conditional = z-(m/beta-beta-z)/(m-2)
    else:
        e = np.exp(-beta**2/2)
        tail = m/beta/(m-1)*e
        norm = 1/(tail+np.sqrt(2*np.pi)*special.ndtr(beta))
        partial = norm*(tail*(-beta-m/beta/(m-2))+e-np.exp(-z*z/2))
        conditional = partial/probability
    return -loc-scale*conditional
