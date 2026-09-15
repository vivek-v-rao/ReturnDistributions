"""Constrained finite normal/t/GED/NIG mixtures with shared tail shape.

EM/ECM uses latent normal precisions for Student t; GED/NIG use direct optimization. Eigenvalue constraints
apply to scatter after scaling each asset by its sample standard deviation.
These are finite mixtures, not the continuous mixtures used to define GH/t.
"""
import time
import numpy as np
from scipy import linalg, optimize, special, stats
from .joint_power import JointPower, covariance_factor
from .joint_gh import JointGH

FINITE_MIXTURE_MODELS = ('normal', 'student-t', 'ged', 'nig-skewed')


def _weights(counts, floor):
    # Exact multinomial M step on the simplex with a lower bound.
    free = np.ones(len(counts), dtype=bool)
    out = np.full(len(counts), floor)
    while free.any():
        trial = counts[free]/counts[free].sum()*(1-floor*(~free).sum())
        if np.all(trial >= floor):
            out[free] = trial
            break
        free[np.flatnonzero(free)[trial < floor]] = False
    return out


def fit_finite_mixture(data, model, components=2, *, baseline=None, location=None,
                       max_iterations=2000, starts=5, seed=12345,
                       min_weight=.01, eigen_floor=1e-4):
    started = time.perf_counter()
    x = np.asarray(data, dtype=float)
    if model not in FINITE_MIXTURE_MODELS or not 2 <= components <= 8:
        raise ValueError('Finite mixtures support normal/student-t/GED/skewed NIG and 2--8 components')
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Require finite complete-case joint returns')
    n, d = x.shape
    if n < max(8, components*(d+2)) or d < 1:
        raise ValueError('Too few observations for requested mixture size')
    if (starts < 2 or max_iterations < 1 or seed < 0 or not np.isfinite(min_weight)
            or not 0 < min_weight < 1/components or not np.isfinite(eigen_floor) or eigen_floor <= 0):
        raise ValueError('Invalid mixture starts, iterations, seed, weight or eigenvalue floor')
    center, unit = x.mean(0), x.std(0)
    if np.any(unit <= 0) or not np.isfinite(unit).all():
        raise ValueError('Constant or invalid asset')
    z = (x-center)/unit
    if np.linalg.matrix_rank(z) < d:
        raise ValueError('Rank-deficient returns')
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    if fixed is not None and not np.isfinite(fixed).all():
        raise ValueError('Invalid fixed location')
    if model in ('ged', 'nig-skewed'):
        from .joint_ged_mixture import fit_ged_mixture
        return fit_ged_mixture(x, components, baseline=baseline, location=location,
                               max_iterations=max_iterations, starts=starts, seed=seed,
                               min_weight=min_weight, eigen_floor=eigen_floor, model=model)

    def clip(matrix):
        values, vectors = np.linalg.eigh((matrix+matrix.T)/2)
        return (vectors*np.maximum(values, eigen_floor))@vectors.T

    def expectation(weights, means, scatters, df):
        logdens, precision, logprecision = [], [], []
        for mu, scatter in zip(means, scatters):
            chol = np.linalg.cholesky(scatter)
            residual = linalg.solve_triangular(chol, (z-mu).T, lower=True, check_finite=False)
            q = (residual**2).sum(0)
            logdet = np.log(chol.diagonal()).sum()
            if df is None:
                logdens.append(-d/2*np.log(2*np.pi)-logdet-q/2)
                precision.append(np.ones(n))
            else:
                logdens.append(special.gammaln((df+d)/2)-special.gammaln(df/2)
                               -d/2*np.log(df*np.pi)-logdet-(df+d)/2*np.log1p(q/df))
                precision.append((df+d)/(df+q))
                logprecision.append(special.digamma((df+d)/2)-np.log((df+q)/2))
        joint = np.array(logdens).T+np.log(weights)
        total = special.logsumexp(joint, axis=1)
        tau = np.maximum(np.exp(joint-total[:, None]), np.finfo(float).tiny)
        tau /= tau.sum(1)[:, None]
        return float(total.sum()), tau, np.array(precision).T, np.array(logprecision).T

    rng = np.random.default_rng(seed)
    base_mu = np.zeros(d) if fixed is None else fixed
    base_scatter = (z-base_mu).T@(z-base_mu)/n
    base_df = 8. if model == 'student-t' else None
    if baseline and 'scatter' in baseline:
        base_mu = (np.asarray(baseline['location'])-center)/unit
        base_scatter = np.asarray(baseline['scatter'])/np.outer(unit, unit)
        if model == 'student-t': base_df = float(np.clip(baseline['df'], .25, 200.))
    base_scatter = clip(base_scatter)
    attempts, candidates = [], []
    for start in range(starts):
        weights = np.full(components, 1/components)
        df = base_df
        # First start retains the nested baseline. Others split observations
        # by radius, a principal axis, or seeded random directions.
        if start == 0:
            means = np.tile(base_mu, (components, 1))
            scatters = np.tile(base_scatter, (components, 1, 1))
        else:
            if start == 1:
                score = ((z-base_mu)**2).sum(1)
            elif start == 2:
                score = z@np.linalg.eigh(base_scatter)[1][:, -1]
            else:
                score = z@rng.normal(size=d)
            groups = np.array_split(np.argsort(score), components)
            means = np.array([z[g].mean(0) if fixed is None else fixed for g in groups])
            scatters = np.array([clip((z[g]-mu).T@(z[g]-mu)/len(g)) for g, mu in zip(groups, means)])
        ll, tau, u, logu = expectation(weights, means, scatters, df)
        converged = False
        for iteration in range(max_iterations):
            counts = tau.sum(0)
            weights_new = _weights(counts, min_weight)
            weighted = tau*u
            means_new = weighted.T@z/weighted.sum(0)[:, None] if fixed is None else np.tile(fixed, (components, 1))
            scatters_new = np.array([clip(((z-mu)*weighted[:, j, None]).T@(z-mu)/counts[j])
                                     for j, mu in enumerate(means_new)])
            df_new = df
            if df is not None:
                c = float((tau*(logu-u)).sum()/n)
                result = optimize.minimize_scalar(lambda v: -(v/2*np.log(v/2)-special.gammaln(v/2)+v/2*c),
                                                 bounds=(.25, 200.), method='bounded', options={'xatol': 1e-7})
                df_new = float(result.x)
            new_ll, new_tau, new_u, new_logu = expectation(weights_new, means_new, scatters_new, df_new)
            if not np.isfinite(new_ll) or new_ll < ll-1e-7*max(n, abs(ll)):
                break
            improvement = new_ll-ll
            weights, means, scatters, df = weights_new, means_new, scatters_new, df_new
            ll, tau, u, logu = new_ll, new_tau, new_u, new_logu
            if abs(improvement)/n < 1e-8:
                converged = True
                break
        attempts.append(dict(start=start, loglik=float(ll-n*np.log(unit).sum()),
                             converged=converged, iterations=iteration+1))
        candidates.append((ll, converged, weights.copy(), means.copy(), scatters.copy(), df))
    # Never prefer a lower-likelihood start merely because it stopped earlier.
    ll, converged, weights, means, scatters, df = max(candidates, key=lambda r: r[0])
    boundary = (np.any(weights <= min_weight+1e-6)
                or np.min(np.linalg.eigvalsh(scatters)) <= eigen_floor*(1+1e-5)
                or (df is not None and (df <= .25001 or df >= 199.999)))
    means = center+means*unit
    scatters = scatters*np.outer(unit, unit)
    # Sorting is only a reproducible label convention, not another fit.
    order = np.argsort(-weights, kind='stable')
    weights, means, scatters = weights[order], means[order], scatters[order]
    ll -= n*np.log(unit).sum()
    k = components*(d*(d+1)//2+(d if fixed is None else 0))+components-1+int(df is not None)
    output = dict(model=model, components=components, family='finite-mixture',
                  mixture_weights=weights.tolist(), component_fits=[dict(model=model, location=mu.tolist(), scatter=s.tolist(), df=df)
                      for mu, s in zip(means, scatters)], df=df,
                  observations=n, dimensions=d, parameters=k, loglik=float(ll),
                  aic=float(2*k-2*ll), bic=float(np.log(n)*k-2*ll),
                  converged=converged, boundary=bool(boundary),
                  status=('boundary' if boundary else 'ok') if converged else 'not_converged',
                  attempts=attempts, mixture_min_weight=min_weight, mixture_eigen_floor=eigen_floor,
                  mixture_scaling=unit.tolist(), mixture_seed=seed, mixture_starts=starts,
                  fit_restriction='Scatter eigenvalue floor in sample-SD-scaled coordinates; weight floor; t shared df in [0.25,200]',
                  fit_sec=time.perf_counter()-started)
    dist = JointFiniteMixture(output)
    output['mean'] = dist.mean().tolist() if df is None or df > 1 else None
    output['covariance'] = dist.cov().tolist() if df is None or df > 2 else None
    return output


class JointFiniteMixture:
    def __init__(self, fit):
        self.weights = np.asarray(fit['mixture_weights'], dtype=float)
        self.records = fit['component_fits']
        if (len(self.weights) != len(self.records) or np.any(self.weights <= 0)
                or not np.isfinite(self.weights).all() or not np.isclose(self.weights.sum(), 1.)):
            raise ValueError('Invalid finite mixture weights')
        if (not self.records or fit['model'] not in FINITE_MIXTURE_MODELS
                or any(r['model'] != fit['model'] or r['df'] != fit.get('df') for r in self.records)
                or len({len(r['location']) for r in self.records}) != 1):
            raise ValueError('Mixtures require matching component families/dimensions and shared df')
        if fit['model'] == 'ged' and any(r.get('power') != fit.get('power') for r in self.records):
            raise ValueError('GED mixture components must share power')
        if fit['model'] == 'nig-skewed':
            if (not np.isfinite(fit.get('psi', np.nan)) or fit['psi'] <= 0 or fit.get('chi') != 1. or fit.get('lambda') != -.5
                    or any(r.get('psi') != fit['psi'] or r.get('chi') != 1. or r.get('lambda') != -.5 for r in self.records)):
                raise ValueError('NIG mixture components require chi=1, lambda=-0.5 and shared positive psi')
        self.distributions = [stats.multivariate_normal(mean=r['location'], cov=r['scatter']) if r['model'] == 'normal'
                              else (JointPower(r) if r['model'] == 'ged' else (JointGH(r) if r['model'] == 'nig-skewed' else stats.multivariate_t(loc=r['location'], shape=r['scatter'], df=r['df']))) for r in self.records]

    def logpdf(self, x):
        values = np.array([np.atleast_1d(dist.logpdf(x)) for dist in self.distributions])
        result = special.logsumexp(values+np.log(self.weights)[:, None], axis=0)
        return result[0] if np.asarray(x).ndim == 1 else result

    def pdf(self, x): return np.exp(self.logpdf(x))

    def mean(self):
        if any(r['df'] is not None and r['df'] <= 1 for r in self.records):
            return np.full(len(self.records[0]['location']), np.nan)
        return self.weights@np.array([f.mean() if r['model'] == 'nig-skewed' else r['location'] for r, f in zip(self.records, self.distributions)])

    def cov(self):
        mean = self.mean()
        if any(r['df'] is not None and r['df'] <= 2 for r in self.records):
            return np.full((len(mean), len(mean)), np.nan)
        if self.records[0]['model'] == 'nig-skewed':
            return sum(w*(f.cov()+np.outer(f.mean()-mean, f.mean()-mean)) for w, f in zip(self.weights, self.distributions))
        return sum(w*(np.array(r['scatter'])*(covariance_factor(len(mean), r['power']) if r['model'] == 'ged' else (1 if r['df'] is None else r['df']/(r['df']-2)))
                      +np.outer(np.array(r['location'])-mean, np.array(r['location'])-mean)) for w, r in zip(self.weights, self.records))

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        labels = rng.choice(len(self.weights), size=size, p=self.weights)
        out = np.empty((size, len(self.records[0]['location'])))
        for j, dist in enumerate(self.distributions):
            count = np.count_nonzero(labels == j)
            if count: out[labels == j] = np.asarray(dist.rvs(size=count, random_state=rng)).reshape(count, -1)
        return out


class ProjectedFiniteMixture:
    def __init__(self, weights, locations, scales, df=None, distributions=None):
        self.weights, self.locations, self.scales = map(np.asarray, (weights, locations, scales))
        self.df = df
        self.generic_components = distributions is not None
        self.distributions = distributions if distributions is not None else [stats.norm(loc=m, scale=s) if df is None else stats.t(df, loc=m, scale=s)
                                                                            for m, s in zip(locations, scales)]

    def pdf(self, x): return sum(w*f.pdf(x) for w, f in zip(self.weights, self.distributions))
    def cdf(self, x): return sum(w*f.cdf(x) for w, f in zip(self.weights, self.distributions))
    def sf(self, x): return sum(w*f.sf(x) for w, f in zip(self.weights, self.distributions))
    def mean(self):
        if self.generic_components: return float(sum(w*f.mean() for w, f in zip(self.weights, self.distributions)))
        return float(self.weights@self.locations) if self.df is None or self.df > 1 else np.nan
    def var(self):
        if self.generic_components:
            return float(sum(w*(f.var()+(f.mean()-self.mean())**2) for w, f in zip(self.weights, self.distributions)))
        if self.df is not None and self.df <= 2: return np.inf
        return float(self.weights@(self.scales**2*(1 if self.df is None else self.df/(self.df-2))+(self.locations-self.mean())**2))
    def std(self): return np.sqrt(self.var())
    def ppf(self, q):
        def one(p):
            if p == 0: return -np.inf
            if p == 1: return np.inf
            if not 0 < p < 1: return np.nan
            endpoints = [f.ppf(p) for f in self.distributions]
            lo, hi = min(endpoints), max(endpoints)
            if lo == hi: return lo
            return optimize.brentq(lambda v: self.cdf(v)-p, lo, hi, xtol=1e-14)
        return np.vectorize(one, otypes=[float])(q)
    def risk(self, confidence):
        p = 1-confidence
        quantile = float(self.ppf(p))
        if self.generic_components:
            from .tail_risk import checked_quad
            shortfall = 0.
            for w, f in zip(self.weights, self.distributions):
                scale = float(f.std())
                shortfall += w*scale*checked_quad(lambda t: t*scale*f.pdf(quantile-scale*t), 0, np.inf, p)
            return dict(var=-quantile, es=-quantile+shortfall/p, es_status='finite', es_method='finite-mixture-tail-quadrature')
        if self.df is not None and self.df <= 1:
            return dict(var=-quantile, es=np.inf, es_status='infinite (df <= 1)', es_method='finite-mixture-partial-moment')
        z = (quantile-self.locations)/self.scales
        if self.df is None:
            partial = self.locations*stats.norm.cdf(z)-self.scales*stats.norm.pdf(z)
        else:
            partial = self.locations*stats.t.cdf(z, self.df)-self.scales*(self.df+z*z)/(self.df-1)*stats.t.pdf(z, self.df)
        return dict(var=-quantile, es=-float(self.weights@partial)/p, es_status='finite', es_method='finite-mixture-partial-moment')
