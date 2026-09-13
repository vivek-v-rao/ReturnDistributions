"""Experimental Sahu--Dey--Branco skew-normal/t, with diagonal skew loadings.

X=mu+(diag(delta)|U|+Z)/sqrt(W/nu), U~N(0,I), Z~N(0,Sigma).
All latent variables are independent; omit the denominator for skew-normal.
Density: Sahu, Dey & Branco (2003), Canadian J. Statistics 31, 129--150.
"""
from functools import lru_cache
import time
import numpy as np
from scipy import linalg, optimize, special, stats

SDB_MODELS = ('sdb-skew-normal', 'sdb-skew-t')


def validate_controls(points, seed):
    if not isinstance(points, (int, np.integer)) or points < 64 or points > 65536 or points & (points-1):
        raise ValueError('SDB CDF points must be a power of two from 64 to 65536')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('SDB seed must be a nonnegative integer')


@lru_cache(maxsize=24)
def uniforms(d, points, seed):
    return stats.qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.log2(points)))


def orthant_logcdf(thresholds, covariance, df=None, points=512, seed=12345):
    """Fixed scrambled Sobol conditional-normal integration, not fresh MC per call.

Student-t uses its common chi-square mixture; sequential conditional-normal
integration leaves d-1 normal uniforms plus one mixture uniform. Log arithmetic
avoids flooring small orthant probabilities. Input rows are processed in chunks.
"""
    h = np.atleast_2d(thresholds)
    d = h.shape[1]
    chol = np.linalg.cholesky(covariance)
    if d == 1:
        z = h[:, 0]/chol[0, 0]
        return special.log_ndtr(z) if df is None else stats.t.logcdf(z, df)
    if df is None and np.allclose(covariance, np.diag(np.diag(covariance)), rtol=0, atol=1e-14):
        return special.log_ndtr(h/np.sqrt(covariance.diagonal())).sum(axis=1)
    u = uniforms(d, points, seed)
    scale = np.ones(points) if df is None else np.sqrt(2*special.gammaincinv(df/2, u[:, -1])/df)
    output = np.empty(len(h))
    for start in range(0, len(h), 64):
        batch = h[start:start+64]
        latent = np.zeros((len(batch), points, d))
        lp = np.zeros((len(batch), points))
        for j in range(d):
            shift = np.sum(latent[:, :, :j]*chol[j, :j], axis=2)
            bound = (batch[:, j, None]*scale-shift)/chol[j, j]
            part = special.log_ndtr(bound)
            lp += part
            if j < d-1:
                latent[:, :, j] = special.ndtri_exp(np.log(u[:, j])+part)
        output[start:start+len(batch)] = special.logsumexp(lp, axis=1)-np.log(points)
    return output


def logdensity(x, mu, sigma, delta, df=None, points=512, seed=12345):
    x = np.atleast_2d(x); d = len(mu)
    omega = sigma+np.diag(delta**2)
    chol = np.linalg.cholesky(omega)
    residual = x-mu
    standardized = linalg.solve_triangular(chol, residual.T, lower=True)
    q = np.sum(standardized**2, axis=0)
    if df is None:
        base = -d/2*np.log(2*np.pi)-np.log(chol.diagonal()).sum()-q/2
    else:
        base = (special.gammaln((df+d)/2)-special.gammaln(df/2)-d/2*np.log(df*np.pi)
                -np.log(chol.diagonal()).sum()-(df+d)/2*np.log1p(q/df))
    if not np.any(delta): return base
    load = np.diag(delta)
    solved = linalg.cho_solve((chol, True), load)
    conditional = np.eye(d)-load@solved
    thresholds = residual@solved
    if df is not None: thresholds *= np.sqrt((df+d)/(df+q))[:, None]
    return base+d*np.log(2)+orthant_logcdf(thresholds, conditional, None if df is None else df+d, points, seed)


class JointSDB:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'], dtype=float)
        self.scatter = np.asarray(fit['scatter'], dtype=float)  # Gaussian residual Sigma
        self.delta = np.asarray(fit['delta'], dtype=float)
        self.df = None if fit['model'] == 'sdb-skew-normal' else float(fit['df'])
        self.points = fit.get('cdf_points', 512); self.seed = fit.get('cdf_seed', 12345)
        validate_controls(self.points, self.seed)
        d = len(self.location)
        if (d < 1 or d > 5 or self.scatter.shape != (d, d) or self.delta.shape != (d,)
                or not np.isfinite(self.location).all() or not np.isfinite(self.scatter).all()
                or not np.isfinite(self.delta).all() or not np.allclose(self.scatter, self.scatter.T)
                or (self.df is not None and (not np.isfinite(self.df) or self.df <= 0))):
            raise ValueError('Invalid SDB parameters; experimental implementation supports 1--5 assets')
        self.chol = np.linalg.cholesky(self.scatter)

    def logpdf(self, x):
        out = logdensity(x, self.location, self.scatter, self.delta, self.df, self.points, self.seed)
        return out[0] if np.asarray(x).ndim == 1 else out

    def pdf(self, x): return np.exp(self.logpdf(x))

    def mean(self):
        if self.df is not None and self.df <= 1: return np.full_like(self.location, np.nan)
        b = np.sqrt(2/np.pi) if self.df is None else np.sqrt(self.df/np.pi)/special.poch((self.df-1)/2, .5)
        return self.location+b*self.delta

    def cov(self):
        if self.df is not None and self.df <= 2: return np.full_like(self.scatter, np.nan)
        factor = 1. if self.df is None else self.df/(self.df-2)
        numerator = self.scatter+(1-2/np.pi)*np.diag(self.delta**2)+2/np.pi*np.outer(self.delta, self.delta)
        shift = self.mean()-self.location
        return factor*numerator-np.outer(shift, shift)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        z = rng.normal(size=(size, len(self.location)))@self.chol.T
        z += np.abs(rng.normal(size=z.shape))*self.delta
        if self.df is not None: z /= np.sqrt(rng.chisquare(self.df, size=size)/self.df)[:, None]
        return self.location+z


def fit_sdb(x, model, location=None, max_iterations=2000, points=512, seed=12345):
    from .multivariate import fit_joint
    validate_controls(points, seed)
    if points > 16384: raise ValueError('SDB fitting points must be <=16384 to permit the 4x accuracy audit')
    started = time.perf_counter(); n, d = x.shape
    if d > 5: raise ValueError('Experimental SDB fitting supports at most 5 assets')
    normal = model == 'sdb-skew-normal'
    center, unit = x.mean(0), x.std(0); z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    nloc = d if fixed is None else 0
    ix = np.tril_indices(d); diag = np.flatnonzero(ix[0] == ix[1]); nc = len(ix[0])
    nested = fit_joint(z, 'normal' if normal else 'student-t', location=fixed, max_iterations=max_iterations)
    df0 = None if normal else np.clip(nested['df'], .3, 200.)
    base_sigma = np.asarray(nested['scatter']); mu0 = np.asarray(nested['location'])
    def pack(mu, sigma, delta):
        v = np.linalg.cholesky(sigma)[ix]; v[diag] = np.log(v[diag])
        return np.r_[mu if fixed is None else [], v, delta, [] if normal else [np.log(df0)]]
    def unpack(theta):
        mu = theta[:d] if fixed is None else fixed
        v = theta[nloc:nloc+nc].copy(); v[diag] = np.exp(v[diag])
        chol = np.zeros((d,d)); chol[ix] = v
        return mu, chol@chol.T, theta[nloc+nc:nloc+nc+d], None if normal else np.exp(theta[-1])
    def objective(theta):
        try:
            with np.errstate(all='ignore'):
                result = -logdensity(z, *unpack(theta), points, seed).sum()
            return float(result) if np.isfinite(result) else 1e100
        except (ValueError, np.linalg.LinAlgError): return 1e100
    bounds = [(None,None)]*(nloc+nc)+[(-10.,10.)]*d+([] if normal else [(np.log(.3),np.log(200.))])
    for j in diag: bounds[nloc+j] = (-8.,8.)
    direction = np.where(stats.skew(z,axis=0) >= 0, 1., -1.)*np.sqrt(base_sigma.diagonal())*.5
    attempts, candidates = [], []
    for label, delta in [('symmetric',np.zeros(d)),('sample-skew',direction),('opposite-skew',-direction)]:
        b = np.sqrt(2/np.pi) if normal else (np.sqrt(df0/np.pi)/special.poch((df0-1)/2,.5) if df0 > 1 else 0.)
        initial = pack(mu0-b*delta if fixed is None else fixed, base_sigma, delta)
        result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                                   options=dict(maxiter=max_iterations,maxfun=100000,ftol=1e-9,gtol=1e-5))
        attempts.append(dict(start=label,converged=bool(result.success),iterations=int(result.nit),message=str(result.message),negative_loglik=float(result.fun)))
        if np.isfinite(result.fun) and result.fun < 1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite SDB fit')
    result = min([r for r in candidates if r.success] or candidates,key=lambda r:r.fun)
    mu,sigma,delta,df = unpack(result.x)
    fitted_lp = logdensity(z,mu,sigma,delta,df,points,seed)
    audit_lp = logdensity(z,mu,sigma,delta,df,points*4,seed+1)
    check_lp = logdensity(z,mu,sigma,delta,df,points*4,seed+2)
    ll_change = float(max(abs((audit_lp-fitted_lp).sum()),abs((audit_lp-check_lp).sum())))
    max_change = float(max(np.max(abs(audit_lp-fitted_lp)),np.max(abs(audit_lp-check_lp))))
    stable = bool(np.isfinite(ll_change) and np.isfinite(max_change) and ll_change <= .1 and max_change <= .02)
    boundary = any((lo is not None and abs(v-lo)<1e-4) or (hi is not None and abs(v-hi)<1e-4) for v,(lo,hi) in zip(result.x,bounds))
    status = 'not_converged' if not result.success else ('boundary' if boundary else ('ok' if stable else 'cdf_unstable'))
    ll = float(audit_lp.sum()-n*np.log(unit).sum()); k = nloc+nc+d+int(not normal)
    fit = dict(model=model,location=(center+unit*mu).tolist(),scatter=(sigma*np.outer(unit,unit)).tolist(),
               delta=(unit*delta).tolist(),df=None if df is None else float(df),observations=n,dimensions=d,parameters=k,
               loglik=ll,aic=2*k-2*ll,bic=np.log(n)*k-2*ll,status=status,converged=bool(result.success),boundary=boundary,
               cdf_points=points*4,cdf_seed=seed+1,cdf_fit_points=points,cdf_fit_seed=seed,cdf_audit_passed=stable,
               cdf_loglik_change=ll_change,cdf_max_logpdf_change=max_change,message=str(result.message),attempts=attempts)
    frozen = JointSDB(fit); cov = frozen.cov() if df is None or df > 2 else None
    fit.update(mean=frozen.mean().tolist() if df is None or df > 1 else None,
               covariance=None if cov is None else cov.tolist(),correlation=None if cov is None else (cov/np.sqrt(np.outer(cov.diagonal(),cov.diagonal()))).tolist(),
               scatter_correlation=(sigma/np.sqrt(np.outer(sigma.diagonal(),sigma.diagonal()))).tolist(),fit_sec=time.perf_counter()-started)
    return fit
