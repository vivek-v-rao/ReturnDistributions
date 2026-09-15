"""Shared-subordinator multivariate NTS; independent NumPy/SciPy implementation.

X=mu+gamma*W+sqrt(W)*L*Z, E W=1, Var W=(1-alpha)/lam.
The positive stable density integral is derived by conditioning the Kanter
sampler on its uniform angle; exponential tilting gives the mixing density.
No multivariate Fourier inversion or copula approximation is used.
"""
from functools import lru_cache
import time

import numpy as np
from scipy import linalg, optimize, special

from .nts import nts, NTS_MODELS


@lru_cache(maxsize=12)
def rule(n):
    return special.roots_legendre(n)


def mixing_logpdf(logw, alpha, lam, points=256):
    logw = np.asarray(logw)
    if alpha == .5:
        w = np.exp(logw)
        return .5*np.log(lam/np.pi)-1.5*logw-lam*(w-1)**2/w
    nodes, weights = rule(points)
    angle = (nodes+1)*np.pi/2
    loga = (alpha*np.log(np.sin(alpha*angle))+(1-alpha)*np.log(np.sin((1-alpha)*angle))
            -np.log(np.sin(angle)))/(1-alpha)
    logscale = ((1-alpha)*np.log(lam)-np.log(alpha))/alpha
    logs = logw-logscale
    exponent = loga[None,:]-alpha/(1-alpha)*logs[:,None]
    with np.errstate(over='ignore'):
        angular = special.logsumexp(np.log(weights/2)[None,:]+loga[None,:]-np.exp(exponent), axis=1)
    return (np.log(alpha/(1-alpha))-logscale-logs/(1-alpha)+angular+lam/alpha-lam*np.exp(logw))


@lru_cache(maxsize=24)
def mixing_rule(alpha, lam, points=256, tail=1e-12):
    if not (0 < alpha < 1 and lam > 0 and np.isfinite([alpha,lam]).all()):
        raise ValueError('Invalid NTS mixing parameters')
    def bound(logw):
        # Optimized Chernoff exponent K(t)-t*w, with K'(t)=w.
        with np.errstate(over='ignore'):
            return lam*(-np.expm1(logw)-(1-alpha)/alpha*np.expm1(alpha/(alpha-1)*logw))
    def endpoint(sign):
        edge = float(sign)
        for _ in range(10):
            if bound(edge) <= np.log(tail): break
            edge *= 2
        else: raise ValueError('Could not bound NTS mixing tails')
        return optimize.brentq(lambda z: bound(z)-np.log(tail), min(edge,0.), max(edge,0.))
    low, high = endpoint(-1), endpoint(1)
    nodes, weights = rule(points)
    logw = low+(nodes+1)*(high-low)/2
    logweights = mixing_logpdf(logw,alpha,lam,points)+logw+np.log(weights*(high-low)/2)
    mass = float(np.exp(special.logsumexp(logweights)))
    if not np.isfinite(mass) or abs(mass-1) > 2e-5:
        raise ValueError(f'NTS mixing quadrature mass {mass:g} differs from one')
    # Do not silently renormalize a defective quadrature rule.
    return np.exp(logw), logweights, mass


class JointNTS:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'],dtype=float)
        self.scatter = np.asarray(fit['scatter'],dtype=float)
        self.gamma = np.asarray(fit.get('gamma',np.zeros_like(self.location)),dtype=float)
        self.alpha, self.lam = float(fit['alpha']),float(fit['lam'])
        d = len(self.location)
        if self.scatter.shape != (d,d) or self.gamma.shape != (d,) or not np.isfinite(self.location).all() or not np.isfinite(self.gamma).all():
            raise ValueError('Invalid NTS location/scatter/skew vector')
        if not np.isfinite(self.scatter).all() or not np.allclose(self.scatter,self.scatter.T):
            raise ValueError('Invalid NTS scatter')
        if not 0 < self.alpha < 1 or not self.lam > 0 or not np.isfinite(self.lam):
            raise ValueError('Invalid NTS shapes')
        self.chol = np.linalg.cholesky(self.scatter)
        self.points = int(fit.get('nts_quadrature_points',512))
        self.tail = float(fit.get('nts_tail_tolerance',1e-16))

    def mean(self): return self.location+self.gamma
    def cov(self): return self.scatter+(1-self.alpha)/self.lam*np.outer(self.gamma,self.gamma)
    def cf(self,t):
        t = np.asarray(t,dtype=float)
        z = 1j*(t@self.gamma)-.5*np.einsum('...i,ij,...j->...',t,self.scatter,t)
        return np.exp(1j*(t@self.location)-self.lam/self.alpha*np.expm1(self.alpha*np.log1p(-z/self.lam)))
    def logpdf(self, x):
        x = np.asarray(x,dtype=float)
        values = np.atleast_2d(x)
        if values.shape[1] != len(self.location) or not np.isfinite(values).all():
            raise ValueError('NTS density requires finite, correctly ordered coordinates')
        w, logweights, _ = mixing_rule(self.alpha,self.lam,self.points,self.tail)
        z = linalg.solve_triangular(self.chol,(values-self.location).T,lower=True,check_finite=False).T
        g = linalg.solve_triangular(self.chol,self.gamma,lower=True,check_finite=False)
        constant = -len(self.location)/2*np.log(2*np.pi)-np.log(self.chol.diagonal()).sum()
        result = np.empty(len(values))
        for start in range(0,len(values),256):
            y = z[start:start+256]
            # Squared residual avoids cancellation at y ~= w*g.
            residual = y[:,None,:]/np.sqrt(w)[None,:,None]-np.sqrt(w)[None,:,None]*g
            logs = constant-.5*np.sum(residual**2,axis=2)-len(g)/2*np.log(w)+logweights
            result[start:start+len(y)] = special.logsumexp(logs,axis=1)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite NTS log density')
        return result[0] if x.ndim == 1 else result

    def pdf(self,x): return np.exp(self.logpdf(x))
    def project(self,weights):
        weights = np.asarray(weights,dtype=float)
        scale = np.sqrt(weights@self.scatter@weights)
        return nts(self.alpha,self.lam,float(weights@self.gamma)/scale,
                   loc=float(weights@self.location),scale=scale)

    def rvs(self,size=1,random_state=None):
        from .nts import sample_mixing
        rng = np.random.default_rng(random_state)
        w = sample_mixing(self.alpha,self.lam,size,rng)
        return self.location+w[:,None]*self.gamma+np.sqrt(w)[:,None]*(rng.normal(size=(size,len(self.location)))@self.chol.T)


def fit_nts_joint(x,model,location=None,max_iterations=2000,points=256):
    started = time.perf_counter()
    n,d = x.shape
    center,unit = x.mean(axis=0),x.std(axis=0)
    z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location,(d,))-center)/unit
    nloc = d if fixed is None else 0
    skewed = model == 'nts-skewed'
    indices = np.tril_indices(d)
    diagonal = np.flatnonzero(indices[0]==indices[1])
    nc = len(indices[0])
    def unpack(v,precision=points,tail=1e-12):
        mu = v[:d] if fixed is None else fixed
        entries = v[nloc:nloc+nc].copy()
        entries[diagonal] = np.exp(entries[diagonal])
        chol = np.zeros((d,d)); chol[indices] = entries
        return dict(location=mu,scatter=chol@chol.T,alpha=v[nloc+nc],lam=np.exp(v[nloc+nc+1]),
                    gamma=v[nloc+nc+2:] if skewed else np.zeros(d),nts_quadrature_points=precision,nts_tail_tolerance=tail)
    def objective(v):
        try:
            value = -float(JointNTS(unpack(v)).logpdf(z).sum())
            return value if np.isfinite(value) else 1e100
        except (ValueError,ArithmeticError): return 1e100
    bounds = [(None,None)]*(nloc+nc)
    for j in diagonal: bounds[nloc+j] = (-6.,6.)
    bounds += [(.1,.9),(np.log(.1),np.log(30.))]+([(-5.,5.)]*d if skewed else [])
    mu0 = np.zeros(d) if fixed is None else fixed
    chol = np.linalg.cholesky((z-mu0).T@(z-mu0)/n)
    entries = chol[indices]; entries[diagonal] = np.log(entries[diagonal])
    candidates,attempts = [],[]
    for alpha in [.25,.5,.75]:
        initial = np.r_[mu0 if fixed is None else [],entries,alpha,np.log(2.),np.zeros(d) if skewed else []]
        r = optimize.minimize(objective,initial,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'maxfun':100000,'ftol':1e-10,'gtol':1e-6})
        score = np.asarray(r.jac).copy()
        for j,(lo,hi) in enumerate(bounds):
            if lo is not None and ((r.x[j]<=lo+1e-6 and score[j]>0) or (r.x[j]>=hi-1e-6 and score[j]<0)): score[j]=0
        projected = float(np.max(np.abs(score))/n)
        if r.success and (not np.isfinite(projected) or projected>1e-3):
            r.success=False; r.message='Nonstationary projected score'
        attempts.append(dict(alpha_start=alpha,converged=bool(r.success),message=str(r.message),projected_score=projected))
        if np.isfinite(r.fun) and r.fun<1e99: candidates.append(r)
    if not candidates: raise ValueError('No finite NTS joint fit; try more quadrature points')
    good = [r for r in candidates if r.success]
    r = min(good or candidates,key=lambda v:v.fun)
    fine = unpack(r.x,points*2,1e-16)
    coarse_logs = JointNTS(unpack(r.x)).logpdf(z)
    fine_logs = JointNTS(fine).logpdf(z)
    delta = fine_logs-coarse_logs
    audit = bool(abs(delta.sum()) <= .01 and np.max(np.abs(delta)) <= .001)
    boundary = any(lo is not None and (abs(v-lo)<1e-4 or abs(v-hi)<1e-4) for v,(lo,hi) in zip(r.x,bounds))
    status = 'not_converged' if not r.success else ('boundary' if boundary else ('ok' if audit else 'quadrature_failed'))
    fine.update(location=(center+unit*fine['location']).tolist(),scatter=(fine['scatter']*np.outer(unit,unit)).tolist(),
                gamma=(unit*fine['gamma']).tolist())
    dist = JointNTS(fine)
    covariance = dist.cov()
    ll = float(fine_logs.sum()-n*np.log(unit).sum())
    k = nloc+nc+2+(d if skewed else 0)
    return dict(fine,model=model,family='shared-subordinator-nts',observations=n,dimensions=d,parameters=k,
        loglik=ll,aic=2*k-2*ll,bic=np.log(n)*k-2*ll,nts_alpha=float(fine['alpha']),mean=dist.mean().tolist(),covariance=covariance.tolist(),
        correlation=(covariance/np.sqrt(np.outer(covariance.diagonal(),covariance.diagonal()))).tolist(),
        converged=bool(r.success),boundary=boundary,status=status,message=str(r.message),attempts=attempts,
        nts_audit_passed=audit,nts_loglik_change=float(delta.sum()),nts_max_logpdf_change=float(np.max(np.abs(delta))),
        fit_sec=time.perf_counter()-started)
