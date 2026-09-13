"""Independent Meixner / NEF-GHS implementation.

Parameterization: Hess (2018), doi:10.15559/18-VMSTA96, equation (2.6), t=1.
Standardized density uses alpha=1, mu=0, delta>0, -pi<b<pi.
Fischer (2002), NEF-GHS: lambda=2*delta, theta=b/2, scale=alpha/2.
"""
from functools import lru_cache
import warnings
import numpy as np
from scipy import integrate, optimize, special, stats

MEIXNER_MODELS=('meixner-symmetric','meixner-skewed')


def logdensity(x,delta,b):
    x,delta,b=np.broadcast_arrays(x,delta,b)
    infinite=np.isinf(x); xx=np.where(infinite,0.,x)
    result=(2*delta*np.log(2*np.cos(b/2))-np.log(2*np.pi)
            -special.gammaln(2*delta)+b*xx+2*special.loggamma(delta+1j*xx).real)
    return np.where(infinite,-np.inf,result)


@lru_cache(maxsize=128)
def geometry(delta,b):
    mean=delta*np.tan(b/2)
    sd=np.sqrt(delta/2)/np.cos(b/2)
    if b == 0: mode=0.
    else:
        bb=abs(b)
        upper=max(abs(mean)+sd,delta,1.)
        score=lambda x: bb-2*special.digamma(delta+1j*x).imag
        for _ in range(100):
            if score(upper)<0: break
            upper*=2
        else: raise ValueError('Could not bracket Meixner mode')
        mode=np.copysign(optimize.brentq(score,0.,upper,xtol=1e-13),b)
    return mean,sd,(mode-mean)/sd


def tail(x,delta,b,survival=False):
    """Integrate in standard-deviation units, splitting at the density mode."""
    if np.isinf(x): return float(x<0) if survival else float(x>0)
    mean,sd,mode=geometry(float(delta),float(b))
    endpoint=(x-mean)/sd
    lo,hi=(endpoint,np.inf) if survival else (-np.inf,endpoint)
    bounds=[lo]+([mode] if lo<mode<hi else [])+[hi]
    value=0.; error=0.
    with warnings.catch_warnings():
        warnings.simplefilter('error',integrate.IntegrationWarning)
        for left,right in zip(bounds[:-1],bounds[1:]):
            try:
                part,err=integrate.quad(lambda z: float(np.exp(logdensity(mean+sd*z,delta,b)))*sd,left,right,epsabs=1e-11,epsrel=1e-9,limit=300)
            except integrate.IntegrationWarning as exc:
                raise ValueError('Meixner probability quadrature failed: '+str(exc)) from exc
            value+=part; error+=err
    if not np.isfinite(value) or error>max(1e-9,abs(value)*1e-7) or not -1e-9<=value<=1+1e-9:
        raise ValueError('Meixner probability quadrature failed accuracy check')
    return min(1.,max(0.,value))


def quantile(p,delta,b):
    if p == 0: return -np.inf
    if p == 1: return np.inf
    # Always solve on the lower half using exact reflection, avoiding 1-CDF.
    if p>.5: return -quantile(1-p,delta,-b)
    mean,sd,_=geometry(float(delta),float(b))
    objective=lambda z: tail(mean+sd*z,delta,b)-p
    lo=-1.; hi=1.
    for _ in range(100):
        if objective(lo)<=0: break
        lo*=2
    else: raise ValueError('Could not bracket Meixner lower quantile')
    for _ in range(100):
        if objective(hi)>=0: break
        hi*=2
    else: raise ValueError('Could not bracket Meixner upper quantile')
    return mean+sd*optimize.brentq(objective,lo,hi,xtol=1e-11,rtol=1e-12)


class Meixner(stats.rv_continuous):
    def _argcheck(self,delta,b):
        return (delta>0)&np.isfinite(delta)&np.isfinite(b)&(np.abs(b)<np.pi)

    def _logpdf(self,x,delta,b): return logdensity(x,delta,b)
    def _pdf(self,x,delta,b): return np.exp(logdensity(x,delta,b))
    def _cdf(self,x,delta,b): return np.vectorize(tail,otypes=[float])(x,delta,b)
    def _sf(self,x,delta,b): return np.vectorize(lambda v,d,s: tail(v,d,s,True),otypes=[float])(x,delta,b)
    def _ppf(self,p,delta,b): return np.vectorize(quantile,otypes=[float])(p,delta,b)
    def _isf(self,p,delta,b): return -self._ppf(p,delta,-b)
    def _stats(self,delta,b):
        return delta*np.tan(b/2),delta/(2*np.cos(b/2)**2),np.sqrt(2/delta)*np.sin(b/2),(2-np.cos(b))/delta
    def _rvs(self,delta,b,size=None,random_state=None):
        return self._ppf(random_state.uniform(size=size),delta,b)


meixner=Meixner(name='meixner',shapes='delta, b')


def fit_meixner(x,symmetric=False,location=None,max_iterations=10000):
    """Multiple shape/skew starts; data already standardized by fit_one."""
    def unpack(v):
        delta=np.exp(v[0]); b=0. if symmetric else v[1]
        loc=v[-2] if location is None else location
        return delta,b,loc,np.exp(v[-1])
    def objective(v):
        delta,b,loc,scale=unpack(v)
        value=-float(np.sum(logdensity((x-loc)/scale,delta,b)-np.log(scale)))
        return value if np.isfinite(value) else 1e100
    bounds=[(np.log(.01),np.log(200.))]+([] if symmetric else [(-np.pi+.001,np.pi-.001)])+([] if location is not None else [(None,None)])+[(-12.,12.)]
    candidates=[]; attempts=[]
    for delta in (.25,1.,5.):
        for b in ((0.,) if symmetric else (0.,.6,-.6)):
            mean,var,_,_=meixner.stats(delta,b,moments='mvsk')
            scale=np.std(x)/np.sqrt(var)
            loc=float(np.mean(x)-scale*mean)
            start=[np.log(delta)]+([] if symmetric else [b])+([] if location is not None else [loc])+[np.log(scale)]
            result=optimize.minimize(objective,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':max_iterations,'maxfun':100000,'ftol':1e-11})
            attempts.append(dict(delta_start=delta,b_start=b,converged=bool(result.success),message=str(result.message)))
            if np.isfinite(result.fun) and result.fun<1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite Meixner fit')
    good=[r for r in candidates if r.success]
    result=min(good or candidates,key=lambda r:r.fun)
    result.boundary=any(lo is not None and (abs(v-lo)<1e-4 or abs(v-hi)<1e-4) for v,(lo,hi) in zip(result.x,bounds))
    result.attempts=attempts
    return unpack(result.x),result
