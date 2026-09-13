"""Real-line Champernowne density C(lam)/(cosh(x)+lam), lam > -1.

Independent implementation; see Baker (2026), Appendix A,
https://doi.org/10.1007/s42519-026-00549-4 (there p=(1-lam)/2).
lam=0 is scipy.stats.hypsecant; lam=1 is scipy.stats.logistic.
"""
import numpy as np
from scipy import optimize, stats


def log_sinh(x):
    return x+np.log(-np.expm1(-2*x))-np.log(2.)


class Champernowne(stats.rv_continuous):
    def _argcheck(self,lam): return (lam > -1) & np.isfinite(lam)

    def _logpdf(self,x,lam):
        x,lam=np.broadcast_arrays(x,lam)
        norm=np.zeros_like(x,dtype=float)
        low=lam<1; high=lam>1
        theta=np.arccos(lam[low])
        norm[low]=np.log(np.sinc(theta/np.pi))
        eta=np.arccosh(lam[high])
        norm[high]=log_sinh(eta)-np.log(eta)
        a=np.abs(x)
        with np.errstate(divide='ignore',invalid='ignore'):
            # (1-exp(-a))^2 + 2*(1+lam)*exp(-a), no cancellation at lam~-1.
            denominator=np.logaddexp(2*np.log(-np.expm1(-a)),np.log(2.)+np.log1p(lam)-a)
            return norm-a-denominator

    def _pdf(self,x,lam): return np.exp(self._logpdf(x,lam))

    def _positive_tail(self,x,lam):
        x,lam=np.broadcast_arrays(np.abs(x),lam)
        u=np.exp(-x); result=np.array(u/(1+u),copy=True)
        low=lam<1; high=lam>1
        theta=np.arccos(lam[low]); root=np.sqrt((1-lam[low])*(1+lam[low]))
        result[low]=np.arctan2(u[low]*root,-np.expm1(-x[low])+(1+lam[low])*u[low])/theta
        eta=np.arccosh(lam[high]); xx=x[high]
        # Small eta uses atanh to avoid subtracting nearly equal logarithms.
        values=np.empty_like(eta); small=eta<1
        v=np.exp(-xx[small])
        values[small]=np.arctanh(v*np.sinh(eta[small])/(1+v*np.cosh(eta[small])))/eta[small]
        values[~small]=(np.logaddexp(0.,eta[~small]-xx[~small])-np.logaddexp(0.,-eta[~small]-xx[~small]))/(2*eta[~small])
        result[high]=values
        return result

    def _cdf(self,x,lam):
        p=self._positive_tail(x,lam)
        return np.where(x<0,p,1-p)

    def _sf(self,x,lam): return self._cdf(-x,lam)

    def _ppf(self,p,lam):
        p,lam=np.broadcast_arrays(p,lam)
        tail=np.minimum(p,1-p)
        value=np.array(np.log1p(-tail)-np.log(tail),copy=True)
        low=lam<1; high=lam>1
        theta=np.arccos(lam[low])
        value[low]=np.log(np.sin(theta*(1-tail[low])))-np.log(np.sin(theta*tail[low]))
        eta=np.arccosh(lam[high])
        value[high]=log_sinh(eta*(1-tail[high]))-log_sinh(eta*tail[high])
        return np.where(p<.5,-value,value)

    def _isf(self,p,lam): return -self._ppf(p,lam)

    def _stats(self,lam):
        lam=np.asarray(lam)
        variance=np.full_like(lam,np.pi**2/3,dtype=float)
        kurt=np.full_like(lam,1.2,dtype=float)
        low=lam<1; high=lam>1
        theta=np.arccos(lam[low]); eta=np.arccosh(lam[high])
        denominator=(np.pi-theta)*(np.pi+theta)
        variance[low]=denominator/3
        kurt[low]=1.2*(np.pi**2+theta**2)/denominator
        variance[high]=(np.pi**2+eta**2)/3
        kurt[high]=1.2*(np.pi**2-eta**2)/(np.pi**2+eta**2)
        return np.zeros_like(variance),variance,np.zeros_like(variance),kurt

    def _rvs(self,lam,size=None,random_state=None):
        return self._ppf(random_state.uniform(size=size),lam)


champernowne=Champernowne(name='champernowne',shapes='lam')


def fit_champernowne(x,location=None,max_iterations=10000):
    """Multi-start fit on already standardized observations.

Constrain log(1+lam) and log(scale) to [-12,12] for numerical stability;
caller flags solutions at these bounds. Return SciPy-shaped parameters.
"""
    def unpack(v):
        return np.expm1(v[0]), v[1] if location is None else location, np.exp(v[-1])
    def objective(v):
        lam,loc,scale=unpack(v)
        value=-champernowne.logpdf(x,lam,loc=loc,scale=scale).sum()
        return float(value) if np.isfinite(value) else 1e100
    bounds=[(-12.,12.)]+([(None,None)] if location is None else [])+[(-12.,12.)]
    candidates=[]
    for lam in (-.8,0.,1.,5.):
        scale=np.std(x)/np.sqrt(champernowne.var(lam))
        start=[np.log1p(lam)]+([float(np.median(x))] if location is None else [])+[np.log(scale)]
        result=optimize.minimize(objective,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':max_iterations,'ftol':1e-11,'maxfun':100000})
        if np.isfinite(result.fun) and result.fun<1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite Champernowne fit')
    good=[r for r in candidates if r.success]
    result=min(good or candidates,key=lambda r:r.fun)
    result.boundary=any(lo is not None and (abs(v-lo)<1e-4 or abs(v-hi)<1e-4) for v,(lo,hi) in zip(result.x,bounds))
    return unpack(result.x),result
