"""EGB2: loc + scale*logit(U), U ~ Beta(a,b).

Independent implementation; see https://doi.org/10.1007/s13209-015-0134-1.
Shape equality gives symmetry, not a zero-valued skew parameter.
"""
import numpy as np
from scipy import optimize, special, stats

EGB2_MODELS = ('egb2-symmetric', 'egb2-skewed')


class EGB2(stats.rv_continuous):
    def _argcheck(self, a, b):
        return (a > 0) & (b > 0) & np.isfinite(a) & np.isfinite(b)

    def _logpdf(self, x, a, b):
        return -a*np.logaddexp(0., -x)-b*np.logaddexp(0., x)-special.betaln(a,b)

    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x,a,b))

    def _cdf(self, x, a, b):
        # Evaluate the small tail directly on either side of zero.
        negative=x <= 0
        aa=np.where(negative,a,b); bb=np.where(negative,b,a)
        xx=-np.abs(x)
        u=special.expit(xx)
        small=special.betainc(aa,bb,u)
        extreme=xx < -700
        logtail=aa*np.where(extreme,xx,-np.inf)-np.log(aa)-special.betaln(aa,bb)
        small=np.where(extreme,np.exp(logtail),small)
        large=np.where(extreme,1-small,special.betaincc(aa,bb,u))
        return np.where(negative,small,large)

    def _sf(self, x, a, b):
        return self._cdf(-x,b,a)

    def _logcdf(self, x, a, b):
        with np.errstate(divide='ignore'):
            ordinary=np.log(self._cdf(x,a,b))
        # Leading incomplete-beta expansion, used only far into the tail.
        return np.where(x < -700, a*x-np.log(a)-special.betaln(a,b), ordinary)

    def _logsf(self, x, a, b):
        return self._logcdf(-x,b,a)

    def _ppf(self, p, a, b):
        # Invert on the U<=1/2 side, avoiding 1-U cancellation.
        reflect=p > special.betainc(a,b,.5)
        aa=np.where(reflect,b,a); bb=np.where(reflect,a,b)
        prob=np.where(reflect,1-p,p)
        u=special.betaincinv(aa,bb,prob)
        with np.errstate(divide='ignore'):
            value=np.log(u)-np.log1p(-u)
            extreme=(np.log(prob)+np.log(aa)+special.betaln(aa,bb))/aa
        value=np.where(u < 1e-300,extreme,value)
        return np.where(reflect,-value,value)

    def _isf(self, p, a, b):
        return -self._ppf(p,b,a)

    def _stats(self, a, b):
        var=special.polygamma(1,a)+special.polygamma(1,b)
        return (special.digamma(a)-special.digamma(b),var,
                (special.polygamma(2,a)-special.polygamma(2,b))/var**1.5,
                (special.polygamma(3,a)+special.polygamma(3,b))/var**2)

    def _rvs(self, a, b, size=None, random_state=None):
        return self._ppf(random_state.uniform(size=size),a,b)


egb2=EGB2(name='egb2',shapes='a, b')


def fit_egb2(x, symmetric=False, location=None, max_iterations=10000):
    """Multi-start bounded MLE on standardized input; tied shapes if symmetric."""
    def unpack(v):
        a=np.exp(v[0]); b=a if symmetric else np.exp(v[1])
        return a,b,v[-2] if location is None else location,np.exp(v[-1])

    def objective(v):
        a,b,loc,scale=unpack(v)
        value=-float(np.sum(egb2._logpdf((x-loc)/scale,a,b)-np.log(scale)))
        return value if np.isfinite(value) else 1e100

    bounds=[(np.log(.01),np.log(200.))]*(1 if symmetric else 2)
    bounds+=([] if location is not None else [(None,None)])+[(-12.,12.)]
    starts=[(.3,.3),(1.,1.),(5.,5.)]
    if not symmetric: starts += [(.3,1.),(1.,.3),(1.,5.),(5.,1.)]
    candidates=[]
    for a,b in starts:
        mean,var=egb2.stats(a,b,moments='mv')
        scale=np.std(x)/np.sqrt(var)
        loc=float(np.mean(x)-scale*mean)
        start=[np.log(a)]+([] if symmetric else [np.log(b)])
        start+=([] if location is not None else [loc])+[np.log(scale)]
        result=optimize.minimize(objective,start,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'maxfun':100000,'ftol':1e-11})
        if np.isfinite(result.fun) and result.fun<1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite EGB2 fit')
    good=[r for r in candidates if r.success]
    result=min(good or candidates,key=lambda r:r.fun)
    result.boundary=any(lo is not None and (abs(v-lo)<1e-4 or abs(v-hi)<1e-4)
                        for v,(lo,hi) in zip(result.x,bounds))
    return unpack(result.x),result
