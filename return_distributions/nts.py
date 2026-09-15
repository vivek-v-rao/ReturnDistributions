"""Normal tempered stable, with an identified mean-one mixing variable.

X = b*W + sqrt(W)*Z; log E[exp(s*W)] = lam/alpha *
(1-(1-s/lam)**alpha). 0<alpha<1, lam>0, Z standard normal.
This rescales the TempStable charNTS definition:
https://search.r-project.org/CRAN/refmans/TempStable/html/charNTS.html
Independent implementation; no R package source code is copied.
"""
import numpy as np
from scipy import integrate, optimize, stats

NTS_MODELS = ('nts-symmetric','nts-skewed')


def sample_mixing(aa,ll,n,rng):
    """Exact tilted-positive-stable mean-one mixing draws, shared by joint NTS."""
    pieces=max(1,int(np.ceil(ll/aa)))
    if pieces>10000: raise ValueError('NTS mixing sampler requires too many pieces for these parameters')
    logscale=((1-aa)*np.log(ll)-np.log(aa)-np.log(pieces))/aa
    w=np.zeros(n)
    for _ in range(pieces):
        pending=np.arange(n)
        while len(pending):
            angle=rng.uniform(np.finfo(float).eps,np.pi,size=len(pending))
            exponential=rng.exponential(size=len(pending))
            logstable=(np.log(np.sin(aa*angle))-np.log(np.sin(angle))/aa
                +(1-aa)/aa*(np.log(np.sin((1-aa)*angle))-np.log(exponential)))
            with np.errstate(over='ignore'):
                draw=np.exp(logscale+logstable)
            accepted=np.log(rng.uniform(size=len(pending))) < -ll*draw
            w[pending[accepted]]+=draw[accepted]
            pending=pending[~accepted]
    return w


def cumulant(z, alpha, lam, b):
    # expm1 preserves the VG limit as alpha tends to zero.
    return -lam/alpha*np.expm1(alpha*np.log1p(-(b*z+.5*z*z)/lam))


def cf(t, alpha, lam, b=0.):
    return np.exp(cumulant(1j*np.asarray(t),alpha,lam,b))


def saddle(x, alpha, lam, b):
    radius=np.sqrt(b*b+2*lam)
    left=np.full_like(x,-b-radius); right=np.full_like(x,-b+radius)
    for _ in range(65):
        h=(left+right)/2
        d=np.maximum(1-(b*h+.5*h*h)/lam,np.finfo(float).tiny)
        derivative=(b+h)*d**(alpha-1)
        left=np.where(derivative<x,h,left)
        right=np.where(derivative<x,right,h)
    return (left+right)/2


def invert(x, alpha, lam, b, probability=False, gradient=False, shortfall=False):
    """Saddlepoint-shifted Fourier quadrature, NOT a saddlepoint approximation.

    Inversion stays inside the MGF strip. Integrate in local curvature units
    and reject inaccurate/nonpositive integrals instead of flooring densities.
    Probability returns CDF, choosing the lower/upper contour by x vs mean.
    """
    x=np.atleast_1d(np.asarray(x,dtype=float))
    h=saddle(x,alpha,lam,b)
    upper=x>=b
    if shortfall: upper=np.zeros_like(x,dtype=bool)
    if probability or shortfall:
        radius=np.sqrt(b*b+2*lam)
        distance=np.where(upper,-b+radius,b+radius)
        minimum=np.minimum(.2/np.sqrt(1+b*b*(1-alpha)/lam),.2*distance)
        h=np.where(upper,np.maximum(h,minimum),np.minimum(h,-minimum))
    d=1-(b*h+.5*h*h)/lam
    if (d<=0).any(): raise ValueError('NTS contour reached MGF boundary')
    curvature=d**(alpha-1)+(1-alpha)/lam*(b+h)**2*d**(alpha-2)
    unit=np.sqrt(curvature)
    kh=cumulant(h,alpha,lam,b)

    def exponent(u):
        t=u/unit
        return cumulant(h+1j*t,alpha,lam,b)-kh-1j*t*x

    cutoff=8.
    for _ in range(25):
        # Include interval scale: a tiny integrand at a very large frequency
        # is not, by itself, evidence of a negligible omitted tail.
        if np.max(exponent(cutoff).real)+np.log(cutoff)<-40: break
        cutoff*=2
    else: raise ValueError('NTS Fourier cutoff did not converge')

    def kernel(u):
        z=h+1j*u/unit
        value=np.exp(exponent(u))
        if probability: value=value/z
        if shortfall: value=value/z**2
        if gradient:
            q=(b*z+.5*z*z)/lam
            logd=np.log1p(-q)
            power=np.exp(alpha*logd)
            derivative_alpha=lam/alpha**2*(np.expm1(alpha*logd)-alpha*power*logd)
            derivative_lam=-np.expm1(alpha*logd)/alpha-q*np.exp((alpha-1)*logd)
            factors=np.array([np.ones_like(z),derivative_alpha,lam*derivative_lam,
                              z*np.exp((alpha-1)*logd),-z])
            return (factors*value).real/unit
        return value.real/unit

    value,error,info=integrate.quad_vec(kernel,0.,cutoff,epsabs=2e-10,
        epsrel=2e-9,norm='max',limit=2000,full_output=True)
    if not info.success or not np.isfinite(value).all() or error>max(2e-9,np.max(np.abs(value))*2e-7):
        raise ValueError('NTS Fourier quadrature failed accuracy check')
    derivatives=None
    if gradient:
        derivatives=value[1:]/value[0]
        value=value[0]
    if probability: value=np.where(upper,value,-value)
    if (value<=0).any(): raise ValueError('NTS Fourier inversion produced a nonpositive integral')
    logvalue=kh-h*x+np.log(value/np.pi)
    if shortfall: return np.exp(logvalue)
    if gradient: return logvalue,derivatives
    if not probability: return logvalue
    tail=np.exp(logvalue)
    if (tail>1+1e-8).any(): raise ValueError('NTS tail probability exceeds one')
    return np.where(upper,-np.expm1(logvalue),tail)


def grouped(x, alpha, lam, b, probability=False):
    """Vectorize by parameter group; process observations in bounded chunks."""
    x,alpha,lam,b=np.broadcast_arrays(x,alpha,lam,b)
    result=np.empty(x.size)
    params=np.column_stack([alpha.ravel(),lam.ravel(),b.ravel()])
    unique,groups=np.unique(params,axis=0,return_inverse=True)
    for group,(aa,ll,bb) in enumerate(unique):
        indices=np.flatnonzero(groups==group)
        for start in range(0,len(indices),256):
            ii=indices[start:start+256]
            result[ii]=invert(x.ravel()[ii],aa,ll,bb,probability)
    return result.reshape(x.shape)


class NormalTemperedStable(stats.rv_continuous):
    def _argcheck(self, alpha, lam, b):
        return (alpha>0)&(alpha<1)&(lam>0)&np.isfinite(lam)&np.isfinite(b)

    def _logpdf(self, x, alpha, lam, b):
        infinite=np.isinf(x)
        result=grouped(np.where(infinite,0.,x),alpha,lam,b)
        return np.where(infinite,-np.inf,result)

    def _pdf(self, x, alpha, lam, b): return np.exp(self._logpdf(x,alpha,lam,b))
    def _cdf(self, x, alpha, lam, b): return grouped(x,alpha,lam,b,True)
    def _sf(self, x, alpha, lam, b): return grouped(-x,alpha,lam,-b,True)

    def _ppf(self, p, alpha, lam, b):
        def scalar(q,aa,ll,bb):
            if q>.5: return -scalar(1-q,aa,ll,-bb)
            sd=np.sqrt(1+bb*bb*(1-aa)/ll)
            lo=bb-sd; hi=bb+sd
            def objective(v): return float(invert([v],aa,ll,bb,True)[0])-q
            for _ in range(80):
                if objective(lo)<=0: break
                lo=bb+2*(lo-bb)
            else: raise ValueError('Could not bracket NTS lower quantile')
            for _ in range(80):
                if objective(hi)>=0: break
                hi=bb+2*(hi-bb)
            else: raise ValueError('Could not bracket NTS upper quantile')
            return optimize.brentq(objective,lo,hi,xtol=1e-10,rtol=1e-11)
        return np.vectorize(scalar,otypes=[float])(p,alpha,lam,b)

    def _isf(self, p, alpha, lam, b): return -self._ppf(p,alpha,lam,-b)

    def _stats(self, alpha, lam, b):
        k2=(1-alpha)/lam
        k3=(1-alpha)*(2-alpha)/lam**2
        k4=(1-alpha)*(2-alpha)*(3-alpha)/lam**3
        var=1+b*b*k2
        return b,var,(3*b*k2+b**3*k3)/var**1.5,(3*k2+6*b*b*k3+b**4*k4)/var**2

    def _rvs(self, alpha, lam, b, size=None, random_state=None):
        # Exact exponential tilting of positive stable draws. Infinite
        # divisibility splits the tilt into pieces with reasonable acceptance.
        alpha,lam,b=np.broadcast_arrays(alpha,lam,b)
        if alpha.size!=1: raise ValueError('NTS sampling requires scalar shape parameters')
        aa=float(alpha.ravel()[0]); ll=float(lam.ravel()[0]); bb=float(b.ravel()[0])
        shape=() if size is None else size
        n=int(np.prod(shape)) if np.ndim(shape)>0 else int(shape or 1)
        w=sample_mixing(aa,ll,n,random_state)
        result=bb*w+np.sqrt(w)*random_state.normal(size=n)
        return result.reshape(shape)


nts=NormalTemperedStable(name='nts',shapes='alpha, lam, b')


def fit_nts(x, symmetric=False, location=None, max_iterations=10000):
    """Multi-start MLE; analytic Fourier scores avoid repeated inversions."""
    def unpack(v):
        alpha=v[0]; lam=np.exp(v[1]); b=0. if symmetric else v[2]
        return alpha,lam,b,v[-2] if location is None else location,np.exp(v[-1])

    def objective(v):
        aa,ll,bb,loc,scale=unpack(v)
        y=(x-loc)/scale
        try:
            logs,scores=invert(y,aa,ll,bb,gradient=True)
            gradient=[scores[0].sum(),scores[1].sum()]
            if not symmetric: gradient.append(scores[2].sum())
            if location is None: gradient.append(-scores[3].sum()/scale)
            gradient.append((-1-y*scores[3]).sum())
            value=-float(np.sum(logs-np.log(scale)))
            grad=-np.asarray(gradient)
            if not np.isfinite(value) or not np.isfinite(grad).all(): raise ValueError('Invalid NTS score')
            return value,grad
        except (ValueError,FloatingPointError,OverflowError):
            return 1e100,np.zeros(len(v))

    bounds=[(.1,.95),(np.log(.1),np.log(50.))]
    bounds+=([] if symmetric else [(-5.,5.)])
    bounds+=([] if location is not None else [(None,None)])+[(-6.,6.)]
    candidates=[]; attempts=[]
    for aa in (.25,.5,.8):
        ll=2.; bb=0. if symmetric else float(np.clip(stats.skew(x),-1.,1.))
        scale=np.std(x)/np.sqrt(1+bb*bb*(1-aa)/ll)
        loc=float(np.mean(x)-scale*bb)
        start=[aa,np.log(ll)]+([] if symmetric else [bb])
        start+=([] if location is not None else [loc])+[np.log(scale)]
        result=optimize.minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'maxfun':100000,'ftol':1e-10,'gtol':1e-6})
        # A rejected integration point can make a line search stop without
        # reaching a stationary fit. Check the feasible projected score too.
        projected=np.array(result.jac,copy=True)
        for j,(lo,hi) in enumerate(bounds):
            if lo is not None and ((result.x[j]<=lo+1e-6 and projected[j]>0)
                                   or (result.x[j]>=hi-1e-6 and projected[j]<0)):
                projected[j]=0.
        result.projected_score=float(np.max(np.abs(projected))/len(x))
        if result.success and result.projected_score>1e-3:
            result.success=False
            result.message='Optimizer stopped with a nonstationary projected score'
        attempts.append(dict(alpha_start=aa,converged=bool(result.success),message=str(result.message)))
        if np.isfinite(result.fun) and result.fun<1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite NTS fit')
    good=[r for r in candidates if r.success]
    result=min(good or candidates,key=lambda r:r.fun)
    result.boundary=any(lo is not None and (abs(v-lo)<1e-4 or abs(v-hi)<1e-4)
                        for v,(lo,hi) in zip(result.x,bounds))
    result.attempts=attempts
    return unpack(result.x),result
