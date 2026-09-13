"""Independent Aas--Haff GH skew-t implementation (JFE 2006,
doi:10.1093/jjfinec/nbj006), derived from the normal/inverse-gamma mixture.
No code from tspydistributions or SkewHyperbolic is incorporated.

X=mu+W*gamma+sqrt(W)*Z; W~InvGamma(nu/2, scale=nu/2), Z~N(0,S).
Location/scatter are not the population mean/covariance.
"""
import time
import warnings
import numpy as np
from scipy import integrate, linalg, optimize, special, stats


def checked_integral(function, lo, hi):
    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        try:
            value, error = integrate.quad(function, lo, hi, epsabs=1e-11, epsrel=1e-8, limit=300)
        except integrate.IntegrationWarning as exc:
            raise ValueError('GH skew-t quadrature failed: '+str(exc)) from exc
    if not np.isfinite(value) or error > max(1e-9, abs(value)*1e-6):
        raise ValueError('GH skew-t quadrature failed')
    return value


def log_kernel(a, b, order):
    """Log integral w^(-order-1) exp[-(a/w+b*w)/2] dw."""
    a, b, order = np.broadcast_arrays(a, b, order)
    with np.errstate(all='ignore'):
        z = np.sqrt(a*b)
        result = np.log(2)+order/2*np.log(b/a)+np.log(special.kve(order,z))-z
        result = np.where(b == 0, special.gammaln(order)+order*np.log(2/a), result)
    # Exact positive integral fallback for small skew / large Bessel order.
    result = np.array(result,copy=True)
    for coordinates in np.argwhere(~np.isfinite(result)):
        index=tuple(coordinates)
        aa, bb, oo = (float(v[index]) for v in (a,b,order))
        mode = aa/(np.hypot(oo,np.sqrt(aa*bb))+oo)
        left, right = aa/(2*mode), bb*mode/2
        width = 1/np.sqrt(left+right)
        def centered(v):
            t = width*v
            if abs(t) > 700: return 0.
            exponent = -oo*t-left*np.expm1(-t)-right*np.expm1(t)
            return np.exp(min(0.,exponent))
        mass = checked_integral(centered,-np.inf,0)+checked_integral(centered,0,np.inf)
        result[index] = -oo*np.log(mode)-left-right+np.log(width*mass)
    return result


def logdensity(x, mu, chol, gamma, df):
    y = linalg.solve_triangular(chol,(np.atleast_2d(x)-mu).T,lower=True)
    g = linalg.solve_triangular(chol,gamma,lower=True)
    d = len(mu)
    return (df/2*np.log(df/2)-special.gammaln(df/2)-d/2*np.log(2*np.pi)
            -np.log(chol.diagonal()).sum()+g@y
            +log_kernel(df+np.sum(y*y,axis=0),float(g@g),(df+d)/2))


class JointGHSkewT:
    def __init__(self, fit):
        self.location = np.asarray(fit['location'],float)
        self.scatter = np.asarray(fit['scatter'],float)
        self.gamma = np.asarray(fit['gamma'],float)
        self.df = float(fit['df'])
        if not np.isfinite(self.df) or self.df <= 0: raise ValueError('df must be positive')
        if self.gamma.shape != self.location.shape or not np.isfinite(self.gamma).all():
            raise ValueError('Invalid gamma')
        self.chol = np.linalg.cholesky(self.scatter)

    def logpdf(self,x):
        value = logdensity(x,self.location,self.chol,self.gamma,self.df)
        return value[0] if np.asarray(x).ndim == 1 else value

    def pdf(self,x): return np.exp(self.logpdf(x))

    def mean(self):
        if np.any(self.gamma) and self.df <= 2: return None
        if not np.any(self.gamma) and self.df <= 1: return None
        return self.location if not np.any(self.gamma) else self.location+self.gamma*self.df/(self.df-2)

    def cov(self):
        if self.df <= (4 if np.any(self.gamma) else 2): return None
        result = self.scatter*self.df/(self.df-2)
        if np.any(self.gamma):
            result += np.outer(self.gamma,self.gamma)*2*self.df**2/((self.df-2)**2*(self.df-4))
        return result

    def rvs(self,size=1,random_state=None):
        rng = np.random.default_rng(random_state)
        w = 1/rng.gamma(self.df/2,2/self.df,size=size)
        return self.location+w[:,None]*self.gamma+np.sqrt(w[:,None])*(rng.normal(size=(size,len(self.location)))@self.chol.T)


class GHSkewT(stats.rv_continuous):
    def _argcheck(self,df,b): return (df > 0) & np.isfinite(df) & np.isfinite(b)

    def _logpdf(self,x,df,b):
        x,df,b = np.broadcast_arrays(x,df,b)
        infinite=np.isinf(x); x=np.where(infinite,0.,x)
        with np.errstate(all='ignore'):
            value = df/2*np.log(df/2)-special.gammaln(df/2)-.5*np.log(2*np.pi)+b*x+log_kernel(df+x*x,b*b,(df+1)/2)
        return np.where(infinite,-np.inf,value)

    def _pdf(self,x,df,b): return np.exp(self._logpdf(x,df,b))

    def _cdf(self,x,df,b): return self._tail(x,df,b,False)
    def _sf(self,x,df,b): return self._tail(x,df,b,True)

    def _tail(self,x,df,b,survival):
        def scalar(v,nu,g):
            if g == 0: return special.stdtr(nu,-v if survival else v)
            # Integrate over log precision, centered at its density mode.
            shape = nu/2
            width = 1/np.sqrt(shape)
            constant = shape*np.log(shape)-special.gammaln(shape)
            def integrand(t):
                u = width*t
                if abs(u) > 700: return 0.
                precision = np.exp(u)
                z = v*np.sqrt(precision)-g/np.sqrt(precision)
                return np.exp(constant+shape*u-shape*precision)*special.ndtr(-z if survival else z)*width
            return checked_integral(integrand,-np.inf,0)+checked_integral(integrand,0,np.inf)
        return np.vectorize(scalar,otypes=[float])(x,df,b)

    def _stats(self,df,b):
        def scalar(nu,g):
            if g == 0: return stats.t.stats(nu,moments='mvsk')
            ew = {j: (nu/2)**j*np.exp(special.gammaln(nu/2-j)-special.gammaln(nu/2)) for j in range(1,5) if nu > 2*j}
            mean = g*ew[1] if nu > 2 else np.nan
            if nu <= 4: return mean,np.inf,np.nan,np.nan
            second = g*g*ew[2]+ew[1]
            variance = second-mean*mean
            third = g**3*ew[3]+3*g*ew[2] if nu > 6 else np.nan
            skew = (third-3*mean*second+2*mean**3)/variance**1.5
            fourth = g**4*ew[4]+6*g*g*ew[3]+3*ew[2] if nu > 8 else np.nan
            kurt = (fourth-4*mean*third+6*mean*mean*second-3*mean**4)/variance**2-3
            return mean,variance,skew,kurt
        return np.vectorize(scalar,otypes=[float]*4)(df,b)

    def _rvs(self,df,b,size=None,random_state=None):
        w = 1/random_state.gamma(df/2,2/df,size=size)
        return w*b+np.sqrt(w)*random_state.normal(size=size)


gh_skew_t = GHSkewT(name='gh_skew_t',shapes='df, b')


def fit_gh_skew_t(x,location=None,max_iterations=2000):
    """Standardized multi-start MLE; accepts validated complete-case input."""
    started=time.perf_counter(); n,d=x.shape
    center=x.mean(axis=0); unit=x.std(axis=0); z=(x-center)/unit
    fixed=None if location is None else (np.broadcast_to(location,(d,))-center)/unit
    nl=d if fixed is None else 0
    tri=np.tril_indices(d); diag=np.flatnonzero(tri[0]==tri[1]); nc=len(tri[0])
    def unpack(theta):
        mu=theta[:d] if fixed is None else fixed
        vals=theta[nl:nl+nc].copy(); vals[diag]=np.exp(vals[diag])
        chol=np.zeros((d,d)); chol[tri]=vals
        return mu,chol,theta[nl+nc:nl+nc+d],np.exp(theta[-1])
    def objective(theta):
        try:
            with np.errstate(all='ignore'):
                value=-float(logdensity(z,*unpack(theta)).sum())
        except (ValueError,ArithmeticError):
            return 1e100
        return value if np.isfinite(value) else 1e100
    bounds=[(None,None)]*nl+[(-10,10) if j in diag else (None,None) for j in range(nc)]+[(-20,20)]*d+[(np.log(.25),np.log(200.))]
    attempts=[]; candidates=[]
    for nu in (3.,8.,30.):
        for sign in (0.,1.,-1.):
            g=sign*.1*np.sign(stats.skew(z,axis=0))
            mu=-g*nu/(nu-2) if fixed is None else fixed
            cov=(z-mu).T@(z-mu)/n*(nu-2)/nu
            vals=np.linalg.cholesky(cov)[tri]; vals[diag]=np.log(vals[diag])
            initial=np.r_[mu if fixed is None else [],vals,g,np.log(nu)]
            result=optimize.minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':max_iterations,'maxfun':200000,'ftol':1e-10})
            attempts.append(dict(df_start=nu,skew_start=sign,converged=bool(result.success),message=str(result.message)))
            if np.isfinite(result.fun) and result.fun < 1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite GH skew-t fit')
    good=[r for r in candidates if r.success]; result=min(good or candidates,key=lambda r:r.fun)
    mu,chol,g,nu=unpack(result.x)
    scatter=(chol@chol.T)*np.outer(unit,unit)
    boundary=any((lo is not None and abs(v-lo)<1e-4) or (hi is not None and abs(v-hi)<1e-4) for v,(lo,hi) in zip(result.x,bounds))
    ll=-result.fun-n*np.log(unit).sum(); k=nl+nc+d+1
    record=dict(model='gh-skew-t',location=(center+unit*mu).tolist(),scatter=scatter.tolist(),gamma=(unit*g).tolist(),df=float(nu),observations=n,dimensions=d,parameters=k,loglik=float(ll),aic=float(2*k-2*ll),bic=float(np.log(n)*k-2*ll),converged=bool(result.success),boundary=boundary,status=('boundary' if boundary else 'ok') if result.success else 'not_converged',message=str(result.message),attempts=attempts,fit_sec=time.perf_counter()-started)
    frozen=JointGHSkewT(record); mean=frozen.mean(); covariance=frozen.cov()
    record.update(mean=None if mean is None else mean.tolist(),covariance=None if covariance is None else covariance.tolist(),scatter_correlation=(scatter/np.sqrt(np.outer(scatter.diagonal(),scatter.diagonal()))).tolist(),correlation=None if covariance is None else (covariance/np.sqrt(np.outer(covariance.diagonal(),covariance.diagonal()))).tolist())
    return record
