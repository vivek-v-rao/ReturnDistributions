"""Location-outside slash mixtures: X=mu+Y/U**(1/q), U independent uniform.

Y is zero-location normal/t or Azzalini skew-normal/t. Derived directly from
the mixture representation in Tan, Tang & Peng (2015), doi:10.1186/s40488-015-0025-9;
the paper's printed CDF/moment formulas are not used.
"""
from functools import lru_cache
import time
import numpy as np
from scipy import integrate, linalg, optimize, special, stats
from .skew_t import azzalini_skew_t

SLASH_MODELS = ('slash-normal','skew-slash-normal','slash-t','skew-slash-t')


@lru_cache(maxsize=12)
def nodes(points,beta):
    """Normalized Gauss-Jacobi rule for v**beta on [0,1].

Golub-Welsch avoids the overflowing 2**beta normalization in roots_jacobi
when the fitted slash parameter is large.
"""
    j=np.arange(points,dtype=float)
    diagonal=beta*beta/((2*j+beta)*(2*j+beta+2))
    diagonal[0]=beta/(beta+2)
    j=np.arange(1,points,dtype=float)
    off=2*j*(j+beta)/((2*j+beta)*np.sqrt((2*j+beta)**2-1))
    x,vectors=linalg.eigh_tridiagonal(diagonal,off)
    weights=vectors[0]**2
    keep=weights>0
    return (x[keep]+1)/2,weights[keep]/weights[keep].sum()


def inverse_moment(q, order):
    return 1. if np.isinf(q) else q/(q-order)


def logdensity(x, mu, chol, a, df, q, points=64):
    """q/(q+d) E[f_Y(V*(x-mu))], V~Beta(q+d,1).

    Gauss-Jacobi integrates the exact power weight; fitting audits a 4x grid.
"""
    y=linalg.solve_triangular(chol,(np.atleast_2d(x)-mu).T,lower=True).T
    d=len(mu); radius=np.sum(y*y,axis=1); slant=y@a
    v,w=(np.ones(1),np.ones(1)) if np.isinf(q) else nodes(points,q+d-1)
    out=np.empty(len(y))
    for start in range(0,len(y),128):
        r=radius[start:start+128,None]*v*v
        s=slant[start:start+128,None]*v
        if df is None:
            lp=-d/2*np.log(2*np.pi)-r/2
            if np.any(a):lp+=np.log(2)+special.log_ndtr(s)
        else:
            lp=special.gammaln((df+d)/2)-special.gammaln(df/2)-d/2*np.log(df*np.pi)-(df+d)/2*np.log1p(r/df)
            if np.any(a):lp+=np.log(2)+stats.t.logcdf(s*np.sqrt((df+d)/(df+r)),df+d)
        out[start:start+len(r)]=special.logsumexp(lp+np.log(w),axis=1)-np.log(chol.diagonal()).sum()
    return out if np.isinf(q) else out+np.log(q/(q+d))


class JointSlash:
    def __init__(self,fit):
        self.location=np.asarray(fit['location'],dtype=float)
        self.scatter=np.asarray(fit['scatter'],dtype=float)
        self.q=float(fit['q']);self.df=None if fit['model'].endswith('normal') else float(fit['df'])
        self.alpha=np.asarray(fit.get('alpha',np.zeros(len(self.location))),dtype=float)
        self.points=fit.get('quadrature_points',256)
        if not isinstance(self.points,(int,np.integer)) or not 1<=self.points<=4096:
            raise ValueError('Invalid saved slash quadrature order')
        d=len(self.location)
        if (self.scatter.shape!=(d,d) or self.alpha.shape!=(d,) or not np.isfinite(self.location).all()
                or not np.isfinite(self.scatter).all() or not np.isfinite(self.alpha).all()
                or not np.allclose(self.scatter,self.scatter.T) or not self.q>0
                or (self.df is not None and (not np.isfinite(self.df) or self.df<=0))):
            raise ValueError('Invalid slash parameters')
        self.chol=np.linalg.cholesky(self.scatter)
        self.a=self.chol.T@(self.alpha/np.sqrt(self.scatter.diagonal()))
        self.delta=self.chol@self.a/np.sqrt(1+self.a@self.a)

    def logpdf(self,x):
        lp=logdensity(x,self.location,self.chol,self.a,self.df,self.q,self.points)
        return lp[0] if np.asarray(x).ndim==1 else lp

    def pdf(self,x):return np.exp(self.logpdf(x))

    def base_mean(self):
        b=np.sqrt(2/np.pi) if self.df is None else np.sqrt(self.df/np.pi)/special.poch((self.df-1)/2,.5)
        return b*self.delta

    def mean(self):
        if self.q<=1 or (self.df is not None and self.df<=1):return np.full_like(self.location,np.nan)
        return self.location+inverse_moment(self.q,1)*self.base_mean()

    def cov(self):
        if self.q<=2 or (self.df is not None and self.df<=2):return np.full_like(self.scatter,np.nan)
        factor=1. if self.df is None else self.df/(self.df-2)
        shift=self.mean()-self.location
        return inverse_moment(self.q,2)*factor*self.scatter-np.outer(shift,shift)

    def rvs(self,size=1,random_state=None):
        rng=np.random.default_rng(random_state)
        z=rng.normal(size=(size,len(self.location)))@np.linalg.cholesky(self.scatter-np.outer(self.delta,self.delta)).T
        z+=np.abs(rng.normal(size=size))[:,None]*self.delta
        if self.df is not None:z/=np.sqrt(rng.chisquare(self.df,size=size)/self.df)[:,None]
        if not np.isinf(self.q):z*=np.exp(rng.exponential(size=size)/self.q)[:,None]
        return self.location+z

    def project(self,w):
        scale=np.sqrt(w@self.scatter@w);delta=float(w@self.delta)/scale
        shape=delta/np.sqrt(1-delta*delta)
        base=(stats.norm() if shape==0 else stats.skewnorm(shape)) if self.df is None else (stats.t(self.df) if shape==0 else azzalini_skew_t(self.df,shape))
        return ProjectedSlash(float(w@self.location),scale,base,self.q,self.df)


def quad_checked(fun,lo,hi):
    value,error=integrate.quad(fun,lo,hi,epsabs=1e-10,epsrel=1e-8,limit=250)
    if not np.isfinite(value) or error>max(1e-8,abs(value)*1e-6):raise ValueError('Slash quadrature failed accuracy check')
    return value


class ProjectedSlash:
    """Exact mixture family, evaluated by adaptive one-dimensional quadrature."""
    def __init__(self,location,scale,base,q,df):
        self.location,self.scale,self.base,self.q,self.df=location,scale,base,q,df

    def mean(self):
        if self.q<=1 or (self.df is not None and self.df<=1):return np.nan
        return self.location+self.scale*inverse_moment(self.q,1)*self.base.mean()

    def var(self):
        if self.q<=2 or (self.df is not None and self.df<=2):return np.nan
        m=self.base.mean()
        return self.scale**2*(inverse_moment(self.q,2)*(self.base.var()+m*m)-inverse_moment(self.q,1)**2*m*m)

    def std(self):return np.sqrt(self.var())

    def _prob(self,x,tail):
        def one(value):
            z=(value-self.location)/self.scale
            fn=self.base.sf if tail else self.base.cdf
            if not np.isfinite(z) or np.isinf(self.q):return float(fn(z))
            return quad_checked(lambda u:float(fn(z*u**(1/self.q))),0,1)
        return np.vectorize(one,otypes=[float])(x)

    def cdf(self,x):return self._prob(x,False)
    def sf(self,x):return self._prob(x,True)

    def ppf(self,p):
        def one(prob):
            if not 0<=prob<=1:return np.nan
            if prob==0:return -np.inf
            if prob==1:return np.inf
            # Solve in standardized units; evaluate the smaller tail directly.
            unit=ProjectedSlash(0.,1.,self.base,self.q,self.df)
            fn=(lambda z:float(unit.cdf(z))-prob) if prob<=.5 else (lambda z:1-prob-float(unit.sf(z)))
            lo,hi=-1.,1.
            for _ in range(512):
                if fn(lo)<=0 and fn(hi)>=0:break
                lo*=2;hi*=2
            else:raise ValueError('Cannot bracket slash quantile')
            return self.location+self.scale*optimize.brentq(fn,lo,hi,xtol=1e-10)
        return np.vectorize(one,otypes=[float])(p)

    def pdf(self,x):
        def one(value):
            z=(value-self.location)/self.scale
            if np.isnan(z):return np.nan
            if np.isinf(z):return 0.
            if np.isinf(self.q):return float(self.base.pdf(z))/self.scale
            # Integrate log v around its mode so extreme tails cannot be missed.
            # t=(q+1)*(-log v) keeps the kernel resolved as q tends to infinity.
            def kernel(s):return np.log(self.q/(self.q+1))-s+float(self.base.logpdf(z*np.exp(-s/(self.q+1))))
            bound=(self.q+1)*max(5.,np.log1p(abs(z))+5.)
            mode=optimize.minimize_scalar(lambda s:-kernel(s),bounds=(0,bound),method='bounded').x
            if kernel(0)>kernel(mode):mode=0.
            peak=kernel(mode)
            f=lambda s:np.exp(kernel(s)-peak)
            value=quad_checked(f,0,mode)+quad_checked(f,mode,np.inf)
            return np.exp(peak)*value/self.scale
        return np.vectorize(one,otypes=[float])(x)

    def rvs(self,size=1,random_state=None):
        rng=np.random.default_rng(random_state);y=self.base.rvs(size=size,random_state=rng)
        if not np.isinf(self.q):y*=np.exp(rng.exponential(size=size)/self.q)
        return self.location+self.scale*y

    def risk(self,confidence):
        threshold=float(self.ppf(1-confidence))
        if not np.isfinite(threshold):raise ValueError('Nonfinite slash risk quantile')
        if self.q<=1 or (self.df is not None and self.df<=1):
            return dict(var=-threshold,es=np.inf,es_status='infinite (q <= 1 or df <= 1)',es_method='slash-moment-existence')
        z=(threshold-self.location)/self.scale
        first=quad_checked(lambda y:y*float(self.base.pdf(y)),-np.inf,min(0.,z))
        if z>0:first+=quad_checked(lambda y:y*float(self.base.pdf(y)),0,z)
        correction=0. if np.isinf(self.q) else z*z/(self.q+1)*quad_checked(lambda t:float(self.base.pdf(z*t**(1/(self.q+1)))),0,1)
        partial=inverse_moment(self.q,1)*(first-correction)
        es=-self.location-self.scale*partial/(1-confidence)
        if not np.isfinite(es):raise ValueError('Numerical slash ES is nonfinite')
        return dict(var=-threshold,es=float(es),es_status='finite',es_method='slash-truncated-moment-quadrature')


def fit_slash(x,model,location=None,max_iterations=2000,points=64):
    from .multivariate import fit_joint
    if not isinstance(points,(int,np.integer)) or not 16<=points<=256:raise ValueError('Slash fitting quadrature points must be an integer in [16,256]')
    started=time.perf_counter();n,d=x.shape;normal=model.endswith('normal');skew=model.startswith('skew-')
    center,unit=x.mean(0),x.std(0);z=(x-center)/unit
    fixed=None if location is None else (np.broadcast_to(location,(d,))-center)/unit
    nloc=d if fixed is None else 0;ix=np.tril_indices(d);diagonal=np.flatnonzero(ix[0]==ix[1]);nc=len(ix[0])
    # Symmetric base fit seeds location/scatter; skew starts explore both signs.
    nested=fit_joint(z,'normal' if normal else 'student-t',location=fixed,max_iterations=max_iterations)
    df0=None if normal else np.clip(nested['df'],.3,1000.)
    def pack(mu,sigma,a,q):
        v=np.linalg.cholesky(sigma)[ix];v[diagonal]=np.log(v[diagonal])
        return np.r_[mu if fixed is None else [],v,a if skew else [],np.log(q),[] if normal else [np.log(df0)]]
    def unpack(theta):
        mu=theta[:d] if fixed is None else fixed;v=theta[nloc:nloc+nc].copy();v[diagonal]=np.exp(v[diagonal])
        chol=np.zeros((d,d));chol[ix]=v;j=nloc+nc+(d if skew else 0)
        return mu,chol,theta[nloc+nc:j] if skew else np.zeros(d),None if normal else np.exp(theta[-1]),np.exp(theta[j])
    def objective(theta):
        with np.errstate(all='ignore'):value=-logdensity(z,*unpack(theta),points).sum()
        return float(value) if np.isfinite(value) else 1e100
    bounds=[(None,None)]*(nloc+nc)+([(-15.,15.)]*d if skew else [])+[(np.log(.3),np.log(10000.))]+([] if normal else [(np.log(.3),np.log(1000.))])
    for j in diagonal:bounds[nloc+j]=(-10.,10.)
    direction=np.where(stats.skew(z,axis=0)>=0,1.,-1.)
    starts=[(3.,np.zeros(d)),(15.,direction if skew else np.zeros(d)),(100.,-direction if skew else np.zeros(d))]
    candidates,attempts=[],[]
    for q,a in starts:
        sigma=np.asarray(nested['scatter'])/inverse_moment(q,2)
        initial=pack(np.asarray(nested['location']),sigma,a,q)
        result=optimize.minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=max_iterations,maxfun=200000,ftol=1e-10,gtol=1e-5))
        attempts.append(dict(q_start=q,shape_start=a.tolist(),converged=bool(result.success),message=str(result.message),negative_loglik=float(result.fun)))
        if np.isfinite(result.fun) and result.fun<1e99:candidates.append(result)
    if not candidates:raise ValueError('No finite slash fit')
    result=min([r for r in candidates if r.success] or candidates,key=lambda r:r.fun)
    mu,chol,a,df,q=unpack(result.x)
    lp=logdensity(z,mu,chol,a,df,q,points);audit=logdensity(z,mu,chol,a,df,q,points*4)
    ll_change=float(abs((audit-lp).sum()));max_change=float(np.max(abs(audit-lp)))
    stable=ll_change<=.1 and max_change<=.02
    boundary=any((lo is not None and abs(v-lo)<1e-4) or (hi is not None and abs(v-hi)<1e-4) for v,(lo,hi) in zip(result.x,bounds))
    status='not_converged' if not result.success else ('boundary' if boundary else ('ok' if stable else 'quadrature_unstable'))
    sigma=chol@chol.T;alpha=np.sqrt(sigma.diagonal())*linalg.solve_triangular(chol.T,a,lower=False)
    ll=float(audit.sum()-n*np.log(unit).sum());k=nloc+nc+(d if skew else 0)+1+int(not normal)
    fit=dict(model=model,location=(center+unit*mu).tolist(),scatter=(sigma*np.outer(unit,unit)).tolist(),alpha=alpha.tolist(),q=float(q),df=None if df is None else float(df),
             observations=n,dimensions=d,parameters=k,loglik=ll,aic=2*k-2*ll,bic=np.log(n)*k-2*ll,status=status,converged=bool(result.success),boundary=boundary,
             quadrature_points=points*4,quadrature_fit_points=points,quadrature_audit_passed=bool(stable),quadrature_loglik_change=ll_change,quadrature_max_logpdf_change=max_change,
             attempts=attempts,message=str(result.message))
    frozen=JointSlash(fit);mean=frozen.mean();cov=frozen.cov();finite_cov=np.isfinite(cov).all()
    fit.update(mean=mean.tolist() if np.isfinite(mean).all() else None,covariance=cov.tolist() if finite_cov else None,
               correlation=(cov/np.sqrt(np.outer(cov.diagonal(),cov.diagonal()))).tolist() if finite_cov else None,
               scatter_correlation=(sigma/np.sqrt(np.outer(sigma.diagonal(),sigma.diagonal()))).tolist(),fit_sec=time.perf_counter()-started)
    return fit
