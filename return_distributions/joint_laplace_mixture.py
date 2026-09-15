"""Fixed-origin normal/exponential Laplace mixtures, distinct from power Laplace.

X = location + W*gamma + sqrt(W)*Z, W~Exp(1), Z~N(0,scatter).
This is the VG shape=1 distribution. Its density is singular at location for
d>=2, so location is never optimized using the Laplace likelihood. Optional
median/KDE-mode pilots produce explicitly labeled two-stage estimates.
"""
import time
import numpy as np
from scipy import optimize, special, stats, linalg
from .variance_gamma import vg_logpdf, JointVG

LAPLACE_MIXTURE_MODELS=('laplace-mixture-symmetric','asymmetric-laplace')


def mode_bandwidth(value):
    if value in ('scott','silverman'): return value
    try: result=float(value)
    except (ValueError,TypeError): raise ValueError('Mode bandwidth must be scott, silverman, or a positive KDE factor')
    if not np.isfinite(result) or result<=0:
        raise ValueError('Mode bandwidth must be finite and positive')
    return result


def pilot_location(x, method, bandwidth='scott', max_iterations=2000):
    """Full-sample Gaussian KDE joint mode, not a vector of marginal modes."""
    if method=='median': return np.median(x,axis=0),dict(pilot_method='componentwise sample medians')
    if method!='mode': raise ValueError('Unknown Laplace location method')
    bandwidth=mode_bandwidth(bandwidth)
    n,d=x.shape; center=x.mean(axis=0); unit=x.std(axis=0)
    z=(x-center)/unit
    kde=stats.gaussian_kde(z.T,bw_method=bandwidth)
    chol=np.linalg.cholesky(kde.covariance)
    points=linalg.solve_triangular(chol,z.T,lower=True).T

    def objective(v):
        delta=points-v
        logs=-.5*np.sum(delta*delta,axis=1)
        normalizer=special.logsumexp(logs)
        gradient=-np.exp(logs-normalizer)@delta
        return -float(normalizer),gradient

    indices=np.unique(np.linspace(0,n-1,min(n,200),dtype=int))
    dense=sorted(indices,key=lambda j:objective(points[j])[0])[:4]
    spread=np.unique(np.linspace(0,n-1,min(n,6),dtype=int))
    starts=[points.mean(axis=0),linalg.solve_triangular(chol,np.median(z,axis=0),lower=True)]
    starts += [points[j] for j in dict.fromkeys([*dense,*spread])]
    runs=[]; attempts=[]
    for i,start in enumerate(starts):
        r=optimize.minimize(objective,start,jac=True,method='L-BFGS-B',
            options={'maxiter':max_iterations,'ftol':1e-12,'gtol':1e-7})
        attempts.append(dict(start=i,converged=bool(r.success),message=str(r.message)))
        if r.success and np.isfinite(r.fun) and np.isfinite(r.x).all(): runs.append(r)
    if not runs: raise ValueError('No converged joint KDE-mode pilot; increase iterations or change bandwidth')
    best=min(runs,key=lambda r:r.fun)
    location=center+unit*(chol@best.x)
    logdensity=-best.fun-np.log(n)-d/2*np.log(2*np.pi)-np.log(chol.diagonal()).sum()-np.log(unit).sum()
    return location,dict(pilot_method='joint Gaussian KDE mode',pilot_bandwidth=bandwidth,
        pilot_bandwidth_factor=float(kde.factor),pilot_log_density=float(logdensity),pilot_attempts=attempts)


class JointLaplaceMixture(JointVG):
    def __init__(self, fit):
        if fit.get('vg_shape',1.)!=1.:
            raise ValueError('Laplace mixture requires exponential mixing (VG shape=1)')
        super().__init__(dict(fit,vg_shape=1.))


def fit_laplace_mixture(x, model, location=None, max_iterations=2000,
                        location_method=None, bandwidth=None):
    """Validated input from fit_joint; fixed-location, bounded multistart MLE."""
    started=time.perf_counter()
    n,d=x.shape
    symmetric=model=='laplace-mixture-symmetric'
    if location_method is not None and location is not None:
        raise ValueError('Laplace location method cannot be combined with an explicit location')
    if location_method not in (None,'zero','median','mode'):
        raise ValueError('Laplace location method must be zero, median, or mode')
    if bandwidth is not None and location_method!='mode':
        raise ValueError('Mode bandwidth requires mode location')
    selected_method=location_method or ('zero' if location is None else 'provided')
    two_stage=selected_method in ('median','mode')
    pilot={}
    if two_stage:
        location,pilot=pilot_location(x,selected_method,'scott' if bandwidth is None else bandwidth,max_iterations)
    mu=np.zeros(d) if location is None else np.broadcast_to(np.asarray(location,dtype=float),(d,)).copy()
    if not np.isfinite(mu).all(): raise ValueError('Fixed location must be finite')
    unit=x.std(axis=0)
    if d>=2 and np.any(np.all(np.abs((x-mu)/unit)<=64*np.finfo(float).eps,axis=1)):
        raise ValueError('An observation equals the frozen Laplace-mixture location: density is singular; choose another location method or an externally specified location, not a density floor')
    y=(x-mu)/unit
    origin=np.zeros(d)
    indices=np.tril_indices(d)
    diagonal=np.flatnonzero(indices[0]==indices[1])
    nc=len(indices[0])
    bounds=[(-20.,20.)]*nc+([] if symmetric else [(-20.,20.)]*d)
    for j in diagonal: bounds[j]=(-8.,8.)

    def pack(scatter,gamma):
        values=np.linalg.cholesky(scatter)[indices]
        values[diagonal]=np.log(values[diagonal])
        return np.r_[values,[] if symmetric else gamma]

    def unpack(v):
        values=v[:nc].copy(); values[diagonal]=np.exp(values[diagonal])
        chol=np.zeros((d,d)); chol[indices]=values
        return chol,np.zeros(d) if symmetric else v[nc:]

    def objective(v):
        chol,gamma=unpack(v)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            value=-float(vg_logpdf(y,origin,chol,gamma,1.).sum())
        return value if np.isfinite(value) else 1e100

    base=y.T@y/n
    candidates=[]; attempts=[]
    for multiplier,fraction in [(1.,0.),(.5,.5),(1.,1.)]:
        gamma=np.clip(y.mean(axis=0)*fraction,-19.,19.)
        start=pack(base*multiplier,gamma)
        start=np.clip(start,[lo+1e-5 for lo,hi in bounds],[hi-1e-5 for lo,hi in bounds])
        result=optimize.minimize(objective,start,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':max_iterations,'ftol':1e-11,'gtol':1e-6,'maxfun':200000})
        attempts.append(dict(scatter_start=multiplier,skew_start_fraction=fraction,
                             converged=bool(result.success),iterations=int(result.nit),message=str(result.message)))
        if np.isfinite(result.fun) and result.fun<1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite fixed-location Laplace-mixture fit')
    successful=[r for r in candidates if r.success]
    result=min(successful or candidates,key=lambda r:r.fun)
    chol,gamma=unpack(result.x)
    scatter=(chol@chol.T)*np.outer(unit,unit); gamma=gamma*unit
    covariance=scatter+np.outer(gamma,gamma)
    mean=mu+gamma
    if not np.isfinite(covariance).all(): raise ValueError('Nonfinite Laplace-mixture covariance')
    boundary=any(min(abs(v-lo),abs(v-hi))<1e-4 for v,(lo,hi) in zip(result.x,bounds))
    ll=-float(result.fun)-n*np.log(unit).sum()
    k=nc+(0 if symmetric else d)
    correlation=covariance/np.sqrt(np.outer(covariance.diagonal(),covariance.diagonal()))
    scatter_corr=scatter/np.sqrt(np.outer(scatter.diagonal(),scatter.diagonal()))
    total_k=k+(d if two_stage else 0)
    return dict(model=model,observations=n,dimensions=d,parameters=total_k,conditional_parameters=k,loglik=float(ll),
        aic=None if two_stage else float(2*k-2*ll),bic=None if two_stage else float(np.log(n)*k-2*ll),
        two_stage_aic=float(2*total_k-2*ll) if two_stage else None,
        two_stage_bic=float(np.log(n)*total_k-2*ll) if two_stage else None,location=mu.tolist(),
        gamma=gamma.tolist(),scatter=scatter.tolist(),covariance=covariance.tolist(),
        mean=mean.tolist(),correlation=correlation.tolist(),scatter_correlation=scatter_corr.tolist(),
        vg_shape=1.,location_fixed=True,location_method=selected_method,
        location_source=('sample-'+selected_method if two_stage else ('default-zero' if location is None else 'provided')),
        estimation_method='two-stage' if two_stage else 'fixed-location-mle',
        fit_label=f'{model} [{selected_method}]',
        information_criteria_note='Descriptive plug-in scores counting d pilot coordinates; not jointly maximized likelihood, KDE complexity/bandwidth selection not accounted for' if two_stage else 'Location fixed externally; no fitted location parameters',
        fit_restriction='Location frozen during Laplace optimization; standardized Cholesky log diagonals [-8,8], off diagonals and gamma [-20,20]; location collisions within 64 machine eps in SD units rejected for dimension>=2',
        converged=bool(result.success),boundary=boundary,
        status='boundary' if result.success and boundary else ('ok' if result.success else 'not_converged'),
        message=str(result.message),attempts=attempts,fit_sec=time.perf_counter()-started,**pilot)
