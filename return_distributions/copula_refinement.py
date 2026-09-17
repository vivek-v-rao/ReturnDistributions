"""Full joint likelihood refinement, initialized by a two-stage copula fit."""
import copy
import time
import warnings
import numpy as np
from scipy import optimize
from .fitting import specification
from .copulas import copula_logpdf


def refine_joint(x, record, max_iterations=1000):
    """Return a new record; failed refinements retain the original parameters.

    Families are fixed. Marginal t dfs and copula df remain distinct (this is
    not a restricted multivariate-t model). No CDF clipping in this likelihood.
    """
    started = time.perf_counter()
    if record['marginal_mode'] != 'fitted': raise ValueError('Joint refinement requires parametric marginals')
    if record['copula'].get('status') != 'ok': raise ValueError('Joint refinement requires a successful interior initial copula')
    if record.get('clipped_entries', 0): raise ValueError('Joint refinement requires an unclipped starting likelihood')
    if record['copula']['model'] not in {'gaussian', 'student-t'}:
        raise ValueError('Joint refinement currently supports only Gaussian and Student-t copulas')
    if max_iterations < 1: raise ValueError('Iteration limit must be positive')
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('Invalid refinement sample')
    n, d = x.shape
    if len(record['marginals']) != d or n < max(8,d+2): raise ValueError('Wrong marginal count or insufficient sample')
    center, unit = x.mean(axis=0), x.std(axis=0)
    if (unit <= 0).any(): raise ValueError('Constant asset')
    z = (x-center)/unit
    specs, initial, bounds = [], [], []
    for j, row in enumerate(record['marginals']):
        dist, names, fixed = specification(row['name'])
        if dist.name not in {'norm','t','laplace','gennorm','genhyperbolic','norminvgauss','fs_skew_t','azzalini_skew_t','jf_skew_t','nct'}:
            raise ValueError('Joint refinement unsupported for marginal '+row['name'])
        keys = [key for key in names if key not in fixed]
        offset = len(initial)
        for key in keys:
            value = float(row[key])
            bound = (None, None)
            if key == 'loc': value = (value-center[j])/unit[j]
            elif key == 'scale': value, bound = np.log(value/unit[j]), (-12, 12)
            elif key in {'df','beta'}:
                value, bound = np.log(value), ((np.log(.1), np.log(100000)) if key == 'df' else (np.log(.1),np.log(10)))
            elif dist.name == 'jf_skew_t' and key in {'a','b'}:
                value,bound=np.log(value),(np.log(.05),np.log(50000))
            elif key == 'skewness': bound = (-8,8) if dist.name == 'fs_skew_t' else (-100,100)
            elif key == 'nc': bound = (-100,100)
            elif key == 'a':
                value, bound = np.log(value*value-float(row['b'])**2), (-24,24)
            elif key == 'p': bound = (-20,20)
            initial.append(value)
            bounds.append(bound)
        specs.append((dist, names, fixed, keys, offset))
    cstart = len(initial)
    indices = np.tril_indices(d,-1)
    chol = np.linalg.cholesky(record['copula']['correlation'])
    initial.extend((chol/chol.diagonal()[:,None])[indices])
    bounds.extend([(-20,20)]*len(indices[0]))
    copula = record['copula']['model']
    if copula == 'student-t':
        initial.append(np.log(record['copula']['df']))
        bounds.append((np.log(.25),np.log(200)))
    initial = np.array(initial)
    if any((a is not None and v < a) or (b is not None and v > b) for v,(a,b) in zip(initial,bounds)):
        raise ValueError('Initial fit is outside joint-refinement bounds')

    def unpack(theta):
        params = []
        for dist, names, fixed, keys, offset in specs:
            values = dict(fixed)
            values.update(zip(keys, theta[offset:offset+len(keys)]))
            for key in ['scale','df','beta']:
                if key in values: values[key] = np.exp(values[key])
            if dist.name == 'jf_skew_t':
                values['a'],values['b']=np.exp(values['a']),np.exp(values['b'])
            elif 'a' in values: values['a'] = np.sqrt(np.exp(values['a'])+values['b']**2)
            params.append([values[key] for key in names])
        lower = np.eye(d)
        lower[indices] = theta[cstart:cstart+len(indices[0])]
        lower /= np.linalg.norm(lower,axis=1)[:,None]
        return params, lower@lower.T, (np.exp(theta[-1]) if copula == 'student-t' else None)

    def evaluate(theta):
        params, corr, df = unpack(theta)
        cdfs, marginal_ll = [], []
        for j, (dist, *_) in enumerate(specs):
            cdfs.append(dist.cdf(z[:,j], *params[j]))
            marginal_ll.append(float(dist.logpdf(z[:,j], *params[j]).sum())-n*np.log(unit[j]))
        u = np.column_stack(cdfs)
        if not np.isfinite(marginal_ll).all(): raise ValueError('Invalid marginal density')
        cll = float(copula_logpdf(u, copula, corr, df).sum())
        return sum(marginal_ll)+cll, marginal_ll, cll

    def objective(theta):
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            try:
                ll = evaluate(theta)[0]
                return -ll if np.isfinite(ll) else 1e100
            except (ValueError, np.linalg.LinAlgError, FloatingPointError): return 1e100

    baseline = evaluate(initial)[0]
    result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter':max_iterations,'maxfun':200000,'ftol':1e-11,'gtol':1e-6})
    candidate_ll = -float(result.fun)
    boundary = any((a is not None and abs(v-a)<1e-4) or (b is not None and abs(v-b)<1e-4)
                   for v,(a,b) in zip(result.x,bounds))
    accepted = bool(result.success and not boundary and np.isfinite(candidate_ll) and candidate_ll >= baseline)
    output = copy.deepcopy(record)
    output['stage'] = 'joint-refined' if accepted else 'two-stage-retained'
    output['refinement'] = dict(accepted=accepted, converged=bool(result.success), boundary=boundary,
        message=str(result.message), iterations=int(result.nit), baseline_loglik=baseline,
        candidate_loglik=candidate_ll, improvement=candidate_ll-baseline if accepted else 0.,
        fit_sec=time.perf_counter()-started)
    if accepted:
        params, corr, df = unpack(result.x)
        ll, marginal_ll, cll = evaluate(result.x)
        for j, (dist,names,fixed,keys,offset) in enumerate(specs):
            values = params[j].copy()
            values[-2] = center[j]+unit[j]*values[-2]
            values[-1] *= unit[j]
            old = record['marginals'][j]
            output['marginals'][j] = dict(name=old['name'], symbol=old.get('symbol'),
                k=len(keys), n=n, status='ok', stage='joint-refined', loglik=marginal_ll[j], **dict(zip(names,map(float,values))))
        output['copula'].update(correlation=corr.tolist(), df=None if df is None else float(df), loglik=cll,
                                 stage='joint-refined', aic=2*record['copula']['parameters']-2*cll,
                                 bic=np.log(n)*record['copula']['parameters']-2*cll)
    else:
        ll, marginal_ll, cll = evaluate(initial)
    k = len(initial)
    summary = output.setdefault('summary', {})
    for key in ['two_stage_aic','two_stage_bic']: summary.pop(key,None)
    summary.update(stage=output['stage'], refinement_accepted=accepted,
        refinement_converged=bool(result.success), refinement_boundary=boundary,
        refinement_seconds=time.perf_counter()-started, likelihood_improvement=candidate_ll-baseline if accepted else 0.,
        joint_loglik=ll, marginal_loglik=sum(marginal_ll), copula_loglik=cll,
        copula_df=output['copula']['df'], total_parameters=k,
        joint_aic=2*k-2*ll if accepted else None, joint_bic=np.log(n)*k-2*ll if accepted else None)
    # Copula-only IC is not comparable after the marginal transforms change.
    for key in ['copula_aic','copula_bic']: summary.pop(key,None)
    return output
