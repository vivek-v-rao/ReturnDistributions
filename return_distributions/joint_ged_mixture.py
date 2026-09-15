"""Direct finite GED/NIG mixtures with a common power/psi and full scatter.

Direct bounded likelihood optimization; inputs validated by fit_finite_mixture.
Scatter=L L' + floor I in sample-SD coordinates. Weight floors are imposed
through a shifted simplex. Component labels have no statistical significance.
"""
import time
import numpy as np
from scipy import optimize, special
from .joint_power import power_logpdf, covariance_factor
from .joint_gh import gh_logpdf, mixture_moments


def fit_ged_mixture(x, components, *, baseline=None, location=None, max_iterations=2000,
                    starts=5, seed=12345, min_weight=.01, eigen_floor=1e-4, model='ged'):
    """Shared numerical fitter; retained name for the original GED entry point."""
    if model not in ('ged', 'nig-skewed'): raise ValueError('Unsupported direct mixture family')
    nig = model == 'nig-skewed'
    started = time.perf_counter()
    n, d = x.shape
    center, unit = x.mean(0), x.std(0)
    z = (x-center)/unit
    fixed = None if location is None else (np.broadcast_to(location, (d,))-center)/unit
    indices = np.tril_indices(d)
    diagonal = np.flatnonzero(indices[0] == indices[1])
    nc, nloc = len(indices[0]), d if fixed is None else 0
    width = nloc+nc+(d if nig else 0)

    def pack(means, scatters, power, gammas):
        theta = []
        for mu, scatter, gamma in zip(means, scatters, gammas):
            values, vectors = np.linalg.eigh(scatter-eigen_floor*np.eye(d))
            excess = (vectors*np.maximum(values, eigen_floor*.01))@vectors.T
            values = np.linalg.cholesky(excess)[indices]
            values[diagonal] = np.log(values[diagonal])
            theta.extend(np.r_[mu if fixed is None else [], values, gamma if nig else []])
        return np.r_[theta, np.zeros(components-1), np.log(power)]

    def unpack(theta):
        means, scatters, chols, gammas = [], [], [], []
        for j in range(components):
            block = theta[j*width:(j+1)*width]
            mu = block[:d] if fixed is None else fixed
            values = block[nloc:nloc+nc].copy()
            values[diagonal] = np.exp(values[diagonal])
            chol = np.zeros((d, d))
            chol[indices] = values
            scatter = chol@chol.T+eigen_floor*np.eye(d)
            means.append(mu)
            scatters.append(scatter)
            chols.append(np.linalg.cholesky(scatter))
            gammas.append(block[nloc+nc:] if nig else np.zeros(d))
        weights = min_weight+(1-components*min_weight)*special.softmax(np.r_[theta[components*width:-1], 0.])
        return weights, np.array(means), np.array(scatters), chols, float(np.exp(theta[-1])), np.array(gammas)

    def objective(theta):
        with np.errstate(all='ignore'):
            try:
                weights, means, scatters, chols, power, gammas = unpack(theta)
                densities = np.array([gh_logpdf(z, mu, chol, gamma, -.5, power) if nig else power_logpdf(z, mu, chol, power)
                                      for mu, chol, gamma in zip(means, chols, gammas)]).T
                value = -float(special.logsumexp(densities+np.log(weights), axis=1).sum())
                return value if np.isfinite(value) else 1e100
            except np.linalg.LinAlgError:
                return 1e100

    bounds = []
    for _ in range(components):
        block = [(None, None)]*width
        for j in diagonal: block[nloc+j] = (-12., 12.)
        if nig: block[nloc+nc:] = [(-10., 10.)]*d
        bounds.extend(block)
    bounds.extend([(-20., 20.)]*(components-1)+[(-12., 12.) if nig else (np.log(.25), np.log(10.))])
    mu0 = np.zeros(d) if fixed is None else fixed
    power0 = 2. if nig else 1.
    factor = lambda shape: mixture_moments(-.5, shape)[0] if nig else covariance_factor(d, shape)
    scatter0 = (z-mu0).T@(z-mu0)/n/factor(power0)
    gamma0 = np.zeros(d)
    if baseline and 'scatter' in baseline:
        mu0 = (np.asarray(baseline['location'])-center)/unit if fixed is None else fixed
        power0 = float(np.clip(baseline['psi'], np.exp(-12), np.exp(12))) if nig else float(np.clip(baseline['power'], .25, 10.))
        scatter0 = np.asarray(baseline['scatter'])/np.outer(unit, unit)
        if nig: gamma0 = np.clip(np.asarray(baseline['gamma'])/unit, -10., 10.)
    rng = np.random.default_rng(seed)
    attempts, candidates = [], []
    for start in range(starts):
        if start == 0:
            means = np.tile(mu0, (components, 1))
            scatters = np.tile(scatter0, (components, 1, 1))
            power = power0
            gammas = np.tile(gamma0, (components, 1))
        else:
            direction = np.linalg.eigh(scatter0)[1][:, -1] if start == 2 else rng.normal(size=d)
            score = ((z-mu0)**2).sum(1) if start == 1 else z@direction
            groups = np.array_split(np.argsort(score), components)
            means = np.array([z[g].mean(0) if fixed is None else fixed for g in groups])
            power = power0 if start < 3 else (1. if start % 2 else 2.)
            gammas = np.zeros((components, d))
            scatters = np.array([(z[g]-mu).T@(z[g]-mu)/len(g)/factor(power) for g, mu in zip(groups, means)])
            if nig and start >= 3:
                gammas = np.tile(gamma0, (components, 1))+.1*rng.normal(size=(components, d))
                gammas = np.clip(gammas, -10., 10.)
                if fixed is None: means -= factor(power)*gammas
        initial = pack(means, scatters, power, gammas)
        result = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter': max_iterations, 'maxfun': 200000, 'ftol': 1e-11, 'gtol': 1e-6})
        attempts.append(dict(start=start, converged=bool(result.success), iterations=int(result.nit),
                             loglik=float(-result.fun-n*np.log(unit).sum()), message=str(result.message)))
        if np.isfinite(result.fun) and result.fun < 1e99: candidates.append(result)
    if not candidates: raise ValueError('No finite '+model+' mixture fit')
    result = min(candidates, key=lambda r: r.fun)
    weights, means, scatters, _, power, gammas = unpack(result.x)
    boundary = (np.min(weights) <= min_weight+1e-6
                or np.min(np.linalg.eigvalsh(scatters)) <= eigen_floor*(1+1e-4)
                or any((lo is not None and abs(v-lo) < 1e-4) or (hi is not None and abs(v-hi) < 1e-4)
                       for v, (lo, hi) in zip(result.x, bounds)))
    means = center+means*unit
    scatters *= np.outer(unit, unit)
    gammas *= unit
    order = np.argsort(-weights, kind='stable')
    weights, means, scatters = weights[order], means[order], scatters[order]
    gammas = gammas[order]
    ll = -float(result.fun)-n*np.log(unit).sum()
    k = components*width+components  # K-1 weights plus one shared power/psi.
    output = dict(model=model, family='finite-mixture', components=components, power=power, df=None,
                  mixture_weights=weights.tolist(), component_fits=[dict(model=model, location=mu.tolist(),
                      scatter=s.tolist(), power=power, df=None) for mu, s in zip(means, scatters)],
                  observations=n, dimensions=d, parameters=k, loglik=float(ll), aic=float(2*k-2*ll), bic=float(np.log(n)*k-2*ll),
                  converged=bool(result.success), boundary=bool(boundary),
                  status=('boundary' if boundary else 'ok') if result.success else 'not_converged',
                  attempts=attempts, message=str(result.message), mixture_min_weight=min_weight,
                  mixture_eigen_floor=eigen_floor, mixture_scaling=unit.tolist(), mixture_seed=seed, mixture_starts=starts,
                  fit_restriction='Scatter floor in sample-SD coordinates; weight floor; shared GED power [0.25,10]; factor log diagonals [-12,12]; logits [-20,20]',
                  fit_sec=time.perf_counter()-started)
    if nig:
        del output['power']
        output.update(psi=power, chi=1., **{'lambda': -.5})
        for record, gamma in zip(output['component_fits'], gammas):
            del record['power']
            record.update(gamma=gamma.tolist(), psi=power, chi=1., **{'lambda': -.5})
        output['fit_restriction'] = 'Scatter floor in sample-SD coordinates; weight floor; shared log(psi) [-12,12]; component gamma [-10,10] in sample-SD units; factor log diagonals [-12,12]; logits [-20,20]'
    from .joint_finite_mixture import JointFiniteMixture
    dist = JointFiniteMixture(output)
    mean, covariance = dist.mean(), dist.cov()
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Nonfinite fitted mixture moments')
    output.update(mean=mean.tolist(), covariance=covariance.tolist())
    return output
