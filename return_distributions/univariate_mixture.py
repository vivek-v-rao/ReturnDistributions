"""One-dimensional adapters for the shared finite-mixture fitter."""
import json
import numpy as np
from .model_names import display_model_label
from .joint_finite_mixture import JointFiniteMixture, fit_finite_mixture
from .projection import project_distribution


def baseline_record(row, model):
    if row is None or row.get('status') != 'ok':
        return None
    base = dict(model=model, location=[float(row['loc'])],
                scatter=[[float(row['scale'])**2]], df=None)
    if model == 'student-t':
        base['df'] = float(row['df'])
    elif model == 'ged':
        base['power'] = float(row['beta'])
    elif model == 'nig-skewed':
        base.update(psi=float(row['a'])**2-float(row['b'])**2,
                    gamma=[float(row['b'])*float(row['scale'])], chi=1., **{'lambda': -.5})
    return base


def fit_univariate_mixture(x, model, components, *, baseline=None, **kwargs):
    fit = fit_finite_mixture(np.asarray(x).reshape(-1, 1), model, components,
                             baseline=baseline_record(baseline, model), **kwargs)
    return dict(name=f'{model} [{components} components]', mixture_model=model,
                components=components, n=fit['observations'], k=fit['parameters'],
                loglik=fit['loglik'], aic=fit['aic'], bic=fit['bic'],
                status=fit['status'], converged=fit['converged'], fit_sec=fit['fit_sec'],
                mixture_fit=json.dumps(fit))


class UnivariateMixture:
    def __init__(self, fit):
        self.joint = JointFiniteMixture(fit)
        self.projected = project_distribution(fit, np.ones(1))

    def __getattr__(self, name):
        return getattr(self.projected, name)

    def support(self):
        return -np.inf, np.inf

    def logpdf(self, x):
        values = np.asarray(x)
        result = np.asarray(self.joint.logpdf(values.reshape(-1, 1))).reshape(values.shape)
        return float(result) if values.ndim == 0 else result

    def rvs(self, size=1, random_state=None):
        return self.joint.rvs(size=size, random_state=random_state).reshape(-1)


def print_components(row, units='return units'):
    fit = json.loads(row['mixture_fit'])
    print(f'\n{display_model_label(row["name"])} components ({units} per input period, not annualized; status: {row["status"]}):')
    print('component    weight          mean            sd')
    for i, (weight, record) in enumerate(sorted(zip(fit['mixture_weights'], fit['component_fits']),
                                               key=lambda pair: pair[0], reverse=True), 1):
        dist = JointFiniteMixture(dict(fit, mixture_weights=[1.], component_fits=[record]))
        mean, sd = dist.mean()[0], np.sqrt(dist.cov()[0, 0])
        fmt = lambda v: f'{v:.6f}' if np.isfinite(v) else 'n/a'
        print(f'{i:9d} {weight:9.6f} {fmt(mean):>13} {fmt(sd):>13}')
    print('n/a denotes an undefined moment; means and SDs are not location/scatter parameters.')
