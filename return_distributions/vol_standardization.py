"""Predictable, zero-mean EWMA scales shared by fitting and portfolio risk."""
import numpy as np
import pandas as pd


def ewma_standardize(frame, decay=.94, warmup=63, floor=1e-8):
    """Return standardized observations, lagged SDs and next-period SDs.

    Seed with the first warmup squared returns; never score seed observations.
    A missing observation resets that asset and requires a new warmup. Decay
    is per input observation, not per calendar day. No demeaning is performed.
    """
    if not np.isfinite(decay) or not 0 < decay < 1:
        raise ValueError('Volatility lambda must be strictly between 0 and 1')
    if warmup < 2 or not np.isfinite(floor) or floor <= 0:
        raise ValueError('Require volatility warmup >=2 and a positive finite SD floor')
    scales = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    next_scale = {}
    for symbol in frame:
        seed = []
        variance = None
        column = []
        for value in frame[symbol].to_numpy(dtype=float):
            if not np.isfinite(value):
                seed, variance = [], None
                column.append(np.nan)
                continue
            if variance is None:
                seed.append(value*value)
                column.append(np.nan)
                if len(seed) == warmup:
                    variance = max(float(np.mean(seed)), floor**2)
            else:
                column.append(np.sqrt(variance))
                variance = max(decay*variance+(1-decay)*value*value, floor**2)
        scales[symbol] = column
        next_scale[symbol] = np.sqrt(variance) if variance is not None else np.nan
    if not np.isfinite(scales.to_numpy()[scales.notna().to_numpy()]).all():
        raise ValueError('Nonfinite EWMA scales')
    return frame/scales, scales, pd.Series(next_scale)


def annotate_fit(fit, scales, next_scale, decay, warmup, floor, model='ewma', parameters=None):
    """Keep distribution parameters in standardized units; scores in return units."""
    fit.update(vol_standardization=model, vol_lambda=None if decay is None else float(decay), vol_warmup=warmup,
               vol_floor=floor, parameter_units='standardized returns',
               next_volatility=next_scale.tolist(),
               likelihood_units='original returns; conditional volatility Jacobian included')
    adjustment = float(np.log(scales.to_numpy()).sum())
    fit['vol_log_jacobian'] = adjustment
    if fit.get('loglik') is not None and np.isfinite(fit['loglik']):
        fit['standardized_loglik'] = fit['loglik']
        fit['loglik'] -= adjustment
        for key in ('aic', 'bic', 'two_stage_aic', 'two_stage_bic'):
            if fit.get(key) is not None and np.isfinite(fit[key]):
                fit[key] += 2*adjustment
    if parameters:
        extra = sum(p['parameters'] for p in parameters.values())
        fit.update(vol_parameters=parameters, vol_parameter_count=extra,
                   estimation_method='two-stage Gaussian-QML volatility / distribution fit',
                   criteria_basis='descriptive two-stage; volatility parameters included')
        if fit.get('parameters') is not None:
            fit['distribution_parameters'] = fit['parameters']
            fit['parameters'] += extra
        if fit.get('conditional_parameters') is not None:
            fit['conditional_parameters'] += extra
        for key in ('aic', 'bic', 'two_stage_aic', 'two_stage_bic'):
            if fit.get(key) is not None and np.isfinite(fit[key]):
                fit[key] += (2 if key.endswith('aic') else np.log(len(scales)))*extra


def conditional_weights(fit, weights):
    """Map original return weights into standardized coordinates exactly once."""
    weights = np.asarray(weights, dtype=float)
    if fit.get('vol_standardization') not in ('ewma', 'garch', 'nagarch'):
        return weights
    scale = np.asarray(fit['next_volatility'], dtype=float)
    if scale.shape != weights.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid next-period volatility scales')
    return weights*scale


class ReturnUnitDistribution:
    """Positive diagonal rescaling for joint density/simulation comparisons."""
    def __init__(self, distribution, fit):
        self.distribution = distribution
        self.scale = conditional_weights(fit, np.ones(len(fit['symbols'])))

    def logpdf(self, x):
        return self.distribution.logpdf(np.asarray(x)/self.scale)-np.log(self.scale).sum()

    def rvs(self, size=1, random_state=None):
        return self.distribution.rvs(size=size, random_state=random_state)*self.scale
