"""Zero-mean Gaussian-QML GARCH/NAGARCH filters (two-stage estimation)."""
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


def variance_path(x, seed, omega, alpha, beta, theta=0.):
    """h[t] uses returns strictly before t; final element forecasts next return."""
    if theta == 0:
        return np.r_[seed, lfilter([1.], [1., -beta], omega+alpha*x*x,
                                  zi=[beta*seed])[0]]
    h = np.empty(len(x)+1)
    h[0] = seed
    for t, value in enumerate(x):
        h[t+1] = omega+beta*h[t]+alpha*(value-theta*np.sqrt(h[t]))**2
    return h


def garch_standardize(frame, model='garch', warmup=63, floor=1e-8, max_iterations=500, verbose=False):
    """Fit once per asset on the supplied block, excluding its seed observations.

    Missing observations are rejected, not silently bridged. Parameters use the
    whole supplied block: this is in-sample QML, not a walk-forward backtest.
    """
    if model not in ('garch', 'nagarch') or warmup < 2 or floor <= 0 or not np.isfinite(floor):
        raise ValueError('Invalid GARCH model, warmup, or volatility floor')
    if len(frame) < warmup+8 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('GARCH requires finite consecutive observations and warmup + 8 returns')
    scales = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    next_scales, parameters = {}, {}
    for symbol in frame:
        started = time.perf_counter()
        if verbose:
            print(f'Fitting {model.upper()} volatility: {symbol} ({len(frame)-warmup} observations)...', flush=True)
        original = frame[symbol].to_numpy(dtype=float)
        unit = np.sqrt(np.mean(original**2))
        if unit <= floor:
            raise ValueError(f'{symbol}: insufficient variation for {model}')
        x = original/unit
        seed = max(float(np.mean(x[:warmup]**2)), (floor/unit)**2)
        observations = x[warmup:]
        def decode(p):
            omega = np.exp(p[0])
            # Positive contributions sum to persistence < 1.
            a, b = np.exp(p[1:3])
            theta = p[3] if model == 'nagarch' else 0.
            return omega, .999*a/(1+a+b)/(1+theta*theta), .999*b/(1+a+b), theta
        def objective(p):
            h = variance_path(observations, seed, *decode(p))[:-1]
            h = np.maximum(h, (floor/unit)**2)
            return .5*np.sum(np.log(h)+observations**2/h)
        candidates = []
        for persistence in (.90, .98):
            for theta in ((0., .7) if model == 'nagarch' else (0.,)):
                contribution = .07
                rest = .999-persistence
                p = [np.log(1-persistence), np.log(contribution/rest),
                     np.log((persistence-contribution)/rest)]
                if model == 'nagarch': p.append(theta)
                candidates.append(minimize(objective, p, method='L-BFGS-B',
                    bounds=[(-16, 4), (-12, 12), (-12, 12)]+([(-5, 5)] if model == 'nagarch' else []),
                    options={'maxiter': max_iterations, 'ftol': 1e-10}))
        good = [fit for fit in candidates if fit.success and np.isfinite(fit.fun)]
        if not good:
            raise ValueError(f'{symbol}: {model} QML did not converge: {candidates[0].message}')
        best = min(good, key=lambda fit: fit.fun)
        omega, alpha, beta, theta = decode(best.x)
        h = variance_path(observations, seed, omega, alpha, beta, theta)
        sd = np.sqrt(np.maximum(h*unit**2, floor**2))
        scales.loc[frame.index[warmup:], symbol] = sd[:-1]
        next_scales[symbol] = sd[-1]
        parameters[symbol] = dict(omega=omega*unit**2, alpha=alpha, beta=beta,
            theta=theta, persistence=beta+alpha*(1+theta**2), mean=0.,
            parameters=4 if model == 'nagarch' else 3, converged=True,
            fit_sec=time.perf_counter()-started, optimizer_message=str(best.message))
        parameters[symbol]['near_boundary'] = bool(any(
            abs(value-lo) < 1e-3 or abs(value-hi) < 1e-3
            for value, (lo, hi) in zip(best.x,
                [(-16, 4), (-12, 12), (-12, 12)]+([(-5, 5)] if model == 'nagarch' else []))))
    return frame/scales, scales, pd.Series(next_scales), parameters


def fit_selected(frame, selected, model, warmup, floor):
    """Include only preceding seed data and selected observations, never later data."""
    start = frame.index.get_loc(selected[0])
    end = frame.index.get_loc(selected[-1])
    block = frame.iloc[max(0, start-warmup):end+1]
    if start < warmup or not block.index[warmup:].equals(selected):
        raise ValueError('GARCH sample has gaps or insufficient preceding warmup data')
    result = garch_standardize(block, model, warmup, floor, verbose=True)
    print(f'{model.upper()} zero-mean Gaussian QML (in-sample; two-stage criteria):')
    table = pd.DataFrame(result[3]).T[['omega', 'alpha', 'beta', 'theta', 'persistence', 'converged', 'near_boundary', 'fit_sec']]
    table['fit_sec'] = table['fit_sec'].map(lambda v: f'{v:.3f}')
    print(table.to_string(float_format=lambda v: f'{v:.6g}'))
    if any(p['near_boundary'] for p in result[3].values()):
        print('Warning: volatility optimizer near a parameter bound; inspect persistence and parameters.')
    return result
