"""Reproducible risk for joint families without a univariate projection family."""
import numpy as np
from .joint_sdb import JointSDB, SDB_MODELS
from .joint_generalized_t import JointGeneralizedT
from .joint_power import JointPower, SKEW_POWER_MODELS
from .copula_portfolio import empirical_es

MC_JOINT_MODELS = (*SDB_MODELS, 'generalized-t-skewed', *SKEW_POWER_MODELS)


def simulate_joint_portfolio(record, weights, quantiles=(.01,.05,.5,.95,.99), risk_levels=(.95,.99),
                             simulations=100000, seed=12345, batches=20, allow_log=False):
    if record.get('model') not in MC_JOINT_MODELS or record.get('status','ok') != 'ok':
        raise ValueError('Requires an accepted simulation-supported joint fit')
    if record.get('return_type') == 'log' and not allow_log:
        raise ValueError('Log returns require explicit opt-in; the sum is not a portfolio log return')
    gt = record['model'] == 'generalized-t-skewed'
    power = record['model'] in SKEW_POWER_MODELS
    dist = JointGeneralizedT(record) if gt else (JointPower(record) if power else JointSDB(record))
    from .vol_standardization import conditional_weights
    w = conditional_weights(record, weights)
    if w.shape != dist.location.shape or not np.isfinite(w).all(): raise ValueError('Invalid portfolio weights')
    if simulations < 1000 or batches < 2 or simulations//batches < 50 or seed < 0:
        raise ValueError('Require >=1000 simulations, >=2 batches, >=50 draws/batch and nonnegative seed')
    if any(not 0 < p < 1 for p in [*quantiles,*risk_levels]): raise ValueError('Probabilities must be inside (0,1)')
    zero = not np.any(w)
    tail_index = dist.power*dist.q if gt else (None if power else dist.df)
    finite_mean = zero or tail_index is None or tail_index > 1
    finite_var = zero or tail_index is None or tail_index > 2
    sizes = [simulations//batches+(i < simulations%batches) for i in range(batches)]
    pieces = [np.zeros(n) if zero else dist.rvs(n,np.random.default_rng(child))@w
              for n,child in zip(sizes,np.random.SeedSequence(seed).spawn(batches))]
    draws = np.concatenate(pieces)
    if not np.isfinite(draws).all(): raise ValueError('Nonfinite simulated returns; no draws discarded')
    row = dict(method='generalized-t-monte-carlo' if gt else ('skew-power-monte-carlo' if power else 'sdb-monte-carlo'),simulations=simulations,seed=seed,mc_batches=batches,endpoint_rounding=0,
               mean=0. if zero else (float(w@dist.mean()) if finite_mean else np.nan),
               volatility=0. if zero else (float(np.sqrt(max(0.,w@dist.cov()@w))) if finite_var else np.nan),
               moment_status='analytic mean/volatility; simulated quantiles and ES' if finite_var else 'some moments undefined')
    def se(function,p,need_var=False):
        if zero: return 0.
        if min(sizes)*min(p,1-p)<10 or (need_var and not finite_var): return np.nan
        return float(np.std([function(piece) for piece in pieces],ddof=1)/np.sqrt(batches))
    for q in dict.fromkeys(quantiles):
        row[f'q_{q:g}'] = float(np.quantile(draws,q))
        row[f'q_{q:g}_mc_se'] = se(lambda v:np.quantile(v,q),q)
    for c in dict.fromkeys(risk_levels):
        q = 1-c
        row[f'var_{c:g}'] = -float(np.quantile(draws,q))
        row[f'var_{c:g}_mc_se'] = se(lambda v:-np.quantile(v,q),q)
        row[f'es_{c:g}'] = empirical_es(draws,q) if finite_mean else np.inf
        row[f'es_{c:g}_mc_se'] = se(lambda v:empirical_es(v,q),q,True) if finite_mean else np.nan
        mass = q*simulations
        if round(mass)>=1 and abs(mass-round(mass))<1e-8: mass=float(round(mass))
        row[f'tail_count_{c:g}'] = int(np.ceil(mass))
        row[f'es_status_{c:g}'] = 'finite-moment Monte Carlo estimate' if finite_mean else ('infinite (power*q <= 1)' if gt else 'infinite (df <= 1)')
        row[f'es_method_{c:g}'] = 'empirical fractional-tail mean' if finite_mean else 'analytic moment-existence check'
    return row,draws
