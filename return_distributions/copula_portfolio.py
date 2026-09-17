"""Monte Carlo portfolio summaries from fitted parametric copulas."""
import numpy as np
from .copulas import CopulaJoint, sample_copula
from .copula_quantiles import marginal_ppf


def empirical_es(values, tail_probability):
    """Integrated empirical quantile: fractional weight at the tail boundary."""
    ordered = np.sort(values)
    mass = tail_probability*len(ordered)
    if round(mass) >= 1 and abs(mass-round(mass)) < 1e-8: mass = float(round(mass))
    k = int(np.floor(mass))
    fraction = mass-k
    total = ordered[:k].sum()
    if fraction > 0: total += fraction*ordered[k]
    return -float(total/mass)


def simulate_portfolio(record, weights, quantiles=(.01,.05,.5,.95,.99), risk_levels=(.95,.99),
                       simulations=100000, seed=12345, batches=20, allow_log=False):
    """Independent batches estimate MC error, not model/parameter uncertainty.

    A nonintegrable held marginal means finite portfolio ES is not established;
    we suppress ES rather than infer finiteness from finite simulated samples.
    """
    if record.get('return_type') == 'log' and not allow_log:
        raise ValueError('Log copula fits require explicit opt-in for a linear combination, not portfolio log returns')
    joint = CopulaJoint(record)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(joint.marginals),) or not np.isfinite(w).all(): raise ValueError('Invalid portfolio weights')
    if simulations < 1000 or batches < 2 or simulations//batches < 50 or seed < 0:
        raise ValueError('Require >=1000 simulations, >=2 batches, >=50 draws per batch and a nonnegative seed')
    if any(not 0 < p < 1 for p in [*quantiles,*risk_levels]): raise ValueError('Probabilities must be inside (0,1)')
    held = np.flatnonzero(w)
    means = [float(joint.marginals[j].mean()) for j in held]
    variances = [float(joint.marginals[j].var()) for j in held]
    finite_mean = bool(np.isfinite(means).all())
    finite_variance = bool(np.isfinite(variances).all())
    seeds = np.random.SeedSequence(seed).spawn(batches)
    sizes = [simulations//batches+(i < simulations % batches) for i in range(batches)]
    pieces, endpoint_count = [], 0
    for count, child in zip(sizes,seeds):
        if not len(held):
            pieces.append(np.zeros(count))
            continue
        u = sample_copula(record['copula'], count, np.random.default_rng(child))
        endpoint_count += int(((u[:,held] <= 0) | (u[:,held] >= 1)).sum())
        u = np.clip(u, np.nextafter(0.,1.), np.nextafter(1.,0.))
        values = np.zeros(count)
        for j in held: values += w[j]*marginal_ppf(joint.marginals[j], u[:,j])
        if not np.isfinite(values).all(): raise ValueError('Nonfinite simulated returns; no draws were silently discarded')
        pieces.append(values)
    draws = np.concatenate(pieces)
    # Endpoint rounding can materially truncate extreme tails; do not publish ES.
    reliable_es = finite_mean and endpoint_count == 0
    row = dict(method='copula-monte-carlo', simulations=simulations, seed=seed, mc_batches=batches,
               endpoint_rounding=endpoint_count,
               mean=float(w[held]@means) if finite_mean else np.nan,
               volatility=float(draws.std(ddof=1)) if finite_variance else np.nan,
               moment_status='finite marginal moments' if finite_variance else 'some held marginal moments unavailable')

    def standard_error(function, p, needs_variance=False):
        if min(sizes)*min(p,1-p) < 10 or endpoint_count or (needs_variance and not finite_variance): return np.nan
        values = [function(piece) for piece in pieces]
        return float(np.std(values,ddof=1)/np.sqrt(batches))

    for q in dict.fromkeys(quantiles):
        row[f'q_{q:g}'] = float(np.quantile(draws,q))
        row[f'q_{q:g}_mc_se'] = standard_error(lambda x: np.quantile(x,q),q)
    for level in dict.fromkeys(risk_levels):
        q = 1-level
        row[f'var_{level:g}'] = -float(np.quantile(draws,q))
        row[f'var_{level:g}_mc_se'] = standard_error(lambda x: -np.quantile(x,q),q)
        row[f'es_{level:g}'] = empirical_es(draws,q) if reliable_es else np.nan
        row[f'es_{level:g}_mc_se'] = standard_error(lambda x: empirical_es(x,q),q,True) if reliable_es else np.nan
        mass = q*simulations
        if round(mass) >= 1 and abs(mass-round(mass)) < 1e-8: mass = float(round(mass))
        row[f'tail_count_{level:g}'] = int(np.ceil(mass))
        row[f'es_status_{level:g}'] = ('finite-moment Monte Carlo estimate' if reliable_es else
                ('unavailable: CDF endpoint rounding' if endpoint_count else 'finite ES not established: held marginal lacks finite mean'))
        row[f'es_method_{level:g}'] = 'empirical fractional-tail mean'
    return row, draws
