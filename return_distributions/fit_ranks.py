"""Comparable AIC/BIC ranks with explicit eligibility and grouping."""
import numpy as np
import pandas as pd


def criterion_ranks(frame, groups=None, include_unconverged=False, suffix=''):
    result = frame.copy()
    for criterion in ('aic', 'bic'):
        scores = pd.to_numeric(result.get(criterion, pd.Series(np.nan, index=result.index)), errors='coerce')
        scores = scores.where(np.isfinite(scores))
        if not include_unconverged:
            scores = scores.where(result.status.eq('ok'))
        ranking = scores.groupby(groups).rank(method='min') if groups is not None else scores.rank(method='min')
        result[f'{criterion}_rank{suffix}'] = ranking
    return result
