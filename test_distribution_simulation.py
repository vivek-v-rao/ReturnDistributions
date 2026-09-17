"""Test joint-distribution simulation presets and reproducible fitting studies.
Check simulated moments, exclusions, and ranking ties.
"""

import unittest
import numpy as np
import pandas as pd
from return_distributions.multivariate import JOINT_MODELS
from return_distributions.simulation import preset, run_study, summarize
from return_distributions import joint_distribution


class SimulationTests(unittest.TestCase):
    def test_presets(self):
        for name in JOINT_MODELS:
            truth = preset(name, 3)
            model = joint_distribution(truth)
            x = model.rvs(size=20000, random_state=np.random.default_rng(77))
            np.testing.assert_allclose(x.mean(axis=0), truth['mean'], atol=.0006)
            np.testing.assert_allclose(np.cov(x.T), truth['covariance'], atol=.00003)

    def test_reproducibility(self):
        truth = [preset('normal')]
        a = run_study(truth, ['normal'], [50], 2, seed=8)
        b = run_study(truth, ['normal'], [50], 2, seed=8)
        np.testing.assert_array_equal(a[0].aic, b[0].aic)
        self.assertTrue(a[2].selection_rate.eq(1).all())
        self.assertTrue(a[3].estimates.eq(2).all())

    def test_exclusion_and_ties(self):
        rows = []
        for rep, statuses in [(1, ['ok', 'ok']), (2, ['ok', 'boundary']), (3, ['failed', 'ok'])]:
            for model, status in zip(['normal', 'student-t'], statuses):
                rows.append(dict(truth_id=0, sample_size=50, replication=rep, model=model, status=status, aic=5., bic=6.))
        selection, _ = summarize(pd.DataFrame(rows), pd.DataFrame(), ['normal', 'student-t'])
        self.assertTrue(selection.eligible.eq(1).all())
        self.assertTrue(selection.excluded.eq(2).all())
        self.assertTrue(selection.selection_rate.eq(.5).all())


if __name__ == '__main__':
    unittest.main()
