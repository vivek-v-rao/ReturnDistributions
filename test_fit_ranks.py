import unittest
import numpy as np
import pandas as pd
from return_distributions.fit_ranks import criterion_ranks
from return_distributions.fitting import fit_many
from return_distributions.table_format import aligned_table


class FitRankTests(unittest.TestCase):
    def test_independent_criteria_and_eligibility(self):
        table = pd.DataFrame(dict(aic=[1., 2., 0., np.inf, np.nan], bic=[5., 4., 3., 2., np.nan],
                                  status=['ok', 'ok', 'not_converged', 'ok', 'failed']))
        ranked = criterion_ranks(table)
        self.assertEqual(ranked.aic_rank.iloc[:2].tolist(), [1., 2.])
        self.assertEqual(ranked.bic_rank.iloc[:2].tolist(), [3., 2.])
        self.assertTrue(ranked.aic_rank.iloc[2:].isna().all())
        self.assertTrue(np.isnan(ranked.bic_rank.iloc[2]))
        self.assertEqual(ranked.bic_rank.iloc[3], 1.)
        self.assertNotIn('aic_rank', table)
        text = aligned_table(ranked[['aic_rank', 'bic_rank']])
        self.assertNotIn('1.00', text)

    def test_univariate_fitter_exports_both_ranks(self):
        fitted = fit_many(np.random.default_rng(7).normal(size=100), ['normal', 'laplace'])
        self.assertIn('aic_rank', fitted)
        self.assertIn('bic_rank', fitted)
        self.assertNotIn('rank', fitted)
        np.testing.assert_array_equal(fitted.aic_rank, fitted.aic.rank(method='min'))
        np.testing.assert_array_equal(fitted.bic_rank, fitted.bic.rank(method='min'))


if __name__ == '__main__':
    unittest.main()
