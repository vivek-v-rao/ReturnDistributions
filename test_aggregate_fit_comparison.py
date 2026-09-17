"""Test aggregate model comparisons across families, component counts, and windows.
Cover AIC/BIC ranks, ties, and selected asset-order filtering.
"""

import unittest

import numpy as np
import pandas as pd

from return_distributions.joint_cli import aggregate_fit_comparison


class AggregateComparisonTests(unittest.TestCase):
    def test_all_families_components_and_window_ranks(self):
        frame = pd.DataFrame(dict(
            model=['normal', 'normal', 'laplace', 'student-t', 'normal'],
            components=[1, 2, 1, 1, 1], window=[126, 126, 126, 126, 252],
            aic=[10., 8., 8., np.nan, -20.], bic=[11., 15., 12., np.inf, -19.],
            status=['ok', 'not_converged', 'ok', 'failed', 'ok']))
        result = aggregate_fit_comparison(frame)
        np.testing.assert_allclose(result.aic_rank, [3, 1, 1, np.nan, 1], equal_nan=True)
        np.testing.assert_allclose(result.bic_rank, [1, 3, 2, np.nan, 1], equal_nan=True)
        self.assertEqual(len(result), len(frame))
        self.assertEqual(result.loc[1, 'status'], 'not_converged')
        self.assertNotIn('aic_rank', frame)

    def test_single_components_and_selected_orders(self):
        frame = pd.DataFrame(dict(model=['ged-skewed', 'ged-skewed', 'normal'],
                                  window=[None]*3, aic=[1., 2., 3.], bic=[2., 3., 4.],
                                  status=['ok']*3, selected_for_display=[True, False, True]))
        result = aggregate_fit_comparison(frame)
        self.assertEqual(list(result.index), [0, 2])
        self.assertEqual(list(result.components), [1, 1])
        self.assertEqual(list(result.aic_rank), [1, 2])


if __name__ == '__main__':
    unittest.main()
