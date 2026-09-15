"""Console compaction must not change saved fit data or hide partial metadata."""
import unittest

import numpy as np
import pandas as pd

from return_distributions.joint_cli import format_fit_comparison


class ComparisonFormatTests(unittest.TestCase):
    def test_shared_empty_and_partial_columns(self):
        frame = pd.DataFrame({
            'model': ['normal', 'asymmetric-laplace'],
            'window': [None, None], 'observations': [5833, 5833],
            'dimensions': [2, 2], 'parameters': [5, 5],
            'first_date': ['2002-07-31'] * 2, 'last_date': ['2025-10-03'] * 2,
            'loglik': [36986., 38054.], 'aic': [-73963., -76099.],
            'status': ['ok', 'ok'], 'fit_sec': [.001, .5],
            'error': [None, np.nan], 'vg_shape': [None, 1.],
            'location_source': [None, 'default-zero'],
            'fit_restriction': [None, 'Long restriction'], 'rank': [2., 1.],
        })
        before = frame.copy(deep=True)
        text = format_fit_comparison(frame)
        shared, header, *rows = text.splitlines()
        self.assertIn('window: all available history', shared)
        self.assertIn('observations: 5833', shared)
        self.assertIn('status: ok', shared)
        self.assertNotIn('observations', header)
        self.assertNotIn('error', text)
        self.assertNotIn('fit_restriction', text)
        self.assertNotIn('None', text)
        self.assertNotIn('nan', text)
        self.assertIn('n/a', rows[0])
        self.assertNotIn('vg_shape', shared)
        self.assertNotIn('location_source', shared)
        self.assertEqual(header.split()[:5], ['model', 'loglik', 'aic', 'rank', 'fit_sec'])
        self.assertIn('vg_shape', header)
        pd.testing.assert_frame_equal(frame, before)

    def test_mixed_windows_failures_and_pilot_scores(self):
        frame = pd.DataFrame({
            'model': ['normal', 'normal', 'asymmetric-laplace'],
            'window': [None, 252, 252], 'observations': [500, 252, 252],
            'status': ['ok', 'failed', 'ok'],
            'error': [None, 'fit failed', None],
            'aic': [-10., None, None], 'two_stage_aic': [None, None, -12.],
            'rank': [1., None, None],
        })
        text = format_fit_comparison(frame)
        header = text.splitlines()[0]
        for name in frame.columns:
            self.assertIn(name, header.split())
        self.assertIn('fit failed', text)
        self.assertIn('n/a', text)

    def test_single_row_and_empty(self):
        text = format_fit_comparison(pd.DataFrame([{'model': 'normal', 'status': 'ok'}]))
        self.assertEqual(text.splitlines()[0], 'status: ok')
        self.assertEqual(text.splitlines()[1].strip(), 'model')
        self.assertEqual(text.splitlines()[2].strip(), 'normal')
        self.assertEqual(format_fit_comparison(pd.DataFrame()), 'No fits.')


if __name__ == '__main__':
    unittest.main()
