"""Test decimal alignment and elapsed-time precision in console tables.
Cover missing values, infinities, scientific notation, and unchanged inputs.
"""

import re
import unittest

import numpy as np
import pandas as pd

from return_distributions.table_format import aligned_table


class TableFormatTests(unittest.TestCase):
    def test_elapsed_seconds_precision(self):
        from return_distributions.joint_cli import format_fit_comparison
        frame = pd.DataFrame({'fit_sec': [1.283425, .841082, .020690, 0.000001],
                              'elapsed_seconds': [1, 2, 3, 4]})
        original = frame.copy(deep=True)
        text = aligned_table(frame)
        for expected in ['1.283', '0.841', '0.021', '0.000', '1.000']:
            self.assertIn(expected, text)
        for line in text.splitlines()[1:]:
            for token in line.split():
                self.assertRegex(token, r'^\d+\.\d{3}$')
        pd.testing.assert_frame_equal(frame, original)
        shared = format_fit_comparison(pd.DataFrame({'model': ['normal'], 'fit_sec': [1.283425]}))
        self.assertIn('fit_sec: 1.283', shared)
        self.assertNotIn('1.2834', shared)

    def test_fixed_decimals_and_unchanged_input(self):
        frame = pd.DataFrame({'name': ['a', 'b', 'c'], 'mean': [.000812, .00049574, .00047227],
                              'skew': [0., -.092253, 1.5], 'fit_sec': [1.1631, 3.9637, .17174],
                              'n': [8226, 8226, 8226]})
        before = frame.copy(deep=True)
        text = aligned_table(frame)
        positions = [[m.start() for m in re.finditer(r'\.', line)] for line in text.splitlines()[1:]]
        self.assertEqual(positions[0], positions[1])
        self.assertEqual(positions[1], positions[2])
        self.assertIn('0.00081200', text)
        self.assertNotIn('8226.', text)
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_infinite_object_and_scientific(self):
        frame = pd.DataFrame({'x': pd.Series([1e-100, -2.5e-8, None, np.inf], dtype=object),
                              'flag': [True, False, None, True], 'rank': [1., 2., np.nan, 3.]})
        text = aligned_table(frame)
        lines = text.splitlines()
        self.assertEqual(lines[1].index('.'), lines[2].index('.'))
        self.assertIn('1.0000e-100', text)
        self.assertIn('-2.5000e-008', text)
        self.assertIn('n/a', text)
        self.assertIn('inf', text)
        self.assertIn('True', text)
        self.assertNotIn('None', text)
        self.assertNotIn('nan', text)


if __name__ == '__main__':
    unittest.main()
