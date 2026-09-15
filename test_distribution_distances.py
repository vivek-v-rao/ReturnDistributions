import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

from return_distributions.distribution_distances import univariate_distance, compare_joint, compare_univariate
from return_distributions.cli import main
from return_distributions.joint_cli import main as joint_main


class DistributionDistanceTests(unittest.TestCase):
    def test_normal_analytic_ks_kl_and_identity(self):
        p, q = stats.norm(), stats.norm(loc=1., scale=2.)
        for metric in ['js', 'ks', 'kl']:
            value, _, _ = univariate_distance(p, p, metric)
            self.assertAlmostEqual(value, 0., places=7)
        kl, error, _ = univariate_distance(p, q, 'kl')
        self.assertAlmostEqual(kl, np.log(2.)+(1+1)/8-.5, places=7)
        reverse, _, _ = univariate_distance(q, p, 'kl')
        self.assertAlmostEqual(reverse, -np.log(2.)+(4+1)/2-.5, places=7)
        value, _, _ = univariate_distance(p, stats.norm(loc=1.), 'ks')
        self.assertAlmostEqual(value, 2*stats.norm.cdf(.5)-1, places=7)
        a, _, _ = univariate_distance(p, q, 'js')
        b, _, _ = univariate_distance(q, p, 'js')
        self.assertAlmostEqual(a, b, places=9)
        self.assertTrue(0 < a < 1)
        shifted, _, _ = univariate_distance(stats.norm(100., .01), stats.norm(100.01, .02), 'js')
        self.assertAlmostEqual(a, shifted, places=6)
        disjoint, _, _ = univariate_distance(stats.uniform(0., 1.), stats.uniform(2., 1.), 'js')
        self.assertAlmostEqual(disjoint, 1., places=7)
        value, _, method = univariate_distance(stats.t(1.5), p, 'kl')
        self.assertTrue(np.isposinf(value))
        self.assertIn('analytic', method)

    def test_joint_mc_against_quadrature_and_order(self):
        def fit(mu, symbols):
            return dict(model='normal', status='ok', symbols=symbols, location=mu,
                        covariance=np.eye(2).tolist(), scatter=np.eye(2).tolist())
        # B is identical in the two distributions, so joint JS equals the A-only JS.
        a, b = fit([0., 0.], ['A', 'B']), fit([0., 1.], ['B', 'A'])
        with contextlib.redirect_stdout(io.StringIO()):
            rows = compare_joint([a, b], simulations=100000, seed=3)
            repeat = compare_joint([a, b], simulations=100000, seed=3)
        value = rows[1]['value']
        expected, _, _ = univariate_distance(stats.norm(), stats.norm(1.), 'js')
        self.assertLess(abs(value-expected), .004)
        self.assertEqual(rows, repeat)
        self.assertEqual(rows[1]['value'], rows[2]['value'])
        self.assertGreater(rows[1]['numerical_error'], 0.)
        identical = fit([0., 0.], ['B', 'A'])
        with contextlib.redirect_stdout(io.StringIO()):
            rows = compare_joint([a, identical], simulations=2000)
        self.assertEqual(rows[1]['value'], 0.)

    def test_failures_and_exclusions(self):
        table = pd.DataFrame([dict(name='normal', status='ok', loc=0., scale=1.),
                              dict(name='laplace', status='boundary', loc=0., scale=1.),
                              dict(name='bad', status='ok')])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rows = compare_univariate(table, ['js', 'ks', 'kl'])
        self.assertIn('excluded laplace: boundary', output.getvalue())
        self.assertTrue(any(r['status']=='failed' for r in rows))
        self.assertTrue(all(np.isnan(r['value']) for r in rows if r['status']=='failed'))
        with self.assertRaises(Exception):
            univariate_distance(stats.uniform(0., 1.), stats.uniform(2., 1.), 'kl')

    def test_cli_outputs_no_save_and_validation(self):
        data = pd.DataFrame(np.random.default_rng(5).normal(size=(60, 2))*.01,
            columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=60))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'returns.csv'
            data.to_csv(source)
            base = [str(source), '--input-type', 'returns', '--models', 'normal', 'laplace',
                    '--js-distance', '--ks-distance', '--kl-divergence']
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([*base, '--output', str(root/'fits.csv')]), 0)
            saved = pd.read_csv(root/'fits_distances.csv')
            self.assertEqual(len(saved), 24)
            self.assertEqual(set(saved.metric), {'js', 'ks', 'kl'})
            self.assertEqual(set(saved.symbol), {'A', 'B'})
            with patch('pandas.DataFrame.to_csv', side_effect=AssertionError('must not save')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([*base, '--no-save', '--output', '.']), 0)
                self.assertEqual(joint_main([*base, '--univariate', '--no-save', '--simulations', '2000']), 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(joint_main([str(source), '--input-type', 'returns', '--models',
                    'normal', 'laplace', '--js-distance', '--simulations', '2000',
                    '--output', str(root/'joint.json')]), 0)
            joint = pd.read_csv(root/'joint_distances.csv')
            self.assertEqual(set(joint.scope), {'joint'})
            with patch('return_distributions.joint_cli.read_returns') as reader, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                joint_main([str(source), '--ks-distance'])
            reader.assert_not_called()
            with patch('return_distributions.cli.read_returns') as reader, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(root/'fits_distances.csv'), '--output', str(root/'fits.csv'), '--js-distance'])
            reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
