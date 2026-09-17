"""Test GARCH/NAGARCH variance recursion, asymmetry, and scale equivariance.
Cover sample boundaries, stationarity, score adjustments, and conditional risk.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.garch_standardization import variance_path, garch_standardize, fit_selected
from return_distributions.vol_standardization import annotate_fit, conditional_weights


class GarchTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(np.random.default_rng(91).normal(size=(140, 2))*.01,
            columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=140))

    def test_recursion_and_asymmetry(self):
        for theta in (0., .7):
            x = np.array([.3, -.4, .1])
            h = variance_path(x, .2, .01, .1, .8, theta)
            expected = [.2]
            for r in x:
                expected.append(.01+.8*expected[-1]+.1*(r-theta*np.sqrt(expected[-1]))**2)
            np.testing.assert_allclose(h, expected)
            changed = variance_path(np.array([.3, -.4, 5.]), .2, .01, .1, .8, theta)
            np.testing.assert_equal(h[:-1], changed[:-1])
        self.assertGreater(variance_path(np.array([-1.]), 1, .1, .1, .8, .7)[1],
                           variance_path(np.array([1.]), 1, .1, .1, .8, .7)[1])

    def test_fit_scale_and_stationarity(self):
        frame = self.frame()[['A']]
        for model in ('garch', 'nagarch'):
            z, h, nxt, pars = garch_standardize(frame, model, 20)
            z2, h2, nxt2, pars2 = garch_standardize(frame*100, model, 20)
            self.assertTrue(z.iloc[:20].isna().all().all())
            np.testing.assert_allclose(z.iloc[20:], z2.iloc[20:], rtol=2e-4)
            np.testing.assert_allclose(nxt*100, nxt2, rtol=2e-4)
            self.assertLess(pars['A']['persistence'], 1)
            self.assertTrue(pars['A']['converged'])
            self.assertTrue((h.iloc[20:] > 0).all().all())

    def test_selected_does_not_use_later_observations(self):
        frame = self.frame()[['A']]
        selected = frame.index[40:100]
        later = frame.copy()
        later.iloc[100:] = 100
        with contextlib.redirect_stdout(io.StringIO()):
            first = fit_selected(frame, selected, 'garch', 20, 1e-8)
            second = fit_selected(later, selected, 'garch', 20, 1e-8)
        np.testing.assert_array_equal(first[0], second[0])
        with self.assertRaisesRegex(ValueError, 'gaps'):
            fit_selected(frame, selected.delete(5), 'garch', 20, 1e-8)
        frame.iloc[30] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite consecutive'):
            fit_selected(frame, selected, 'garch', 20, 1e-8)

    def test_annotation_scores_and_risk(self):
        fit = dict(loglik=10., aic=-10., bic=5*np.log(30)-20, parameters=5)
        scales = pd.DataFrame(np.full((30, 2), .02), columns=['A', 'B'])
        pars = {'A': {'parameters': 4}, 'B': {'parameters': 4}}
        annotate_fit(fit, scales, pd.Series([.03, .04]), None, 20, 1e-8, 'nagarch', pars)
        ll = 10-np.log(.02)*60
        self.assertAlmostEqual(fit['loglik'], ll)
        self.assertAlmostEqual(fit['aic'], 2*13-2*ll)
        self.assertAlmostEqual(fit['bic'], np.log(30)*13-2*ll)
        np.testing.assert_allclose(conditional_weights(fit, [.6, .4]), [.018, .016])

    def test_cli_saved_fits_and_univariate(self):
        from return_distributions.joint_cli import main as joint
        from return_distributions.cli import main as univariate
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.frame().to_csv(path/'r.csv')
            common = [str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal',
                '--days', '60', '--vol-warmup', '20', '--standardize-vol', 'none', 'garch', 'nagarch']
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(joint(common+['--output', str(path/'joint.json'),
                    '--weights', 'A=.6', 'B=.4', '--risk-levels', '.99', '--univariate']), 0)
                self.assertEqual(univariate(common+['--output', str(path/'uni.csv'), '--risk-levels', '.99']), 0)
            fits = json.loads((path/'joint.json').read_text())
            self.assertEqual([f['parameters'] for f in fits], [5, 11, 13])
            self.assertEqual(len({f['first_date'] for f in fits}), 1)
            self.assertEqual(fits[2]['vol_parameter_count'], 8)
            self.assertEqual(len(fits[2]['next_volatility']), 2)
            uni = pd.read_csv(path/'uni.csv')
            self.assertEqual(set(uni.loc[uni.vol_standardization.eq('nagarch'), 'k']), {6})
            risks = pd.read_csv(path/'joint_portfolio_risk.csv')
            self.assertTrue(risks.loc[risks.vol_standardization.eq('nagarch'), 'risk_basis'].str.contains('conditional').all())


if __name__ == '__main__':
    unittest.main()
