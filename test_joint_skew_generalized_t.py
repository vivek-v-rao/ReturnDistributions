"""Test two-piece multivariate generalized-t densities and moment calculations.
Cover limiting cases, fit reconstruction, seeded risk, and CLI integration.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import integrate

from return_distributions.generalized_t import generalized_t
from return_distributions.joint_generalized_t import JointGeneralizedT
from return_distributions.multivariate import fit_joint, joint_distribution, ALL_JOINT_MODELS
from return_distributions.joint_portfolio_mc import simulate_joint_portfolio
from return_distributions.joint_cli import main
from return_distributions.projection import project_distribution
from return_distributions.simulation import preset
from return_distributions.portfolio_cli import main as portfolio_main


def truth(d=2, power=1.5, q=5.):
    return dict(model='generalized-t-skewed', location=[0.]*d,
                scatter=np.eye(d).tolist(), power=power, q=q,
                skewness=np.linspace(-.35, .4, d).tolist(), status='ok')


class JointSkewGeneralizedTTests(unittest.TestCase):
    def test_univariate_and_symmetric_limits(self):
        for s in [-.7, 0., .5]:
            fit = dict(truth(1), location=[.2], scatter=[[.49]], skewness=[s])
            dist = JointGeneralizedT(fit)
            ref = generalized_t(1.5, 5., s, loc=.2, scale=.7)
            x = np.linspace(-5, 5, 101)
            np.testing.assert_allclose(dist.pdf(x[:, None]), ref.pdf(x), rtol=1e-12)
            np.testing.assert_allclose(dist.mean(), [ref.mean()], atol=1e-12)
            np.testing.assert_allclose(dist.cov(), [[ref.var()]], atol=1e-12)
        for d in [2, 4]:
            fit = dict(truth(d), skewness=[0.]*d)
            symmetric = dict(fit, model='generalized-t')
            del symmetric['skewness']
            x = np.random.default_rng(1).normal(size=(30, d))
            a, b = JointGeneralizedT(fit), JointGeneralizedT(symmetric)
            np.testing.assert_array_equal(a.logpdf(x), b.logpdf(x))
            np.testing.assert_array_equal(a.cov(), b.cov())

    def test_normalization_and_moments(self):
        # Independently integrate each quadrant in the observed coordinates.
        f = JointGeneralizedT(truth())
        total = 0.
        for a, b in [(-np.inf, 0.), (0., np.inf)]:
            for c, d in [(-np.inf, 0.), (0., np.inf)]:
                total += integrate.dblquad(lambda y, x: f.pdf([x, y]), a, b,
                                           lambda x: c, lambda x: d, epsabs=2e-7)[0]
        self.assertAlmostEqual(total, 1., places=6)
        fit = truth(3)
        chol = np.array([[1., 0., 0.], [.4, .8, 0.], [-.3, .2, 1.2]])
        fit['scatter'] = (chol@chol.T).tolist()
        f = JointGeneralizedT(fit)
        draws = f.rvs(300000, random_state=17)
        np.testing.assert_allclose(draws.mean(axis=0), f.mean(), atol=.015)
        np.testing.assert_allclose(np.cov(draws.T), f.cov(), atol=.035, rtol=.025)
        self.assertGreater(np.linalg.eigvalsh(f.cov()).min(), 0.)

    def test_fitting_roundtrip(self):
        x = JointGeneralizedT(truth(2, 1.5, 3.)).rvs(700, random_state=23)
        for location, k in [(None, 9), (0., 7)]:
            fit = fit_joint(x, 'generalized-t-skewed', location=location, max_iterations=1000)
            self.assertEqual(fit['status'], 'ok')
            self.assertEqual(fit['parameters'], k)
            model = joint_distribution(json.loads(json.dumps(fit, allow_nan=False)))
            self.assertAlmostEqual(model.logpdf(x).sum(), fit['loglik'], places=6)
            self.assertAlmostEqual(fit['aic'], 2*k-2*fit['loglik'])
            if location is not None:
                np.testing.assert_allclose(fit['location'], 0., atol=1e-12)
        self.assertEqual(fit_joint(x, 'generalized-t-skewed', max_iterations=1)['status'], 'not_converged')

    def test_risk_thresholds_seed_and_cli(self):
        fit = dict(truth(), symbols=['A', 'B'], return_type='simple')
        w = [.6, -.4]
        a, draws = simulate_joint_portfolio(fit, w, [.05], [.95], 20000, 4, 10)
        b, repeated = simulate_joint_portfolio(fit, w, [.05], [.95], 20000, 4, 10)
        np.testing.assert_array_equal(draws, repeated)
        self.assertEqual(a['var_0.95'], b['var_0.95'])
        self.assertGreater(a['es_0.95'], a['var_0.95'])
        self.assertTrue(np.isfinite(a['es_0.95_mc_se']))
        with self.assertRaisesRegex(ValueError, 'simulate_joint_portfolio'):
            project_distribution(fit, w)
        for index in [.8, 1., 1.5, 2., 3.]:
            f = dict(fit, power=1., q=index)
            dist = JointGeneralizedT(f)
            self.assertEqual(np.isfinite(dist.mean()).all(), index > 1)
            self.assertEqual(dist.cov() is not None, index > 2)
            row, _ = simulate_joint_portfolio(f, w, [], [.95], 2000, 2, 4)
            self.assertEqual(np.isfinite(row['es_0.95']), index > 1)
            if index <= 2: self.assertTrue(np.isnan(row['es_0.95_mc_se']))
        zero, _ = simulate_joint_portfolio(dict(fit, power=1., q=.8), [0., 0.], [], [.95], 2000, 2, 4)
        self.assertEqual(zero['es_0.95'], 0.)
        with self.assertRaises(ValueError):
            simulate_joint_portfolio(dict(fit, status='boundary'), w)
        with self.assertRaises(ValueError):
            simulate_joint_portfolio(dict(fit, return_type='log'), w)
        sample = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 2)),
            columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=20))
        fixture = dict(fit, parameters=9, observations=20, dimensions=2, loglik=1., aic=16., bic=20.,
                       covariance=JointGeneralizedT(fit).cov().tolist(), correlation=np.eye(2).tolist())
        with patch('return_distributions.joint_cli.read_returns', return_value=sample), \
             patch('return_distributions.joint_cli.fit_joint', return_value=fixture), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = main(['unused.csv', '--models', 'generalized-t-skewed', '--no-save',
                         '--weights', 'A=.6', 'B=.4', '--simulations', '2000'])
        self.assertEqual(code, 0)
        self.assertIn('asset ordering matters', output.getvalue())
        self.assertIn('simulated risk estimates', output.getvalue())
        self.assertIn('Monte Carlo SE: VaR', output.getvalue())
        self.assertIn('generalized-t-skewed', ALL_JOINT_MODELS)
        p = preset('generalized-t-skewed', dimensions=3)
        model = joint_distribution(p)
        np.testing.assert_allclose(model.mean(), p['mean'], atol=1e-12)
        np.testing.assert_allclose(model.cov(), p['covariance'], atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fits.json').write_text(json.dumps([fit]), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                code = portfolio_main([str(root/'fits.json'), '--weights', 'A=.6', 'B=.4',
                    '--risk-levels', '.95', '--simulations', '2000', '--output', str(root/'risk.csv')])
            self.assertEqual(code, 0)
            saved = pd.read_csv(root/'risk.csv')
            self.assertEqual(saved.loc[0, 'method'], 'generalized-t-monte-carlo')
            self.assertTrue(np.isfinite(saved.loc[0, 'es_0.95']))


if __name__ == '__main__':
    unittest.main()
