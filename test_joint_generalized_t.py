import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import integrate, stats

from return_distributions.generalized_t import generalized_t
from return_distributions.joint_generalized_t import JointGeneralizedT, ProjectedGeneralizedT, covariance_factor
from return_distributions.joint_power import JointPower
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.projection import project_distribution, PointMass, ProjectedPower
from return_distributions.tail_risk import portfolio_risk
from return_distributions.simulation import preset
from return_distributions.joint_cli import main


def truth(d=2, power=1.5, q=5.):
    return dict(model='generalized-t', power=power, q=q, status='ok',
                location=np.zeros(d).tolist(), scatter=np.eye(d).tolist())


class JointGeneralizedTTests(unittest.TestCase):
    def test_density_nesting_and_normalization(self):
        rng = np.random.default_rng(1)
        for d in [1, 2, 4]:
            f = JointGeneralizedT(truth(d, 2., 3.))
            x = rng.normal(size=(12, d))
            ref = stats.multivariate_t(loc=np.zeros(d), shape=np.eye(d)/2, df=6.)
            np.testing.assert_allclose(f.logpdf(x), ref.logpdf(x), atol=1e-12)
            np.testing.assert_allclose(f.cov(), np.eye(d)*.75)
            g = JointGeneralizedT(truth(d, 1.5, 1e7))
            ged = JointPower(truth(d, 1.5, 1e7))
            np.testing.assert_allclose(g.logpdf(x), ged.logpdf(x), atol=2e-5)
        x = np.linspace(-4, 4, 15)
        g = JointGeneralizedT(truth(1, 1.25, 4.))
        np.testing.assert_allclose(g.pdf(x[:, None]), generalized_t.pdf(x, 1.25, 4., 0.), rtol=1e-12)
        f = JointGeneralizedT(truth())
        total = integrate.quad(lambda r: 2*np.pi*r*f.pdf([r, 0.]), 0, np.inf)[0]
        self.assertAlmostEqual(total, 1., places=8)

    def test_student_t_projection_direct_integrals(self):
        # Test numerical implementation itself, not only the p=2 shortcut.
        for d in [2, 3, 5]:
            f = ProjectedGeneralizedT(.1, .7, d, 2., 3.)
            ref = stats.t(6., loc=.1, scale=.7/np.sqrt(2))
            for x in [-3., 0., .1, 2.]:
                self.assertAlmostEqual(float(f.pdf(x)), ref.pdf(x), places=8)
                self.assertAlmostEqual(float(f.cdf(x)), ref.cdf(x), places=8)
            for probability in [.001, .05, .5, .95, .999]:
                self.assertAlmostEqual(float(f.ppf(probability)), ref.ppf(probability), places=7)
            for confidence in [.05, .95, .99]:
                self.assertAlmostEqual(portfolio_risk(f, confidence)['es'], portfolio_risk(ref, confidence)['es'], places=7)

    def test_general_projection_and_sampling(self):
        fit = preset('generalized-t', dimensions=3)
        f = joint_distribution(fit)
        weights = np.array([.7, -.3, .6])
        projected = project_distribution(fit, weights)
        self.assertEqual(projected.dimension, 3)
        self.assertAlmostEqual(projected.var(), weights@f.cov()@weights, places=12)
        draws = f.rvs(150000, random_state=10)@weights
        self.assertLess(abs(draws.mean()-projected.mean()), .015*projected.std())
        self.assertLess(abs(draws.var()/projected.var()-1), .03)
        for probability in [.01, .05, .5, .95]:
            quantile = float(projected.ppf(probability))
            self.assertAlmostEqual(float(projected.cdf(quantile)), probability, places=8)
            self.assertLess(abs(np.mean(draws <= quantile)-probability), .006)
        quantile = float(projected.ppf(.05))
        estimated_es = -draws[draws <= quantile].mean()
        self.assertLess(abs(portfolio_risk(projected, .95)['es']/estimated_es-1), .025)
        one = ProjectedGeneralizedT(0., 1., 2, 1.5, 5.)
        original = JointGeneralizedT(truth())
        for x in [0., .5, 2.]:
            expected = integrate.quad(lambda y: original.pdf([x, y]), -np.inf, np.inf)[0]
            self.assertAlmostEqual(float(one.pdf(x)), expected, places=8)
        # q->infinity gives the existing dimension-aware GED projection.
        limit = ProjectedGeneralizedT(0., 1., 3, 1.5, 1e6)
        ged = ProjectedPower(0., 1., 3, 1.5)
        self.assertAlmostEqual(float(limit.cdf(.7)), float(ged.cdf(.7)), places=5)
        self.assertAlmostEqual(portfolio_risk(limit)['es'], portfolio_risk(ged)['es'], places=4)

    def test_moment_thresholds_and_weights(self):
        for index in [.8, 1., 1.5, 2., 3.]:
            fit = truth(2, 1., index)
            joint = JointGeneralizedT(fit)
            f = project_distribution(fit, [1., -.2])
            self.assertEqual(np.isfinite(f.mean()), index > 1)
            self.assertEqual(np.isfinite(f.var()), index > 2)
            self.assertEqual(joint.cov() is not None, index > 2)
            self.assertEqual(np.isfinite(portfolio_risk(f)['es']), index > 1)
        self.assertIsInstance(project_distribution(truth(), [0., 0.]), PointMass)
        for w in [[1.], [np.nan, 1.]]:
            with self.assertRaises(ValueError): project_distribution(truth(), w)
        with self.assertRaises(ValueError): project_distribution(dict(truth(), status='boundary'), [1., 0.])

    def test_fitting_serialization_and_cli_risk(self):
        original = truth(2, 1.5, 3.)
        x = JointGeneralizedT(original).rvs(500, random_state=3)
        for location, k in [(None, 7), (0., 5)]:
            fit = fit_joint(x, 'generalized-t', location=location, max_iterations=1000)
            self.assertEqual(fit['status'], 'ok')
            self.assertEqual(fit['parameters'], k)
            model = joint_distribution(json.loads(json.dumps(fit, allow_nan=False)))
            self.assertAlmostEqual(model.logpdf(x).sum(), fit['loglik'], places=6)
            if location is not None: np.testing.assert_allclose(fit['location'], 0., atol=1e-12)
        limited = fit_joint(x, 'generalized-t', max_iterations=1)
        self.assertEqual(limited['status'], 'not_converged')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = pd.DataFrame(x*.01, columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=500))
            data.to_csv(root/'returns.csv')
            # Isolate CLI/CSV/risk plumbing from numerical optimizer variation.
            fixture = fit.copy()
            with patch('return_distributions.joint_cli.fit_joint', return_value=fixture), contextlib.redirect_stdout(io.StringIO()):
                code = main([str(root/'returns.csv'), '--input-type', 'returns', '--models', 'generalized-t',
                    '--weights', 'A=.6', 'B=.4', '--risk-levels', '.95', '.99', '--output', str(root/'fits.json')])
            self.assertEqual(code, 0)
            risk = pd.read_csv(root/'fits_portfolio_risk.csv')
            self.assertEqual(set(risk.model), {'generalized-t', 'empirical'})


if __name__ == '__main__':
    unittest.main()
