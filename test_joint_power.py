"""Test elliptical power-exponential joint distributions and normal limits.
Cover univariate consistency, simulation, fitting, and fixed parameters.
"""

import json
import unittest
import numpy as np
from scipy import stats
from return_distributions import fit_joint, joint_distribution
from return_distributions.joint_power import JointPower


class PowerTests(unittest.TestCase):
    def model(self, p=1., d=2):
        return JointPower(dict(location=[.1]*d, scatter=(.49*np.eye(d)).tolist(), power=p))

    def test_univariate(self):
        for p in [.5, 1., 1.25, 2., 4.]:
            model = self.model(p, 1)
            x = np.linspace(-4, 4, 100)
            ref = stats.gennorm(p, loc=.1, scale=.7)
            np.testing.assert_allclose(model.logpdf(x[:, None]), ref.logpdf(x), atol=1e-12)
            np.testing.assert_allclose(model.cov(), ref.var(), atol=1e-12)
        np.testing.assert_allclose(self.model(1, 1).logpdf(x[:, None]), stats.laplace.logpdf(x, loc=.1, scale=.7))

    def test_normal_limit(self):
        model = self.model(2)
        x = np.random.default_rng(4).normal(size=(100, 2))
        np.testing.assert_allclose(model.logpdf(x), stats.multivariate_normal.logpdf(x, mean=[.1, .1], cov=.245*np.eye(2)))

    def test_simulation(self):
        model = self.model(1.25)
        x = model.rvs(80000, random_state=74)
        np.testing.assert_allclose(x.mean(axis=0), model.mean(), atol=.015)
        np.testing.assert_allclose(np.cov(x.T), model.cov(), atol=.025)

    def test_fits(self):
        x = self.model(1.25).rvs(300, random_state=36)
        fits = {}
        for name in ['laplace', 'ged']:
            fit = fit_joint(x, name)
            fits[name] = fit
            self.assertTrue(fit['converged'])
            self.assertFalse(fit['boundary'])
            self.assertEqual(fit['parameters'], 5 if name == 'laplace' else 6)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)
            self.assertGreater(np.linalg.eigvalsh(fit['covariance']).min(), 0)
            json.dumps(fit, allow_nan=False)
        self.assertEqual(fits['laplace']['power'], 1)
        self.assertTrue(.7 < fits['ged']['power'] < 2)
        self.assertGreaterEqual(fits['ged']['loglik']+1e-5, fits['laplace']['loglik'])

    def test_fixed_and_limit(self):
        x = self.model().rvs(100, random_state=82)
        fit = fit_joint(x, 'ged', location=0, max_iterations=1)
        np.testing.assert_allclose(fit['location'], [0, 0], atol=1e-16)
        self.assertEqual(fit['parameters'], 4)
        self.assertFalse(fit['converged'])


if __name__ == '__main__':
    unittest.main()
