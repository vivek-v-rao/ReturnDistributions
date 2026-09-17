"""Test skewed multivariate power-exponential laws and their limiting cases.
Cover sampling, portfolio risk, fixed power, and asset-order dependence.
"""

import json
import unittest

import numpy as np
from scipy import integrate, stats

from return_distributions.joint_power import JointPower
from return_distributions.skew_ged import skew_ged
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.joint_portfolio_mc import simulate_joint_portfolio
from return_distributions.joint_cli import fitting_tasks
from return_distributions.simulation import preset


class JointSkewPowerTests(unittest.TestCase):
    def test_univariate_and_normal_limits(self):
        for p in (1., 1.4, 2.):
            for s in (-.4, 0., .5):
                f = JointPower(dict(location=[.2], scatter=[[.49]], power=p, skewness=[s]))
                ref = skew_ged(p, s, loc=.2, scale=.7)
                x = np.linspace(-5, 5, 101)
                np.testing.assert_allclose(f.pdf(x[:, None]), ref.pdf(x), rtol=1e-12)
                np.testing.assert_allclose(f.mean(), [ref.mean()], atol=1e-12)
                np.testing.assert_allclose(f.cov(), [[ref.var()]], atol=1e-12)
                self.assertAlmostEqual(integrate.quad(lambda t: f.pdf([t]), -np.inf, .2)[0]
                                       +integrate.quad(lambda t: f.pdf([t]), .2, np.inf)[0], 1., places=7)
        scatter = np.array([[2., .5], [.5, 1.]])
        f = JointPower(dict(location=[0., 0.], scatter=scatter, power=2.))
        x = np.random.default_rng(1).normal(size=(20, 2))
        np.testing.assert_allclose(f.logpdf(x), stats.multivariate_normal(cov=scatter/2).logpdf(x))
        np.testing.assert_allclose(f.cov(), scatter/2)

    def test_sampling_and_portfolio(self):
        for model in ('ged-skewed', 'fs-skew-normal'):
            record = dict(preset(model, 3), status='ok')
            f = joint_distribution(record)
            x = f.rvs(150000, 23)
            np.testing.assert_allclose(x.mean(axis=0), f.mean(), atol=.0002)
            np.testing.assert_allclose(np.cov(x.T), f.cov(), atol=.00001, rtol=.025)
            w = np.array([.6, -.2, .6])
            a, draws = simulate_joint_portfolio(record, w, simulations=10000)
            b, repeat = simulate_joint_portfolio(record, w, simulations=10000)
            np.testing.assert_array_equal(draws, repeat)
            self.assertEqual(a['mean'], float(w@f.mean()))
            self.assertGreater(a['es_0.99'], a['var_0.99'])
            self.assertEqual(a['method'], 'skew-power-monte-carlo')

    def test_fit_roundtrip_and_fixed_power(self):
        for model, k in [('ged-skewed', 8), ('fs-skew-normal', 7)]:
            data = joint_distribution(preset(model)).rvs(400, 12)
            for location in (None, 0.):
                fit = fit_joint(data, model, location=location, max_iterations=700)
                self.assertEqual(fit['status'], 'ok')
                self.assertEqual(fit['parameters'], k if location is None else k-2)
                frozen = joint_distribution(json.loads(json.dumps(fit)))
                self.assertAlmostEqual(frozen.logpdf(data).sum(), fit['loglik'], places=6)
                if model == 'fs-skew-normal':
                    self.assertEqual(fit['power'], 2.)
                if location is not None:
                    np.testing.assert_array_equal(fit['location'], [0., 0.])

    def test_asset_orders(self):
        tasks = list(fitting_tasks(['ged', 'ged-skewed', 'fs-skew-normal'], ['A', 'B'], asset_orders='all'))
        self.assertEqual(len(tasks), 5)
        self.assertEqual(sum(t[0] == 'ged' for t in tasks), 1)


if __name__ == '__main__':
    unittest.main()
