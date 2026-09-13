import unittest
import numpy as np
from scipy import stats
from return_distributions import project_distribution, portfolio_risk, joint_distribution
from return_distributions.simulation import preset
from return_distributions.multivariate import JOINT_MODELS


class TailTests(unittest.TestCase):
    def test_normal_and_t(self):
        risk = portfolio_risk(stats.norm(.001, .02), .95)
        self.assertAlmostEqual(risk['var'], -.001+.02*1.644853626951)
        self.assertAlmostEqual(risk['es'], -.001+.02*2.062712807508)
        self.assertTrue(np.isinf(portfolio_risk(stats.t(.8), .99)['es']))
        self.assertTrue(np.isfinite(portfolio_risk(stats.t(1.5), .99)['es']))
        self.assertLess(portfolio_risk(stats.norm(1, .01), .95)['es'], 0)
        with self.assertRaises(ValueError): portfolio_risk(stats.norm(), 1)

    def test_every_family_monte_carlo(self):
        w = np.array([.6, -.4])
        for model in JOINT_MODELS:
            fit = preset(model)
            dist = project_distribution(fit, w)
            risk = portfolio_risk(dist, .95)
            x = joint_distribution(fit).rvs(150000, random_state=np.random.default_rng(62))@w
            empirical = -x[x <= np.quantile(x, .05)].mean()
            self.assertAlmostEqual(risk['es'], empirical, delta=.0007, msg=model)
            self.assertGreater(risk['es'], risk['var'])
            self.assertGreater(portfolio_risk(dist, .99)['es'], risk['es'])

    def test_zero_and_numerical_normal(self):
        from return_distributions.projection import ProjectedPower
        fit = preset('normal')
        self.assertEqual(portfolio_risk(project_distribution(fit, [0, 0]))['es'], 0)
        for confidence in [.1, .5, .99]:
            ref = portfolio_risk(stats.norm(.1, .7/np.sqrt(2)), confidence)
            actual = portfolio_risk(ProjectedPower(.1, .7, 3, 2), confidence)
            self.assertAlmostEqual(ref['es'], actual['es'], places=8)


if __name__ == '__main__':
    unittest.main()
