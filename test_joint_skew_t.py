"""Test multivariate Azzalini skew-t densities, projections, and moments.
Cover finite expected shortfall with infinite variance and fit reconstruction.
"""

import unittest
import numpy as np
from scipy import stats
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.projection import project_distribution
from return_distributions.simulation import preset
from return_distributions.skew_t import azzalini_skew_t
from return_distributions.tail_risk import portfolio_risk


class JointSkewTTests(unittest.TestCase):
    def test_density_limits(self):
        x = np.linspace(-4, 4, 51)
        fit = dict(model='azzalini-skew-t', location=[.2], scatter=[[1.44]], alpha=[-2.], df=5.)
        np.testing.assert_allclose(joint_distribution(fit).pdf(x[:, None]), azzalini_skew_t.pdf(x, 5, -2, loc=.2, scale=1.2), rtol=1e-12)
        fit = preset('azzalini-skew-t'); fit['alpha'] = [0., 0.]
        data = np.array([[.01, -.02], [.02, .03]])
        np.testing.assert_allclose(joint_distribution(fit).pdf(data), stats.multivariate_t.pdf(data, loc=fit['location'], shape=fit['scatter'], df=fit['df']), rtol=1e-12)

    def test_projection_and_moments(self):
        fit = preset('azzalini-skew-t')
        dist = joint_distribution(fit)
        x = dist.rvs(150000, random_state=84)
        np.testing.assert_allclose(x.mean(axis=0), dist.mean(), atol=.0002)
        np.testing.assert_allclose(np.cov(x.T), dist.cov(), rtol=.04, atol=1e-6)
        for w in ([.6, .4], [-.4, 1.3]):
            projected = project_distribution(fit, w)
            np.testing.assert_allclose(projected.mean(), np.array(w)@dist.mean(), atol=1e-12)
            np.testing.assert_allclose(projected.var(), np.array(w)@dist.cov()@w, rtol=1e-12)
            np.testing.assert_allclose(np.quantile(x@w, [.05, .5, .95]), projected.ppf([.05, .5, .95]), atol=.0005)
            risk = portfolio_risk(projected, .95)
            sample = x@w
            self.assertAlmostEqual(risk['es'], -sample[sample <= np.quantile(sample, .05)].mean(), delta=.001)

    def test_infinite_variance_finite_es(self):
        fit = preset('azzalini-skew-t'); fit['df'] = 1.5
        self.assertTrue(np.isfinite(portfolio_risk(project_distribution(fit, [.5, .5]))['es']))
        fit['df'] = .9
        self.assertTrue(np.isinf(portfolio_risk(project_distribution(fit, [.5, .5]))['es']))

    def test_fit_roundtrip(self):
        x = joint_distribution(preset('azzalini-skew-t')).rvs(250, random_state=123)
        for loc in (None, 0.):
            fit = fit_joint(x, 'azzalini-skew-t', location=loc, max_iterations=500)
            self.assertEqual(fit['parameters'], 8 if loc is None else 6)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)
            self.assertTrue(np.linalg.eigvalsh(fit['scatter']).min() > 0)
            if loc is not None: np.testing.assert_allclose(fit['location'], 0., atol=1e-16)


if __name__ == '__main__': unittest.main()
