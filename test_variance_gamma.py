"""Test variance-gamma densities, CDFs, moments, and portfolio projections.
Cover special cases and univariate/joint fit reconstruction.
"""

import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.variance_gamma import variance_gamma, JointVG, VG_MODELS
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.fitting import fit_one, fitted_distribution
from return_distributions.simulation import preset
from return_distributions.projection import project_distribution
from return_distributions.tail_risk import portfolio_risk


class TestVarianceGamma(unittest.TestCase):
    def test_density_and_cdf(self):
        # Gamma(1,1) normal mixture is Laplace with scale 1/sqrt(2).
        x = np.array([-3., -.2, 0., .3, 2.])
        np.testing.assert_allclose(variance_gamma.pdf(x, 1, 0), stats.laplace.pdf(x, scale=1/np.sqrt(2)), rtol=1e-12)
        f = variance_gamma(2.3, .4)
        self.assertAlmostEqual(integrate.quad(f.pdf, -np.inf, np.inf)[0], 1., places=7)
        np.testing.assert_allclose(f.cdf(x)+f.sf(x), 1., atol=1e-8)
        np.testing.assert_allclose(f.cdf(f.ppf([.05,.5,.95])), [.05,.5,.95], atol=1e-8)

    def test_projection_and_moments(self):
        truth = preset(VG_MODELS[1], 2)
        joint = JointVG(truth)
        point = np.array([.01,-.005])
        reference = integrate.quad(lambda s: stats.gamma.pdf(s,joint.shape,scale=1/joint.shape)*stats.multivariate_normal.pdf(point,mean=joint.location+s*joint.gamma,cov=s*joint.scatter),0,np.inf)[0]
        self.assertAlmostEqual(joint.pdf(point)/reference, 1., places=8)
        draws = joint.rvs(150000, random_state=24)
        np.testing.assert_allclose(draws.mean(axis=0), joint.mean(), atol=.00015)
        np.testing.assert_allclose(np.cov(draws.T), joint.cov(), atol=4e-6)
        truth.update(status='ok', dimensions=2, return_type='simple')
        w = np.array([.7,-.3])
        projected = project_distribution(truth, w)
        self.assertAlmostEqual(projected.mean(), w@joint.mean())
        self.assertAlmostEqual(projected.var(), w@joint.cov()@w)
        sample = draws@w
        self.assertAlmostEqual(projected.ppf(.05), np.quantile(sample,.05), delta=.00025)
        risk = portfolio_risk(projected, .95)
        self.assertEqual(risk['es_status'], 'finite')
        self.assertAlmostEqual(risk['es'], -sample[sample <= np.quantile(sample,.05)].mean(), delta=.0003)
        self.assertGreater(risk['es'], risk['var'])

    def test_fitting_roundtrip(self):
        for model in VG_MODELS:
            x = JointVG(preset(model, 2)).rvs(90, random_state=42)
            fit = fit_joint(x, model, max_iterations=2)
            self.assertEqual(fit['status'], 'not_converged')
            self.assertGreaterEqual(fit['vg_shape'], 1.05)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)
        x = variance_gamma.rvs(2., .3, size=40, random_state=42)
        row = fit_one(x, VG_MODELS[0], max_iterations=5)
        self.assertEqual(row['b'], 0.)
        self.assertAlmostEqual(fitted_distribution(row).logpdf(x).sum(), row['loglik'])
        row = fit_one(x, VG_MODELS[1], max_iterations=5, location=0.)
        self.assertAlmostEqual(row['loc'], 0.)
        self.assertEqual(row['k'], 3)


if __name__ == '__main__':
    unittest.main()
