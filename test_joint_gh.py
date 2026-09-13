import unittest
import json
import numpy as np
from scipy import stats
from return_distributions import fit_joint, joint_distribution
from return_distributions.joint_gh import JointGH, GH_MODELS


class GHTests(unittest.TestCase):
    def fixture(self, d=1, lam=-.5):
        return dict(location=[.1]*d, scatter=(np.eye(d)*.49).tolist(), gamma=[-.2]*d,
                    psi=1.3, **{'lambda': lam})

    def test_univariate_pdf_and_moments(self):
        for lam in [-3., -.5, .7, 1., 4.]:
            fit = self.fixture(lam=lam)
            model = JointGH(fit)
            b = -.2/.7
            a = np.sqrt(1.3+b*b)
            reference = stats.genhyperbolic(lam, a, b, loc=.1, scale=.7)
            x = np.linspace(-4, 4, 71)
            np.testing.assert_allclose(model.logpdf(x[:, None]), reference.logpdf(x), atol=1e-12)
            np.testing.assert_allclose(model.mean(), reference.mean(), atol=1e-12)
            np.testing.assert_allclose(model.cov(), reference.var(), atol=1e-12)

    def test_simulation_moments(self):
        model = JointGH(self.fixture(d=2))
        x = model.rvs(60000, random_state=19)
        np.testing.assert_allclose(x.mean(axis=0), model.mean(), atol=.015)
        np.testing.assert_allclose(np.cov(x.T), model.cov(), atol=.02)

    def test_all_fits(self):
        x = JointGH(self.fixture(d=2)).rvs(160, random_state=71)
        fitted = {}
        for name in GH_MODELS:
            fit = fit_joint(x, name)
            fitted[name] = fit
            self.assertTrue(fit['converged'], str(fit))
            self.assertEqual(fit['parameters'], (6 if name.endswith('symmetric') else 8)+int(name.startswith('gh-')))
            if not name.startswith('gh-'):
                self.assertEqual(fit['lambda'], 1.5 if name.startswith('hyperbolic') else -.5)
            if name.endswith('symmetric'):
                np.testing.assert_equal(fit['gamma'], [0, 0])
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)
            self.assertGreater(np.linalg.eigvalsh(fit['covariance']).min(), 0)
            json.dumps(fit, allow_nan=False)
        for family in ['hyperbolic', 'nig', 'gh']:
            self.assertGreaterEqual(fitted[family+'-skewed']['loglik']+1e-5, fitted[family+'-symmetric']['loglik'])
        for symmetry in ['symmetric', 'skewed']:
            for family in ['hyperbolic', 'nig']:
                self.assertGreaterEqual(fitted['gh-'+symmetry]['loglik']+1e-5, fitted[family+'-'+symmetry]['loglik'])

    def test_fixed_location_and_nonconvergence(self):
        x = JointGH(self.fixture(d=2)).rvs(100, random_state=23)
        fit = fit_joint(x, 'nig-symmetric', location=0, max_iterations=1)
        np.testing.assert_allclose(fit['location'], [0, 0], atol=1e-16)
        self.assertEqual(fit['parameters'], 4)
        self.assertFalse(fit['converged'])

    def test_free_lambda_fixed_location(self):
        x = JointGH(self.fixture(d=2, lam=.7)).rvs(120, random_state=44)
        for name, count in [('gh-symmetric', 5), ('gh-skewed', 7)]:
            fit = fit_joint(x, name, location=0, max_iterations=1)
            self.assertEqual(fit['parameters'], count)
            np.testing.assert_allclose(fit['location'], 0., atol=1e-16)
            self.assertNotEqual(fit['status'], 'ok')
            self.assertTrue(-20 <= fit['lambda'] <= 20)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)


if __name__ == '__main__':
    unittest.main()
