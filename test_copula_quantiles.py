import unittest
from unittest.mock import patch
import numpy as np
from scipy import stats
from return_distributions.copula_quantiles import nig_ppf, marginal_ppf, _inverse
from return_distributions.copulas import CopulaJoint


class NIGQuantileTests(unittest.TestCase):
    def test_reference_and_reflection(self):
        u = np.r_[1e-6, np.linspace(.01, .99, 13), 1-1e-6]
        for a, b in [(.35861248165, -.03982260778), (2., 0.), (3., 2.5)]:
            x = nig_ppf(u, a, b)
            # SciPy's generic upper-tail bracket can fail for skewed NIG.
            # Use its independently implemented lower-tail PPF and reflection.
            reference = stats.norminvgauss.ppf(np.minimum(u, 1-u), a, np.where(u > .5, -b, b))
            reference = np.where(u > .5, -reference, reference)
            np.testing.assert_allclose(x, reference, atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(stats.norminvgauss.cdf(x, a, b), u, atol=1e-9, rtol=0)
            np.testing.assert_allclose(nig_ppf(1-u, a, -b), -x, atol=3e-6, rtol=1e-6)

    def test_tails_and_endpoints(self):
        a, b = .35861248165, -.03982260778
        for u in [1e-12, 1e-9, 1-1e-9, 1-1e-12]:
            x = nig_ppf(u, a, b)
            p = stats.norminvgauss.cdf(-x if u > .5 else x, a, -b if u > .5 else b)
            self.assertAlmostEqual(p/min(u, 1-u), 1., delta=2e-3)
        np.testing.assert_equal(nig_ppf([0., 1., -.1, 1.1, np.nan], a, b),
                                [-np.inf, np.inf, np.nan, np.nan, np.nan])

    def test_cache_dispatch_and_scale(self):
        _inverse.cache_clear()
        u = np.array([[.1, .5], [.8, .9]])
        plain = nig_ppf(u, 2., .1)
        misses = _inverse.cache_info().misses
        dist = stats.norminvgauss(a=2., b=.1, loc=.01, scale=.02)
        np.testing.assert_allclose(marginal_ppf(dist, u), .01+.02*plain)
        self.assertEqual(_inverse.cache_info().misses, misses)
        np.testing.assert_equal(marginal_ppf(stats.norm(), u), stats.norm.ppf(u))
        self.assertIsInstance(nig_ppf(.5, 2., .1), float)

    def test_failed_setup_and_validation(self):
        _inverse.cache_clear()
        with patch('return_distributions.copula_quantiles.NumericalInversePolynomial', side_effect=RuntimeError('test')):
            with self.assertWarns(RuntimeWarning): self.assertIsNone(_inverse(2., .1))
        _inverse.cache_clear()
        with patch('return_distributions.copula_quantiles._inverse', return_value=None):
            x = nig_ppf([.1, .5, .9], 2., .1)
        np.testing.assert_allclose(stats.norminvgauss.cdf(x, 2., .1), [.1, .5, .9], atol=1e-9)
        for a, b in [(0., 0.), (1., 1.), (np.nan, 0.)]:
            with self.assertRaises(ValueError): nig_ppf(.5, a, b)
        with self.assertRaises(ValueError): nig_ppf(.5, 1., 0., scale=0)

    def test_saved_model_dispatch(self):
        marginal = dict(name='nig-skewed', a=2., b=.1, loc=0., scale=.01)
        joint = CopulaJoint(dict(marginal_mode='fitted', marginals=[marginal, marginal],
                                copula=dict(model='gaussian', correlation=[[1., .3], [.3, 1.]], df=None)))
        x = joint.rvs(500, 4)
        self.assertEqual(x.shape, (500, 2))
        self.assertTrue(np.isfinite(joint.logpdf(x)).all())
        np.testing.assert_equal(x, joint.rvs(500, 4))


if __name__ == '__main__': unittest.main()
