"""Test accelerated skew-t copula quantiles against reference calculations.
Cover likelihood derivatives, interpolation fallbacks, and final fit audits.
"""

import unittest
from unittest.mock import patch
import numpy as np
from return_distributions.skew_t_copula import _fast_marginal_ppf, marginal_ppf, marginal_cdf, latent_model, logpdf
from return_distributions.skew_t import azzalini_skew_t
from return_distributions.copulas import fit_copula, sample_copula


class FastSkewCopulaTests(unittest.TestCase):
    def test_quantiles_against_newton(self):
        u = np.r_[1e-10, np.linspace(.0001, .9999, 600), 1-1e-10]
        for df, a in [(3.4, -.58), (4.85, -.97), (1., 5.), (.25, -12.), (100., .01), (5., 0.)]:
            exact, fast = marginal_ppf(u, df, a), _fast_marginal_ppf(u, df, a)
            np.testing.assert_allclose(fast, exact, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(marginal_cdf(fast, df, a), u, atol=2e-10)

    def test_high_df_shape_against_adaptive_cdf(self):
        # The original Newton routine can stall here; independently audit the
        # interpolated answer rather than treating that routine as an oracle.
        u = np.linspace(.001, .999, 80)
        x = _fast_marginal_ppf(u, 200., 12.)
        np.testing.assert_allclose(azzalini_skew_t.cdf(x, 200., 12.), u, atol=2e-10)

    def test_likelihood_and_finite_differences(self):
        u = np.random.default_rng(4).uniform(.0001, .9999, (600, 2))
        theta = np.array([-.3, 3.4, -.58, .033])
        def likelihood(theta, fast):
            rho, df, *alpha = theta
            corr = [[1, rho], [rho, 1]]
            if not fast: return logpdf(u, corr, df, alpha).sum()
            model, shapes = latent_model(corr, df, alpha)
            z = np.column_stack([_fast_marginal_ppf(u[:, j], df, a) for j, a in enumerate(shapes)])
            return (model.logpdf(z)-sum(azzalini_skew_t.logpdf(z[:, j], df, a) for j, a in enumerate(shapes))).sum()
        exact, fast = likelihood(theta, False), likelihood(theta, True)
        self.assertAlmostEqual(exact, fast, places=7)
        for j in range(4):
            shifted = theta.copy(); shifted[j] += 1e-5
            self.assertAlmostEqual((likelihood(shifted, False)-exact)/1e-5,
                                   (likelihood(shifted, True)-fast)/1e-5, delta=.002)

    def test_bad_table_falls_back(self):
        u = np.linspace(.01, .99, 300)
        with patch('return_distributions.skew_t_copula._quintic_inverse', side_effect=ValueError('test')):
            np.testing.assert_array_equal(_fast_marginal_ppf(u, 5, -.5), marginal_ppf(u, 5, -.5))

    def test_fit_records_exact_audit(self):
        u = sample_copula(dict(model='student-t', correlation=[[1,.3],[.3,1]], df=5), 260, 7)
        f = fit_copula(u, 'azzalini-skew-t', 3)
        self.assertEqual(f['quantile_method'], 'audited-quintic')
        self.assertLess(f['likelihood_audit_error'], 1e-5)
        self.assertAlmostEqual(f['loglik'], logpdf(u, f['correlation'], f['df'], f['alpha']).sum(), places=9)
        self.assertEqual(len(f['attempts']), 5)


if __name__ == '__main__': unittest.main()
