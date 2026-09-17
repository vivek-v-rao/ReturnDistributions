"""Regression tests for accelerated skew-t quantiles and cached copula fitting.

Check quantile accuracy, broadcasting, location/scale handling, symmetry,
endpoint behavior, and numerical fallbacks. Verify that Gaussian fitting reuses
normal quantiles and that Gaussian/Student-t fitted likelihoods agree with
direct evaluation. These are correctness tests, not timing benchmarks.
"""

import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from scipy import stats
from return_distributions.skew_t import azzalini_skew_t, fs_skew_t
from return_distributions.copulas import fit_copula, copula_logpdf


class CopulaSpeedTests(unittest.TestCase):
    def test_fast_ppf_matches_original(self):
        u = np.array([1e-6, .01, .2, .5, .9, .99, 1-1e-6])
        for df, a in [(1., -2.), (5., .5), (100., 3.)]:
            expected = stats.rv_continuous._ppf(azzalini_skew_t, u, df, a)
            actual = azzalini_skew_t.ppf(u, df, a)
            np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)

    def test_broadcast_scale_and_symmetric(self):
        u = np.array([[.1, .5], [.8, .95]])
        df, a = np.array([4., 8.]), np.array([0., -.5])
        x = azzalini_skew_t.ppf(u, df, a, loc=3, scale=2)
        np.testing.assert_allclose(azzalini_skew_t.cdf(x, df, a, loc=3, scale=2), u, atol=1e-10)
        np.testing.assert_allclose(x[:, 0], 3+2*stats.t.ppf(u[:, 0], 4))
        np.testing.assert_allclose(fs_skew_t.cdf(fs_skew_t.ppf(u, 4, .5), 4, .5), u, atol=1e-10)
        np.testing.assert_equal(azzalini_skew_t.ppf([0., 1.], 5, .5), [-np.inf, np.inf])

    def test_failure_and_extreme_tail_fallback(self):
        u = np.array([.1, .5, .9])
        expected = stats.rv_continuous._ppf(azzalini_skew_t, u, 5., .5)
        for replacement in [None, np.full(3, 99.)]:
            with patch('return_distributions.skew_t_copula.marginal_ppf',
                       **({'side_effect': ValueError('test')} if replacement is None else {'return_value': replacement})):
                np.testing.assert_allclose(azzalini_skew_t.ppf(u, 5, .5), expected)
        with patch('return_distributions.skew_t_copula.marginal_ppf', side_effect=AssertionError('must use fallback')):
            x = azzalini_skew_t.ppf(1e-9, 5, .5)
        self.assertAlmostEqual(azzalini_skew_t.cdf(x, 5, .5)/1e-9, 1., places=6)

    def test_cached_likelihood(self):
        u = np.random.default_rng(17).uniform(.01, .99, (100, 2))
        with patch('return_distributions.copulas.stats.norm.ppf', wraps=stats.norm.ppf) as ppf:
            f = fit_copula(u, 'gaussian')
            self.assertEqual(ppf.call_count, 1)
        for model in ['gaussian', 'student-t']:
            f = fit_copula(u, model)
            self.assertAlmostEqual(f['loglik'], copula_logpdf(u, model, f['correlation'], f['df']).sum(), places=9)


if __name__ == '__main__': unittest.main()
