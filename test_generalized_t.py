"""Test univariate generalized-t limits, densities, moments, and tail risk.
Cover quantiles, fitted-law reconstruction, and copula marginal integration.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import integrate, stats

from return_distributions.generalized_t import generalized_t
from return_distributions.skew_ged import skew_ged
from return_distributions.fitting import fit_one, fitted_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copula_cli import main as copula_main


class GeneralizedTTests(unittest.TestCase):
    def test_student_t_and_ged_limits(self):
        x = np.array([-8., -2., 0., .3, 3., 10.])
        probabilities = np.array([1e-10, .001, .05, .5, .95, .999, 1-1e-10])
        for q in [.5, 2., 5.]:
            gt = generalized_t(2., q, 0., loc=.001, scale=.02)
            t = stats.t(2*q, loc=.001, scale=.02/np.sqrt(2))
            for method in ['pdf', 'logpdf', 'cdf', 'sf', 'ppf', 'isf']:
                values = probabilities if method in ['ppf', 'isf'] else x*.01
                np.testing.assert_allclose(getattr(gt, method)(values), getattr(t, method)(values), rtol=2e-8, atol=1e-12)
            if q > .5:
                self.assertAlmostEqual(portfolio_risk(gt)['es'], portfolio_risk(t)['es'], places=9)
        for power in [.75, 1.25, 2.]:
            for skew in [0., -.3]:
                np.testing.assert_allclose(generalized_t.pdf(x, power, 1e7, skew),
                                           skew_ged.pdf(x, power, skew), rtol=.001, atol=1e-8)

    def test_moments_and_density(self):
        f = generalized_t(1.5, 5., -.3)
        for r, expected in [(0, 1.), (1, f.mean()), (2, f.var()+f.mean()**2)]:
            actual = sum(integrate.quad(lambda x: x**r*f.pdf(x), a, b)[0]
                         for a, b in [(-np.inf, 0), (0, np.inf)])
            self.assertAlmostEqual(actual, expected, places=7)
        for index in [.8, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5.]:
            f = generalized_t(1., index, 0.)
            mean, variance, skew, kurt = f.stats(moments='mvsk')
            self.assertEqual(np.isfinite(mean), index > 1)
            self.assertEqual(np.isfinite(variance), index > 2)
            self.assertEqual(np.isfinite(skew), index > 3)
            self.assertEqual(np.isfinite(kurt), index > 4)
            risk = portfolio_risk(f)
            self.assertEqual(np.isfinite(risk['es']), index > 1)
        self.assertTrue(np.isnan(generalized_t.pdf(0., 0., 1., 0.)))
        self.assertTrue(np.isnan(generalized_t.pdf(0., 1., -1., 0.)))

    def test_quantiles_sampling_and_partial_es(self):
        probs = np.array([1e-12, .001, .05, .5, .95, .999, 1-1e-12])
        flat = generalized_t(10., .1, 0.)
        near_mode = np.array([.499, .499999, .500001, .501])
        np.testing.assert_allclose(flat.cdf(flat.ppf(near_mode)), near_mode, atol=1e-13)
        for power, q, skew in [(1., 1.5, -.8), (.5, 5., .5), (2., 3., 3.), (1.2, 1000., -.3)]:
            f = generalized_t(power, q, skew, loc=.001, scale=.02)
            np.testing.assert_allclose(f.cdf(f.ppf(probs)), probs, rtol=2e-7, atol=1e-12)
            np.testing.assert_allclose(f.sf(f.isf(probs)), probs, rtol=2e-7, atol=1e-12)
            for confidence in [.05, .95, .99]:
                expected = -integrate.quad(lambda p: f.ppf(p), 0, 1-confidence, epsabs=1e-9)[0]/(1-confidence)
                self.assertAlmostEqual(portfolio_risk(f, confidence)['es'], expected, places=6)
        f = generalized_t(2., 5., -.4)
        sample = f.rvs(size=120000, random_state=3)
        self.assertLess(abs(sample.mean()-f.mean()), .015*f.std())
        self.assertLess(abs(sample.var()/f.var()-1), .025)

    def test_fit_roundtrip_and_limits(self):
        sample = generalized_t.rvs(2., 2., -.4, loc=.001, scale=.02, size=500, random_state=24)
        for name, k in [('generalized-t', 4), ('generalized-t-skewed', 5)]:
            row = fit_one(sample, name, max_iterations=2500)
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['k'], k)
            self.assertAlmostEqual(row['tail_index'], row['power']*row['q'])
            fitted = fitted_distribution(json.loads(json.dumps(row)))
            self.assertAlmostEqual(fitted.logpdf(sample).sum(), row['loglik'], places=7)
            limited = fit_one(sample, name, location=0., max_iterations=1)
            self.assertEqual(limited['status'], 'not_converged')
            self.assertEqual(limited['k'], k-1)
            self.assertAlmostEqual(limited['loc'], 0.)

    def test_copula_marginal_integration(self):
        sample = generalized_t.rvs(2., 3., -.3, size=(150, 2), random_state=4)*.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(sample, columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=150)).to_csv(root/'returns.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                result = copula_main([str(root/'returns.csv'), '--input-type', 'returns',
                    '--marginal-models', 'generalized-t', '--copulas', 'gaussian',
                    '--max-iterations', '2500', '--output', str(root/'fits.json')])
            self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
