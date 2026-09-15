import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import integrate, stats

from return_distributions.skew_ged import skew_ged
from return_distributions.fitting import fit_one, fitted_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copula_cli import main as copula_main


class SkewGEDTests(unittest.TestCase):
    def test_symmetric_limit_reflection_and_tails(self):
        x = np.array([-10., -2., 0., .5, 3., 12.])
        probs = np.array([1e-12, .001, .05, .5, .95, .999, 1-1e-12])
        for power in [.5, 1., 1.25, 2., 5.]:
            for method in ['pdf', 'logpdf', 'cdf', 'sf', 'ppf', 'isf']:
                values = probs if method in ['ppf', 'isf'] else x
                np.testing.assert_allclose(getattr(skew_ged, method)(values, power, 0),
                                           getattr(stats.gennorm, method)(values, power), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(skew_ged.stats(power, 0, moments='mvsk'),
                                       stats.gennorm.stats(power, moments='mvsk'), rtol=1e-10, atol=1e-12)
            for skew in [-3., -.5, .5, 3.]:
                f = skew_ged(power, skew)
                np.testing.assert_allclose(f.cdf(f.ppf(probs)), probs, rtol=1e-9, atol=1e-14)
                np.testing.assert_allclose(f.sf(f.isf(probs)), probs, rtol=1e-9, atol=1e-14)
                np.testing.assert_allclose(f.pdf(x), skew_ged.pdf(-x, power, -skew))
        self.assertTrue(np.isnan(skew_ged.pdf(0, 0, 0)))

    def test_density_moments_and_es(self):
        f = skew_ged(1.25, -.35)
        mean, variance, skew, kurt = f.stats(moments='mvsk')
        sd = np.sqrt(variance)
        for power, expected in [(0, 1.), (1, mean), (2, variance+mean**2)]:
            value = sum(integrate.quad(lambda x: x**power*f.pdf(x), a, b)[0]
                        for a, b in [(-np.inf, 0), (0, np.inf)])
            self.assertAlmostEqual(value, expected, places=7)
        sample = f.rvs(size=150000, random_state=10)
        self.assertLess(abs(sample.mean()-mean), .015*sd)
        self.assertLess(abs(sample.var()/variance-1), .025)
        for confidence in [.05, .95, .99]:
            shifted = skew_ged(1.25, -.35, loc=.001, scale=.02)
            q = shifted.ppf(1-confidence)
            expected = -integrate.quad(lambda p: shifted.ppf(p), 0, 1-confidence, epsabs=1e-10)[0]/(1-confidence)
            risk = portfolio_risk(shifted, confidence)
            self.assertAlmostEqual(risk['var'], -q, places=12)
            self.assertAlmostEqual(risk['es'], expected, places=7)

    def test_fitting_serialization_and_iteration_limit(self):
        x = skew_ged.rvs(1.5, -.4, loc=.001, scale=.02, size=600, random_state=24)
        row = fit_one(x, 'ged-skewed', max_iterations=1800)
        self.assertEqual(row['status'], 'ok')
        self.assertEqual(row['k'], 4)
        self.assertLess(row['skewness'], 0.)
        frozen = fitted_distribution(json.loads(json.dumps(row)))
        self.assertAlmostEqual(frozen.logpdf(x).sum(), row['loglik'], places=7)
        self.assertAlmostEqual(row['aic'], 8-2*row['loglik'])
        limited = fit_one(x, 'ged-skewed', location=0., max_iterations=1)
        self.assertEqual(limited['status'], 'not_converged')
        self.assertEqual(limited['k'], 3)
        self.assertAlmostEqual(limited['loc'], 0.)

    def test_copula_cli(self):
        rng = np.random.default_rng(12)
        z = rng.multivariate_normal([0., 0.], [[1., .4], [.4, 1.]], size=120)
        returns = skew_ged.ppf(stats.norm.cdf(z), 1.5, -.3)*.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame(returns, columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=120)).to_csv(path/'returns.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                result = copula_main([str(path/'returns.csv'), '--input-type', 'returns',
                                      '--marginal-models', 'ged-skewed', '--copulas', 'gaussian',
                                      '--output', str(path/'fits.json')])
            self.assertEqual(result, 0)
            self.assertTrue((path/'fits.json').exists())


if __name__ == '__main__':
    unittest.main()
