import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import integrate, stats

from return_distributions.scipy_extra import crystal_ball
from return_distributions.fitting import fit_one, fitted_distribution, specification
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copula_cli import main as copula_main


class ScipyExtraTests(unittest.TestCase):
    def test_crystalball_scipy_equivalence_and_moments(self):
        x = np.array([-20., -2., -.1, 0., 1., 10.])
        p = np.array([1e-10, .001, .05, .5, .95, 1-1e-10])
        for beta, m in [(1., 2.5), (2., 6.), (.1, 100.)]:
            f = crystal_ball(beta, m)
            ref = stats.crystalball(beta, m)
            for method in ['pdf', 'logpdf', 'cdf', 'sf', 'ppf', 'isf']:
                values = p if method in ['ppf', 'isf'] else x
                np.testing.assert_allclose(getattr(f, method)(values), getattr(ref, method)(values), rtol=1e-12)
            if m == 6.:
                np.testing.assert_allclose(f.stats(moments='mvsk'), ref.stats(moments='mvsk'), rtol=1e-10)
            np.testing.assert_allclose(f.rvs(size=30, random_state=1), ref.rvs(size=30, random_state=1))
        for m in [1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6.]:
            mean, var, skew, kurt = crystal_ball.stats(1., m, moments='mvsk')
            for value, cutoff in [(mean, 2), (var, 3), (skew, 4), (kurt, 5)]:
                self.assertEqual(np.isfinite(value), m > cutoff)
            self.assertEqual(np.isfinite(portfolio_risk(crystal_ball(1., m))['es']), m > 2)
            if m <= 2:
                self.assertTrue(np.isneginf(mean))

    def test_analytic_expected_shortfall(self):
        families = [stats.johnsonsu(.7, 1.3, loc=.001, scale=.02),
                    stats.johnsonsu(0., 2., loc=.001, scale=.02),
                    crystal_ball(1., 2.5, loc=.001, scale=.02),
                    crystal_ball(2., 6., loc=.001, scale=.02)]
        for f in families:
            for confidence in [.05, .95, .99, .999]:
                probability = 1-confidence
                expected = -integrate.quad(lambda p: f.ppf(p), 0, probability,
                                          epsabs=1e-10, limit=200)[0]/probability
                result = portfolio_risk(f, confidence)
                self.assertAlmostEqual(result['var'], -f.ppf(probability), places=10)
                self.assertAlmostEqual(result['es'], expected, places=7)
        f = stats.johnsonsu(.5, 1.)
        p = np.array([1e-12, .5, 1-1e-12])
        np.testing.assert_allclose(f.cdf(f.ppf(p)), p, rtol=1e-9, atol=1e-14)

    def test_fits_aliases_fixed_location_and_failures(self):
        samples = {'johnson-su': stats.johnsonsu.rvs(.7, 1.2, size=500, random_state=4)*.01,
                   'johnson-su-symmetric': stats.johnsonsu.rvs(0., 1.2, size=500, random_state=4)*.01,
                   'crystal-ball': stats.crystalball.rvs(1., 4., size=500, random_state=5)*.01}
        for name, x in samples.items():
            row = fit_one(x, name, max_iterations=2500)
            self.assertEqual(row['status'], 'ok')
            k = 3 if name.endswith('symmetric') else 4
            self.assertEqual(row['k'], k)
            self.assertAlmostEqual(row['aic'], 2*k-2*row['loglik'])
            f = fitted_distribution(json.loads(json.dumps(row)))
            self.assertAlmostEqual(f.logpdf(x).sum(), row['loglik'], places=7)
            if name.endswith('symmetric'):
                self.assertEqual(row['a'], 0.)
            fixed = fit_one(x, name, location=0., max_iterations=2500)
            self.assertTrue(fixed['converged'])
            self.assertEqual(fixed['k'], k-1)
            self.assertAlmostEqual(fixed['loc'], 0.)
            limited = fit_one(x, name, max_iterations=1)
            self.assertEqual(limited['status'], 'not_converged')

    def test_copula_marginals(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({'A': stats.johnsonsu.rvs(.5, 1., size=150, random_state=rng)*.01,
                              'B': stats.crystalball.rvs(1., 4., size=150, random_state=rng)*.01},
                              index=pd.bdate_range('2020-01-01', periods=150))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root/'returns.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                code = copula_main([str(root/'returns.csv'), '--input-type', 'returns',
                    '--marginal', 'A=johnson-su', '--marginal', 'B=crystal-ball',
                    '--copulas', 'gaussian', '--max-iterations', '2500', '--output', str(root/'fits.json')])
            self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
