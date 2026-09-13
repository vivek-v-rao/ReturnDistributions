import unittest
import numpy as np
from scipy import stats
from return_distributions import DEFAULT_MODELS, fit_one, fit_many, fitted_distribution


class FitTests(unittest.TestCase):
    def setUp(self):
        self.x = stats.t.rvs(6, loc=.001, scale=.02, size=250, random_state=52)

    def test_normal_units_location(self):
        r = fit_one(self.x, 'normal')
        self.assertAlmostEqual(r['loc'], self.x.mean())
        self.assertAlmostEqual(r['scale'], self.x.std())
        self.assertEqual(r['k'], 2)
        self.assertEqual(fit_one(self.x, 'normal', location=0)['k'], 1)
        self.assertAlmostEqual(r['loglik'], fitted_distribution(r).logpdf(self.x).sum())
        self.assertTrue(np.isnan(r['ks_p']))

    def test_variants_and_parameter_counts(self):
        fits = fit_many(self.x)
        self.assertEqual(len(fits), 8)
        for _, r in fits.iterrows():
            self.assertEqual(r.status, 'ok', str(r))
            expected = 2 if r['name'] in ['normal', 'laplace'] else (4 if r['name'].endswith('skewed') else 3)
            self.assertEqual(r.k, expected)
            self.assertAlmostEqual(r.aic, 2*r.k-2*r.loglik)
            if r['name'].endswith('symmetric'):
                self.assertEqual(r.b, 0)
            if r['name'].startswith('hyperbolic'):
                self.assertEqual(r.p, 1)
            self.assertAlmostEqual(fitted_distribution(r).logpdf(self.x).sum(), r.loglik)

    def test_failures_and_convergence(self):
        for data in [np.ones(10), [1, np.nan]*10, [[1, 2]]*10]:
            with self.assertRaises(ValueError):
                fit_one(data, 'normal')
        r = fit_one(self.x, 'nig-skewed', max_iterations=1)
        self.assertFalse(r['converged'])
        r = fit_many(self.x, ['not-a-model', 'normal'])
        self.assertEqual(r.iloc[0]['name'], 'normal')
        self.assertEqual(r.iloc[-1].status, 'failed')

    def test_nig_hyperbolic_identity(self):
        x = np.linspace(-3, 3, 40)
        np.testing.assert_allclose(stats.norminvgauss.logpdf(x, 2, .5),
                                   stats.genhyperbolic.logpdf(x, -.5, 2, .5), atol=1e-12)

    def test_scale_equivariance(self):
        a = fit_one(self.x, 'student-t')
        b = fit_one(100*self.x, 'student-t')
        self.assertAlmostEqual(b['loc'], 100*a['loc'], places=6)
        self.assertAlmostEqual(b['loglik'], a['loglik']-len(self.x)*np.log(100), places=6)

    def test_cli_and_missing_prices(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        import pandas as pd
        from return_distributions.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'prices.csv', Path(tmp)/'fits.csv'
            prices = pd.DataFrame({'A': 100*np.cumprod(1+self.x[:30])},
                                  index=pd.bdate_range('2020-01-01', periods=30))
            prices.iloc[10, 0] = np.nan
            prices.to_csv(source)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(source), '--models', 'normal', '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(pd.read_csv(output).iloc[0]['n'], 27)


if __name__ == '__main__':
    unittest.main()
