"""Test Fernandez-Steel skew-normal specifications and the normal limit.
Cover parameter counts, fitted-law reconstruction, and copula marginals.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from return_distributions.fitting import fit_one, fitted_distribution, specification
from return_distributions.skew_ged import skew_ged
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copula_cli import main as copula_main


class FSSkewNormalTests(unittest.TestCase):
    def test_specification_and_normal_limit(self):
        dist, names, fixed = specification('fs-skew-normal')
        self.assertIs(dist, skew_ged)
        self.assertEqual(fixed, {'power': 2.})
        f = dist(2., 0., loc=.001, scale=.02)
        normal = stats.norm(loc=.001, scale=.02/np.sqrt(2))
        x = np.linspace(-.05, .05, 31)
        for method in ['pdf', 'cdf', 'sf']:
            np.testing.assert_allclose(getattr(f, method)(x), getattr(normal, method)(x), atol=1e-12)
        for key in ['var', 'es']:
            self.assertAlmostEqual(portfolio_risk(f)[key], portfolio_risk(normal)[key], places=10)

    def test_fit_parameter_count_and_roundtrip(self):
        x = skew_ged.rvs(2., -.5, loc=.001, scale=.02, size=600, random_state=3)
        for location, count in [(None, 3), (.001, 2)]:
            fit = fit_one(x, 'fs-skew-normal', location=location)
            self.assertEqual(fit['status'], 'ok')
            self.assertEqual(fit['power'], 2.)
            self.assertEqual(fit['k'], count)
            self.assertLess(fit['skewness'], 0.)
            self.assertAlmostEqual(fit['aic'], 2*count-2*fit['loglik'])
            f = fitted_distribution(json.loads(json.dumps(fit)))
            self.assertAlmostEqual(f.logpdf(x).sum(), fit['loglik'], places=8)
            if location is not None:
                self.assertAlmostEqual(fit['loc'], location)
        self.assertEqual(fit_one(x, 'fs-skew-normal', max_iterations=1)['status'], 'not_converged')

    def test_copula_marginal(self):
        sample = skew_ged.rvs(2., -.3, size=(120, 2), random_state=10)*.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(sample, columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=120)).to_csv(root/'returns.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                code = copula_main([str(root/'returns.csv'), '--input-type', 'returns',
                    '--marginal-models', 'fs-skew-normal', '--copulas', 'gaussian',
                    '--output', str(root/'fits.json')])
            self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
