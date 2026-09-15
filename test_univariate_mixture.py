import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from return_distributions.cli import main
from return_distributions.fitting import fitted_distribution
from return_distributions.joint_finite_mixture import JointFiniteMixture
from return_distributions.univariate_mixture import baseline_record, UnivariateMixture
from return_distributions.distribution_distances import univariate_distance


class UnivariateMixtureTests(unittest.TestCase):
    def test_baseline_density_conversions(self):
        grid = np.linspace(-.05, .05, 51)
        for model, shapes, law in [
            ('normal', {}, stats.norm(loc=.001, scale=.01)),
            ('student-t', {'df': 5.}, stats.t(5, loc=.001, scale=.01)),
            ('ged', {'beta': 1.2}, stats.gennorm(1.2, loc=.001, scale=.01)),
            ('nig-skewed', {'a': 2., 'b': -.3}, stats.norminvgauss(2., -.3, loc=.001, scale=.01))]:
            base = baseline_record(dict(status='ok', loc=.001, scale=.01, **shapes), model)
            fit = dict(base, mixture_weights=[1.], component_fits=[base])
            frozen = UnivariateMixture(fit)
            np.testing.assert_allclose(frozen.logpdf(grid), law.logpdf(grid), atol=1e-10)
            np.testing.assert_allclose(frozen.cdf(grid), law.cdf(grid), atol=1e-8)
            np.testing.assert_allclose(frozen.ppf([.01, .5, .99]), law.ppf([.01, .5, .99]), atol=1e-8)
            self.assertEqual(frozen.rvs(10, random_state=1).shape, (10,))
            self.assertAlmostEqual(frozen.mean(), law.mean())
            self.assertAlmostEqual(frozen.var(), law.var())

    def test_cli_and_saved_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rng = np.random.default_rng(18)
            x = np.r_[rng.normal(-.03, .005, 80), rng.normal(.02, .008, 120)]
            pd.DataFrame({'A': x}, index=pd.date_range('2020-01-01', periods=len(x))).to_csv(path/'r.csv')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal', 'laplace',
                             '--mixture-models', 'all', '--max-components', '2', '--mixture-starts', '2',
                             '--output', str(path/'fits.csv')])
            self.assertEqual(code, 0, out.getvalue())
            table = pd.read_csv(path/'fits.csv')
            self.assertEqual(len(table), 3)
            row = table.loc[table.components.eq(2)].iloc[0]
            law = fitted_distribution(row)
            self.assertAlmostEqual(np.sum(law.logpdf(x)), row.loglik, places=6)
            self.assertAlmostEqual(law.cdf(law.ppf(.01)), .01, places=7)
            self.assertGreater(law.risk(.99)['es'], law.risk(.99)['var'])
            self.assertEqual(row.k, 5)
            self.assertIn('Aggregate univariate fit comparison', out.getvalue())
            self.assertIn('weight', out.getvalue())
            value, _, _ = univariate_distance(law, law, 'ks')
            self.assertAlmostEqual(value, 0.)

    def test_validation_before_data(self):
        for options in [ ['--mixture-models', 'all'],
                         ['--mixture-models', 'all', 'normal', '--max-components', '2'],
                         ['--models', 'laplace', '--mixture-models', 'all', '--max-components', '2'],
                         ['--mixture-models', 'laplace', '--max-components', '2'] ]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['missing.csv', *options])


if __name__ == '__main__':
    unittest.main()
