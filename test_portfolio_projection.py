import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from return_distributions import project_distribution, joint_distribution
from return_distributions.multivariate import JOINT_MODELS
from return_distributions.simulation import preset
from return_distributions.projection import ProjectedPower
from return_distributions.portfolio_cli import main, read_weights


class ProjectionTests(unittest.TestCase):
    def test_all_moments_and_simulation(self):
        w = np.array([.8, -.3])
        for family in JOINT_MODELS:
            fit = preset(family)
            projected = project_distribution(fit, w)
            self.assertAlmostEqual(projected.mean(), w@fit['mean'], places=10)
            self.assertAlmostEqual(projected.var(), w@np.array(fit['covariance'])@w, places=10)
            x = joint_distribution(fit).rvs(40000, random_state=np.random.default_rng(94))@w
            # Monte Carlo validation of the actual joint projection.
            for q in [.05, .5, .95]:
                value = float(projected.ppf(q))
                self.assertLess(abs(np.mean(x <= value)-q), .012, family)
                self.assertAlmostEqual(float(projected.cdf(value)), q, places=6)

    def test_power_normal_limit(self):
        for d in [2, 3, 5]:
            projected = ProjectedPower(.2, .7, d, 2.)
            ref = stats.norm(.2, .7/np.sqrt(2))
            grid = np.linspace(-1, 1, 7)
            np.testing.assert_allclose(projected.pdf(grid), ref.pdf(grid), atol=1e-8)
            np.testing.assert_allclose(projected.cdf(grid), ref.cdf(grid), atol=1e-8)

    def test_validation_and_zero(self):
        fit = preset('normal')
        with self.assertRaises(ValueError): project_distribution(fit, [.5])
        with self.assertRaises(ValueError): project_distribution(fit, [np.nan, 1])
        fit['return_type'] = 'log'
        with self.assertRaises(ValueError): project_distribution(fit, [.5, .5])
        self.assertTrue(np.isfinite(project_distribution(fit, [.5, .5], allow_log=True).var()))
        zero = project_distribution(fit, [0, 0], allow_log=True)
        self.assertEqual(zero.std(), 0)
        np.testing.assert_array_equal(zero.ppf([.01, .99]), [0, 0])
        with self.assertRaises(ValueError): read_weights(['SPY=.5', 'spy=.5'])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'fits.json', Path(tmp)/'projection.csv'
            fit = preset('normal')
            fit.update(symbols=['SPY', 'TLT'], status='ok', return_type='simple')
            source.write_text(json.dumps([fit]))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(source), '--weights', 'TLT=.4', 'SPY=.6', '--risk-levels', '.95', '.99', '--output', str(output)])
            self.assertEqual(code, 0)
            row = pd.read_csv(output).iloc[0]
            self.assertAlmostEqual(row.volatility, project_distribution(fit, [.6, .4]).std())
            self.assertGreater(row['es_0.95'], row['var_0.95'])
            self.assertEqual(row['es_status_0.99'], 'finite')


if __name__ == '__main__':
    unittest.main()
