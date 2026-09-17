"""Test Azzalini skew-t copula densities, quantiles, sampling, and estimation.
Cover symmetry, permutations, saved models, portfolio risk, and CLI integration.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
from return_distributions.copulas import copula_logpdf, sample_copula, fit_copula, CopulaJoint
from return_distributions.skew_t_copula import marginal_cdf, marginal_ppf, latent_model
from return_distributions.skew_t import azzalini_skew_t
from return_distributions.fitting import fit_one


class SkewTCopulaTests(unittest.TestCase):
    def fixture(self):
        return dict(model='azzalini-skew-t', correlation=[[1., .45], [.45, 1.]],
                    df=5., alpha=[-2., 1.], status='ok', parameters=4,
                    cdf_audit_max_error=0., loglik=5., aic=0., bic=10.)

    def test_cdf_and_quantiles(self):
        x = np.array([-100., -5., -1., 0., 1., 5., 100.])
        u = np.array([1e-10, .001, .1, .5, .9, .999, 1-1e-10])
        for df in [.25, 1., 5., 200.]:
            for a in [-12., .001, 2., 12.]:
                np.testing.assert_allclose(marginal_cdf(x, df, a), azzalini_skew_t.cdf(x, df, a), atol=2e-11)
                z = marginal_ppf(u, df, a)
                np.testing.assert_allclose(azzalini_skew_t.cdf(z, df, a), u, atol=2e-10)

    def test_zero_shape_and_permutation(self):
        u = np.random.default_rng(2).uniform(.001, .999, (120, 2))
        f = self.fixture()
        t = copula_logpdf(u, 'student-t', f['correlation'], f['df'])
        ac = copula_logpdf(u, f['model'], f['correlation'], f['df'], [0., 0.])
        np.testing.assert_allclose(t, ac, atol=1e-12)
        a = copula_logpdf(u, f['model'], f['correlation'], f['df'], f['alpha'])
        b = copula_logpdf(u[:, ::-1], f['model'], np.array(f['correlation'])[::-1, ::-1], f['df'], f['alpha'][::-1])
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_uniforms_and_density(self):
        f = self.fixture()
        u = sample_copula(f, 12000, 8)
        self.assertEqual(u.shape, (12000, 2))
        for j in range(2): self.assertLess(stats.kstest(u[:, j], 'uniform').statistic, .018)
        model, shapes = latent_model(f['correlation'], f['df'], f['alpha'])
        # Marginal shapes are generally not the joint alpha coefficients.
        self.assertFalse(np.allclose(shapes, f['alpha']))
        x = model.rvs(30, 7)
        scores = np.column_stack([marginal_cdf(x[:, j], f['df'], a) for j, a in enumerate(shapes)])
        expected = model.logpdf(x)-sum(azzalini_skew_t.logpdf(x[:, j], f['df'], a) for j, a in enumerate(shapes))
        np.testing.assert_allclose(copula_logpdf(scores, f['model'], f['correlation'], f['df'], f['alpha']), expected, atol=1e-8)

    def test_fit(self):
        u = sample_copula(self.fixture(), 120, 71)
        # Exercise the optimizer and likelihood bookkeeping without requiring
        # a slow, sample-dependent convergence path in the unit suite.
        f = fit_copula(u, 'azzalini-skew-t', 8)
        self.assertEqual(f['parameters'], 4)
        self.assertIn(f['status'], ['ok', 'not_converged', 'boundary'])
        self.assertGreaterEqual(f['loglik'], f['nested_t_loglik']-1e-7)
        ll = copula_logpdf(u, f['model'], f['correlation'], f['df'], f['alpha']).sum()
        self.assertAlmostEqual(f['loglik'], ll, places=7)
        self.assertLess(f['cdf_audit_max_error'], 2e-7)
        self.assertEqual(len(f['attempts']), 5)
        self.assertGreater(np.linalg.eigvalsh(np.asarray(f['correlation'])-np.outer(f['delta'], f['delta'])).min(), 0.)

    def test_saved_joint_and_portfolio(self):
        from return_distributions.copula_portfolio import simulate_portfolio
        rng = np.random.default_rng(8)
        margins = [fit_one(rng.normal(0, .01, 100), 'normal') for _ in range(2)]
        record = dict(copula=self.fixture(), marginals=margins, marginal_mode='fitted',
                      symbols=['A', 'B'], return_type='simple')
        joint = CopulaJoint(json.loads(json.dumps(record)))
        self.assertTrue(np.isfinite(joint.logpdf(joint.rvs(10, 7))).all())
        metrics, _ = simulate_portfolio(record, [.6, .4], [], [.95], 2000, 7, 4, False)
        self.assertTrue(np.isfinite(metrics['var_0.95']))
        self.assertGreater(metrics['es_0.95'], metrics['var_0.95'])

    def test_cli_wiring_and_refinement_guard(self):
        from return_distributions.copula_cli import main
        from return_distributions.joint_cli import main as joint_main
        from return_distributions.copula_refinement import refine_joint
        with self.assertRaises(ValueError): refine_joint(None, {'copula': self.fixture(), 'marginal_mode': 'fitted'})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['missing.csv', '--copulas', 'azzalini-skew-t', '--joint-refine'])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'r.csv'
            pd.DataFrame(np.random.default_rng(1).normal(0, .01, (40, 2)), columns=['A', 'B'],
                         index=pd.bdate_range('2020-01-01', periods=40)).to_csv(source)
            common = [str(source), '--input-type', 'returns', '--copulas', 'azzalini-skew-t',
                      '--marginal-models', 'normal', '--no-save']
            for runner, target, extra in [(main, 'return_distributions.copula_cli.fit_copula', []),
                    (joint_main, 'return_distributions.joint_copula.run_fit', ['--models', 'normal'])]:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream), patch(target, return_value=self.fixture()):
                    self.assertEqual(runner(common+extra), 0)
                self.assertIn('Joint copula shape alpha:', stream.getvalue())


if __name__ == '__main__': unittest.main()
