"""Test finite joint mixtures, density reconstruction, moments, and constraints.
Cover shared Student-t degrees of freedom, fit ranks, and CLI comparisons.
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

from return_distributions.joint_finite_mixture import fit_finite_mixture, JointFiniteMixture, ProjectedFiniteMixture, _weights
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.projection import project_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.joint_cli import main, rank_mixture_comparison, format_fit_comparison


class FiniteMixtureTests(unittest.TestCase):
    def test_mixture_ranks_include_nonconverged_and_ties(self):
        frame = pd.DataFrame(dict(model=['normal', 'normal', 'student-t', 'student-t', 'normal', 'normal', 'student-t', 'normal'],
                                  window=[None]*7+[126],
                                  aic=[10., 5., 5., 2., np.nan, np.inf, -np.inf, 999.],
                                  bic=[4., 9., 3., 8., 6., np.inf, np.nan, 1.],
                                  status=['ok', 'not_converged', 'ok', 'boundary', 'failed', 'failed', 'failed', 'ok']))
        ranked = rank_mixture_comparison(frame)
        self.assertEqual(ranked.aic_rank_family.iloc[:4].tolist(), [2, 1, 2, 1])
        self.assertEqual(ranked.aic_rank_all.iloc[:4].tolist(), [4, 2, 2, 1])
        self.assertTrue(ranked.aic_rank_all.iloc[4:7].isna().all())
        self.assertEqual(ranked.aic_rank_all.iloc[7], 1)
        self.assertEqual(ranked.bic_rank_family.iloc[:5].tolist(), [1, 3, 1, 2, 2])
        self.assertEqual(ranked.bic_rank_all.iloc[:5].tolist(), [2, 5, 1, 4, 3])
        self.assertNotIn('aic_rank_family', frame)
        self.assertIn('not_converged', format_fit_comparison(ranked))

    def test_density_moments_and_projection(self):
        for df in (None, 5.):
            model = 'normal' if df is None else 'student-t'
            record = dict(model=model, components=2, mixture_weights=[.3, .7], df=df,
                          component_fits=[dict(model=model, location=m, scatter=s, df=df) for m, s in
                                          [([-.01, .02], [[.0001, .00002], [.00002, .0004]]),
                                           ([.005, -.01], [[.0003, 0.], [0., .0002]])]])
            frozen = joint_distribution(record)
            points = np.array([[0., 0.], [.01, -.01]])
            reference = .3*frozen.distributions[0].pdf(points)+.7*frozen.distributions[1].pdf(points)
            np.testing.assert_allclose(frozen.pdf(points), reference)
            self.assertEqual(frozen.logpdf(points[:1]).shape, (1,))
            self.assertAlmostEqual(frozen.pdf(points[0]), reference[0])
            draws = frozen.rvs(150000, 43)
            np.testing.assert_allclose(draws.mean(0), frozen.mean(), atol=.0002)
            np.testing.assert_allclose(np.cov(draws.T), frozen.cov(), rtol=.025, atol=.00001)
            w = np.array([.6, -.4])
            projection = project_distribution(record, w)
            self.assertAlmostEqual(projection.mean(), w@frozen.mean())
            self.assertAlmostEqual(projection.var(), w@frozen.cov()@w)
            risk = portfolio_risk(projection, .99)
            q = float(projection.ppf(.01))
            partial = integrate.quad(lambda x: x*projection.pdf(x), -np.inf, q, epsabs=1e-11)[0]
            self.assertAlmostEqual(risk['es'], -partial/.01, places=7)
            self.assertAlmostEqual(projection.cdf(q), .01)
            normalized = dict(record, vol_standardization='ewma', next_volatility=[.5, 2.])
            a = project_distribution(normalized, w)
            b = project_distribution(record, w*[.5, 2.])
            np.testing.assert_allclose(a.ppf([.01, .5, .99]), b.ppf([.01, .5, .99]))
        t = ProjectedFiniteMixture([.5, .5], [0., 1.], [1., 2.], .8)
        self.assertTrue(np.isinf(portfolio_risk(t, .99)['es']))

    def test_fit_recovery_counts_and_constraints(self):
        rng = np.random.default_rng(5)
        x = np.r_[rng.normal(-2, .4, (250, 2)), rng.normal(2, .8, (350, 2))]
        baseline = fit_joint(x, 'normal')
        fit = fit_finite_mixture(x, 'normal', baseline=baseline, max_iterations=500)
        self.assertEqual(fit['status'], 'ok')
        self.assertEqual(fit['parameters'], 11)
        self.assertGreater(fit['loglik'], baseline['loglik']+100)
        self.assertAlmostEqual(joint_distribution(json.loads(json.dumps(fit))).logpdf(x).sum(), fit['loglik'])
        np.testing.assert_allclose(sorted(fit['mixture_weights']), [250/600, 350/600], atol=.03)
        fixed = fit_finite_mixture(x, 'normal', location=0., starts=2, max_iterations=300)
        self.assertEqual(fixed['parameters'], 7)
        for r in fixed['component_fits']: np.testing.assert_allclose(r['location'], [0., 0.], atol=1e-14)
        constrained = fit_finite_mixture(x, 'normal', starts=2, eigen_floor=3., max_iterations=100)
        self.assertEqual(constrained['status'], 'boundary')
        with self.assertRaises(ValueError): project_distribution(constrained, [.5, .5])
        np.testing.assert_allclose(_weights(np.array([1., 99.]), .1), [.1, .9])

    def test_shared_student_df(self):
        rng = np.random.default_rng(21)
        x = np.r_[rng.standard_t(5, (300, 1))*.4-2, rng.standard_t(5, (400, 1))*.6+2]
        fit = fit_finite_mixture(x, 'student-t', max_iterations=1000, starts=3)
        self.assertEqual(fit['status'], 'ok')
        self.assertEqual(fit['parameters'], 6)
        self.assertTrue(all(r['df'] == fit['df'] for r in fit['component_fits']))
        self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'])

    def test_cli_separate_table_and_same_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rng = np.random.default_rng(5)
            x = np.r_[rng.normal(-.02, .004, (140, 2)), rng.normal(.02, .008, (160, 2))]
            pd.DataFrame(x, columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=len(x))).to_csv(path/'r.csv')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main([str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal',
                               '--max-components', '2', '--standardize-vol', 'none', 'ewma',
                               '--weights', 'A=.6', 'B=.4', '--js-distance', '--simulations', '2000',
                               '--output', str(path/'fits.json')])
            self.assertEqual(status, 0)
            fits = json.loads((path/'fits.json').read_text())
            self.assertEqual(len(fits), 4)
            self.assertEqual(len({(f['first_date'], f['last_date'], f['observations']) for f in fits}), 1)
            self.assertEqual([f['components'] for f in fits], [1, 2, 1, 2])
            saved = pd.read_csv(path/'fits_mixtures.csv')
            self.assertEqual(len(saved), 4)
            self.assertTrue({'aic_rank_family', 'aic_rank_all', 'bic_rank_family', 'bic_rank_all'}.issubset(saved.columns))
            np.testing.assert_array_equal(saved.aic_rank_all, saved.aic.rank(method='min'))
            np.testing.assert_array_equal(saved.bic_rank_all, saved.bic.rank(method='min'))
            self.assertIn('Finite mixture comparison', out.getvalue())
            detailed = out.getvalue().split('Joint fit comparison:')[1].split('Finite mixture comparison')[0]
            self.assertNotIn('components', detailed)
            self.assertIn('normal [2 components]', out.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(path/'r.csv'), '--models', 'laplace', '--max-components', '2'])


if __name__ == '__main__':
    unittest.main()
