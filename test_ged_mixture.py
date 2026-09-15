import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import integrate

from return_distributions.joint_finite_mixture import fit_finite_mixture
from return_distributions.multivariate import joint_distribution, fit_joint
from return_distributions.projection import project_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.joint_cli import main


def record(power=1.5, d=2):
    return dict(model='ged', components=2, power=power, df=None, mixture_weights=[.4, .6],
                component_fits=[dict(model='ged', location=[mu]*d, scatter=(np.eye(d)*s).tolist(),
                                     df=None, power=power) for mu, s in [(-.01, .0001), (.02, .0004)]])


class GEDMixtureTests(unittest.TestCase):
    def test_gaussian_limit_density_and_risk(self):
        ged = record(2.)
        normal = dict(ged, model='normal', component_fits=[dict(r, model='normal', scatter=(np.array(r['scatter'])/2).tolist())
                                                        for r in ged['component_fits']])
        x = np.random.default_rng(4).normal(size=(40, 2))*.02
        a, b = joint_distribution(ged), joint_distribution(normal)
        np.testing.assert_allclose(a.logpdf(x), b.logpdf(x), atol=1e-12)
        np.testing.assert_allclose(a.cov(), b.cov())
        ga, gb = project_distribution(ged, [.6, -.4]), project_distribution(normal, [.6, -.4])
        np.testing.assert_allclose(ga.ppf([.01, .5, .99]), gb.ppf([.01, .5, .99]), atol=1e-10)
        self.assertAlmostEqual(portfolio_risk(ga, .99)['es'], portfolio_risk(gb, .99)['es'], places=8)

    def test_moments_and_non_normal_projection(self):
        fit = record()
        dist = joint_distribution(fit)
        draws = dist.rvs(150000, 17)
        np.testing.assert_allclose(draws.mean(0), dist.mean(), atol=.0002)
        np.testing.assert_allclose(np.cov(draws.T), dist.cov(), rtol=.025, atol=.00001)
        w = np.array([.6, -.4])
        projection = project_distribution(fit, w)
        scaled = project_distribution(dict(fit, vol_standardization='ewma', next_volatility=[.5, 2.]), w)
        equivalent = project_distribution(fit, w*[.5, 2.])
        np.testing.assert_allclose(scaled.ppf([.05, .5]), equivalent.ppf([.05, .5]))
        self.assertAlmostEqual(projection.mean(), w@dist.mean())
        self.assertAlmostEqual(projection.var(), w@dist.cov()@w)
        risk = portfolio_risk(projection, .99)
        q = -risk['var']
        self.assertAlmostEqual(projection.cdf(q), .01, places=8)
        self.assertAlmostEqual(np.mean(draws@w <= q), .01, delta=.001)
        empirical = draws@w
        self.assertAlmostEqual(-empirical[empirical <= np.quantile(empirical, .01)].mean(), risk['es'], delta=.001)
        univariate = project_distribution(record(d=1), [1.])
        q = float(univariate.ppf(.025))
        partial = integrate.quad(lambda x: x*univariate.pdf(x), -np.inf, q, epsabs=1e-11)[0]
        self.assertAlmostEqual(portfolio_risk(univariate, .975)['es'], -partial/.025, places=7)

    def test_fit_counts_likelihood_and_fixed_location(self):
        rng = np.random.default_rng(22)
        x = np.r_[rng.normal(-2, .5, (130, 2)), rng.normal(2, .7, (170, 2))]
        baseline = fit_joint(x, 'ged')
        fit = fit_finite_mixture(x, 'ged', baseline=baseline, starts=3, max_iterations=600)
        self.assertEqual(fit['parameters'], 12)
        self.assertEqual(fit['status'], 'ok')
        self.assertGreater(fit['loglik'], baseline['loglik']+50)
        self.assertTrue(all(r['power'] == fit['power'] for r in fit['component_fits']))
        self.assertAlmostEqual(joint_distribution(json.loads(json.dumps(fit))).logpdf(x).sum(), fit['loglik'], places=6)
        fixed = fit_finite_mixture(x, 'ged', location=0., starts=2, max_iterations=1)
        self.assertEqual(fixed['parameters'], 8)
        self.assertEqual(fixed['status'], 'not_converged')
        for r in fixed['component_fits']: np.testing.assert_allclose(r['location'], [0., 0.], atol=1e-14)

    def test_cli_all_selects_only_supported_requested_families(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            x = np.random.default_rng(42).normal(0, .01, (100, 2))
            pd.DataFrame(x, columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=len(x))).to_csv(path/'r.csv')
            args = [str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal', 'laplace',
                    '--mixture-models', 'all', '--max-components', '2', '--mixture-starts', '2',
                    '--max-iterations', '5', '--output', str(path/'fit.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            fits = json.loads((path/'fit.json').read_text())
            self.assertEqual([(f['model'], f['components']) for f in fits],
                             [('normal', 1), ('normal', 2), ('laplace', 1)])
            for options in (['--mixture-models', 'all'],
                            ['--mixture-models', 'all', 'normal', '--max-components', '2'],
                            ['--models', 'laplace', '--mixture-models', 'all', '--max-components', '2']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(path/'r.csv'), *options])

    def test_cli_selector_adds_baseline_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rng = np.random.default_rng(22)
            x = np.r_[rng.normal(-.02, .005, (130, 2)), rng.normal(.02, .007, (170, 2))]
            pd.DataFrame(x, columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=len(x))).to_csv(path/'r.csv')
            args = [str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal', '--mixture-models', 'ged',
                    '--max-components', '2', '--mixture-starts', '3', '--weights', 'A=.6', 'B=.4',
                    '--risk-levels', '.95', '--js-distance', '--simulations', '2000', '--output', str(path/'fit.json')]
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch('return_distributions.joint_cli.fit_joint', wraps=fit_joint) as fitter:
                self.assertEqual(main(args), 0, out.getvalue())
                self.assertEqual(fitter.call_count, 2)
            fits = json.loads((path/'fit.json').read_text())
            self.assertEqual([(f['model'], f['components']) for f in fits], [('normal', 1), ('ged', 1), ('ged', 2)])
            self.assertEqual(set(pd.read_csv(path/'fit_mixtures.csv').model), {'ged'})
            self.assertIn('Added single-component baselines: ged', out.getvalue())
            for options in (['--mixture-models', 'laplace', '--max-components', '2'], ['--mixture-models', 'ged']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(path/'r.csv'), *options])


if __name__ == '__main__':
    unittest.main()
