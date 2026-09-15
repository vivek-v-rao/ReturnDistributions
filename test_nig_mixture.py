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
from return_distributions.distribution_distances import compare_joint


def record(d=2):
    psi = 2.5
    return dict(model='nig-skewed', components=2, df=None, psi=psi, chi=1., **{'lambda': -.5},
                mixture_weights=[.35, .65], component_fits=[dict(model='nig-skewed', location=[mu]*d,
                    scatter=(np.eye(d)*scatter).tolist(), gamma=np.linspace(gamma, gamma/2, d).tolist(),
                    psi=psi, chi=1., df=None, **{'lambda': -.5}) for mu, scatter, gamma in
                    [(-.01, .0001, -.006), (.02, .0004, .01)]])


class NIGMixtureTests(unittest.TestCase):
    def test_density_moments_and_risk(self):
        fit = record()
        dist = joint_distribution(fit)
        draws = dist.rvs(180000, 15)
        np.testing.assert_allclose(draws.mean(0), dist.mean(), atol=.0002)
        np.testing.assert_allclose(np.cov(draws.T), dist.cov(), rtol=.025, atol=.000005)
        w = np.array([.6, -.4])
        projection = project_distribution(fit, w)
        self.assertAlmostEqual(projection.mean(), w@dist.mean())
        self.assertAlmostEqual(projection.var(), w@dist.cov()@w)
        risk = portfolio_risk(projection, .99)
        q = -risk['var']
        self.assertAlmostEqual(projection.cdf(q), .01, places=8)
        partial = integrate.quad(lambda x: x*projection.pdf(x), -np.inf, q, epsabs=1e-11)[0]
        self.assertAlmostEqual(risk['es'], -partial/.01, places=7)
        raw = draws@w
        self.assertAlmostEqual(-raw[raw <= np.quantile(raw, .01)].mean(), risk['es'], delta=.001)
        scaled = project_distribution(dict(fit, vol_standardization='ewma', next_volatility=[.5, 2.]), w)
        equivalent = project_distribution(fit, w*[.5, 2.])
        np.testing.assert_allclose(scaled.ppf([.01, .5]), equivalent.ppf([.01, .5]))
        frozen1 = joint_distribution(record(1))
        self.assertAlmostEqual(integrate.quad(lambda x: frozen1.pdf([x]), -np.inf, np.inf)[0], 1., places=7)

    def test_fit_shared_shape_counts_and_bounds(self):
        x = joint_distribution(record()).rvs(200, 33)
        for location, k in [(None, 16), (0., 12)]:
            fit = fit_finite_mixture(x, 'nig-skewed', starts=2, max_iterations=2, location=location)
            self.assertEqual(fit['parameters'], k)
            self.assertEqual(fit['status'], 'not_converged')
            self.assertTrue(all(r['psi'] == fit['psi'] and r['chi'] == 1. and r['lambda'] == -.5 for r in fit['component_fits']))
            self.assertAlmostEqual(joint_distribution(json.loads(json.dumps(fit))).logpdf(x).sum(), fit['loglik'], places=6)
            for r in fit['component_fits']:
                scale = np.asarray(fit['mixture_scaling'])
                self.assertGreaterEqual(np.linalg.eigvalsh(np.array(r['scatter'])/np.outer(scale, scale)).min(), fit['mixture_eigen_floor']*.9999)
                if location is not None: np.testing.assert_allclose(r['location'], [0., 0.], atol=1e-14)
        bad = record()
        bad['component_fits'][1]['psi'] = 3.
        with self.assertRaises(ValueError): joint_distribution(bad)

    def test_js_and_asset_permutation(self):
        fit = dict(record(), status='ok', symbols=['A', 'B'], fit_label='original')
        reversed_fit = json.loads(json.dumps(fit))
        reversed_fit.update(symbols=['B', 'A'], fit_label='reversed')
        for r in reversed_fit['component_fits']:
            r['location'] = r['location'][::-1]
            r['gamma'] = r['gamma'][::-1]
            r['scatter'] = np.array(r['scatter'])[::-1, ::-1].tolist()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = compare_joint([fit, reversed_fit], simulations=2000)
        self.assertTrue(all(r['status'] == 'ok' for r in rows))
        self.assertLess(max(r['value'] for r in rows), 1e-7)

    def test_cli_selection_alias_and_saved_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            x = joint_distribution(record()).rvs(150, 22)
            pd.DataFrame(x, columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=len(x))).to_csv(path/'r.csv')
            # A short optimization intentionally exercises visible non-convergence.
            args = [str(path/'r.csv'), '--input-type', 'returns', '--models', 'normal', '--mixture-models', 'nig',
                    '--max-components', '2', '--max-iterations', '2', '--mixture-starts', '2', '--output', str(path/'fits.json')]
            with contextlib.redirect_stdout(io.StringIO()), patch('return_distributions.joint_cli.fit_joint', wraps=fit_joint) as fitter:
                self.assertEqual(main(args), 1)
                self.assertEqual(fitter.call_count, 2)
            fits = json.loads((path/'fits.json').read_text())
            self.assertEqual([(f['model'], f['components']) for f in fits], [('normal', 1), ('nig-skewed', 1), ('nig-skewed', 2)])
            self.assertEqual(set(pd.read_csv(path/'fits_mixtures.csv').model), {'nig-skewed'})
            self.assertEqual(fits[-1]['status'], 'not_converged')


if __name__ == '__main__':
    unittest.main()
