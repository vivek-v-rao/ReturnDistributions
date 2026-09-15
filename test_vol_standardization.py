import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.vol_standardization import ewma_standardize, annotate_fit
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.projection import project_distribution
from return_distributions.joint_portfolio_mc import simulate_joint_portfolio
from return_distributions.joint_cli import main
from return_distributions.simulation import preset
from return_distributions.joint_risk_report import risk_report
from return_distributions.distribution_distances import compare_joint


class VolStandardizationTests(unittest.TestCase):
    def test_multiple_lambdas_fit_raw_once_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame(np.random.default_rng(19).normal(size=(300, 2))*.01,
                                 columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=300))
            frame.to_csv(path/'returns.csv')
            args = [str(path/'returns.csv'), '--input-type', 'returns', '--models', 'normal', 'fs-skew-normal',
                    '--days', '126', '400', '--standardize-vol', 'none', 'ewma', '--vol-warmup', '20',
                    '--vol-lambda', '.94', '.96', '.94', '--asset-orders', 'all', '--best-asset-order',
                    '--univariate', '--weights', 'A=.6', 'B=.4', '--js-distance', '--simulations', '2000',
                    '--output', str(path/'fit.json')]
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch('return_distributions.joint_cli.ewma_standardize', wraps=ewma_standardize) as filtering:
                status = main(args)
                self.assertEqual(status, 0, out.getvalue())
                self.assertEqual(filtering.call_count, 2)
            fits = json.loads((path/'fit.json').read_text())
            self.assertEqual(len(fits), 18)
            for window in (126, 400):
                group = [f for f in fits if f['window'] == window]
                self.assertEqual(len({(f['first_date'], f['last_date'], f['observations']) for f in group}), 1)
                for decay in (None, .94, .96):
                    subset = [f for f in group if f.get('vol_lambda') == decay]
                    self.assertEqual(len(subset), 3)
                    self.assertEqual(sum(f['selected_for_display'] for f in subset), 2)
                    if decay is not None:
                        z, scales, nxt = ewma_standardize(frame, decay, 20)
                        fit = next(f for f in subset if f['model'] == 'normal')
                        expected = z.dropna().tail(window)
                        np.testing.assert_allclose(fit['next_volatility'], nxt)
                        np.testing.assert_allclose(fit['location'], expected.mean())
                        ll = joint_distribution(fit).logpdf(expected.to_numpy()).sum()-np.log(scales.loc[expected.index]).to_numpy().sum()
                        self.assertAlmostEqual(fit['loglik'], ll)
            risks = pd.read_csv(path/'fit_portfolio_risk.csv')
            self.assertEqual(set(risks.vol_lambda.dropna()), {.94, .96})
            self.assertEqual(len(risks[risks.model == 'empirical']), 6)
            univariate = pd.read_csv(path/'fit_univariate.csv')
            self.assertEqual(set(univariate.vol_lambda.dropna()), {.94, .96})
            distances = pd.read_csv(path/'fit_distances.csv')
            joint = distances[distances.scope == 'joint']
            self.assertEqual(len(joint), 162)
            self.assertTrue(joint.model_p.str.contains('lambda=0.96', regex=False).any())
            for values in (['.94', '1.'], ['.94', 'nan'], ['.94', '-.2']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(path/'returns.csv'), '--standardize-vol', 'ewma', '--vol-lambda', *values])

    def test_return_unit_js_matches_equivalent_laws(self):
        rng = np.random.default_rng(6)
        fit = fit_joint(rng.normal(size=(200, 2)), 'normal')
        fit.update(symbols=['A', 'B'], fit_label='ewma', vol_standardization='ewma', next_volatility=[.01, .02])
        raw = dict(fit, fit_label='raw', vol_standardization='none',
                   location=(np.array(fit['location'])*[.01, .02]).tolist(),
                   scatter=(np.array(fit['scatter'])*np.outer([.01, .02], [.01, .02])).tolist())
        with contextlib.redirect_stdout(io.StringIO()):
            rows = compare_joint([raw, fit], simulations=2000, return_units=True)
        self.assertTrue(all(r['status'] == 'ok' for r in rows))
        self.assertLess(max(r['value'] for r in rows), 1e-7)

    def test_combined_cli_common_dates_orders_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame(np.random.default_rng(12).normal(size=(140, 2))*.01,
                                 columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=140))
            frame.to_csv(path/'returns.csv')
            args = [str(path/'returns.csv'), '--input-type', 'returns', '--models', 'normal', 'fs-skew-normal',
                    '--days', '50', '130', '--standardize-vol', 'none', 'ewma', '--vol-warmup', '20',
                    '--asset-orders', 'all', '--best-asset-order', '--univariate', '--js-distance',
                    '--simulations', '2000', '--weights', 'A=.6', 'B=.4', '--output', str(path/'fit.json')]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(args), 0)
            fits = json.loads((path/'fit.json').read_text())
            self.assertEqual(len(fits), 12)
            z, scales, nxt = ewma_standardize(frame, warmup=20)
            for window in (50, 130):
                group = [f for f in fits if f['window'] == window]
                self.assertEqual(len({(f['first_date'], f['last_date'], f['observations']) for f in group}), 1)
                for mode in ('none', 'ewma'):
                    subset = [f for f in group if f['vol_standardization'] == mode]
                    self.assertEqual(sum(f['selected_for_display'] for f in subset), 2)
                    normal = next(f for f in subset if f['model'] == 'normal')
                    expected = (frame if mode == 'none' else z).loc[z.dropna().index].tail(window)
                    self.assertEqual(normal['observations'], min(window, 120))
                    np.testing.assert_allclose(normal['location'], expected.mean())
                    if mode == 'ewma':
                        ll = joint_distribution(normal).logpdf(expected.to_numpy()).sum()-np.log(scales.loc[expected.index]).to_numpy().sum()
                        self.assertAlmostEqual(normal['loglik'], ll)
            risks = pd.read_csv(path/'fit_portfolio_risk.csv')
            self.assertEqual(set(risks.vol_standardization), {'none', 'ewma'})
            self.assertEqual(len(risks[risks.model == 'empirical']), 4)
            tables = pd.read_csv(path/'fit_univariate.csv')
            self.assertEqual(set(tables.vol_standardization), {'none', 'ewma'})
            distances = pd.read_csv(path/'fit_distances.csv')
            joint = distances[distances.scope == 'joint']
            self.assertEqual(len(joint), 72)
            self.assertTrue(joint.comparison_units.str.startswith('original returns').all())
            self.assertIn('vol_standardization', out.getvalue())

    def test_lag_warmup_floor_and_gaps(self):
        frame = pd.DataFrame({'A': [.1, .2, .3, .4, .5]})
        z, h, nxt = ewma_standardize(frame, .94, 2)
        self.assertTrue(z.iloc[:2].isna().all().all())
        self.assertAlmostEqual(h.A.iloc[2]**2, .025)
        self.assertAlmostEqual(h.A.iloc[3]**2, .94*.025+.06*.09)
        changed = frame.copy()
        changed.iloc[3:] = 100.
        _, hh, _ = ewma_standardize(changed, .94, 2)
        np.testing.assert_array_equal(h.iloc[:4], hh.iloc[:4])
        self.assertAlmostEqual(nxt.A**2, .94*h.A.iloc[-1]**2+.06*.25)
        _, zero, _ = ewma_standardize(frame*0, .94, 2, 1e-6)
        self.assertEqual(zero.A.iloc[-1], 1e-6)
        gap = pd.DataFrame({'A': [.1, .2, .3, np.nan, .1, .2, .3]})
        _, scales, _ = ewma_standardize(gap, .94, 2)
        self.assertEqual(scales.A.notna().tolist(), [False, False, True, False, False, False, True])

    def test_likelihood_and_projection(self):
        rng = np.random.default_rng(4)
        raw = pd.DataFrame(rng.normal(size=(150, 2))*.01, columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=150))
        z, scales, nxt = ewma_standardize(raw, .94, 10)
        sample = z.dropna().tail(100)
        record = fit_joint(sample.to_numpy(), 'normal')
        original = dict(record)
        annotate_fit(record, scales.loc[sample.index], nxt, .94, 10, 1e-8)
        record.update(symbols=['A', 'B'], return_type='simple')
        ll = joint_distribution(record).logpdf(sample.to_numpy()).sum()-np.log(scales.loc[sample.index]).to_numpy().sum()
        self.assertAlmostEqual(record['loglik'], ll)
        self.assertAlmostEqual(record['aic'], 2*record['parameters']-2*ll)
        a = project_distribution(record, [.6, .4])
        b = project_distribution(original, np.array([.6, .4])*nxt.to_numpy())
        np.testing.assert_allclose(a.ppf([.01, .5]), b.ppf([.01, .5]))
        report = risk_report([record], sample, {'A': .6, 'B': .4}, [.95])
        empirical = sample.to_numpy()@(np.array([.6, .4])*nxt.to_numpy())
        self.assertAlmostEqual(report.iloc[-1]['var_0.95'], -np.quantile(empirical, .05))
        self.assertIn('risk_basis', report)

    def test_mc_rescaling(self):
        record = dict(preset('ged-skewed'), status='ok')
        normalized = dict(record, vol_standardization='ewma', next_volatility=[.5, 2.])
        a, x = simulate_joint_portfolio(normalized, [.6, -.4], simulations=2000)
        b, y = simulate_joint_portfolio(record, [.3, -.8], simulations=2000)
        np.testing.assert_array_equal(x, y)
        self.assertEqual(a, b)

    def test_cli_saved_metadata_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame(np.random.default_rng(3).normal(size=(130, 2))*.01,
                                 columns=['A', 'B'], index=pd.date_range('2020-01-01', periods=130))
            frame.to_csv(path/'returns.csv')
            args = [str(path/'returns.csv'), '--input-type', 'returns', '--models', 'normal',
                    '--days', '50', '70', '--standardize-vol', 'ewma', '--vol-halflife', '11.20230558',
                    '--univariate', '--weights', 'A=.6', 'B=.4', '--output', str(path/'fit.json')]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(args), 0)
            fits = json.loads((path/'fit.json').read_text())
            self.assertEqual(len(fits), 2)
            self.assertEqual(fits[0]['next_volatility'], fits[1]['next_volatility'])
            self.assertAlmostEqual(fits[0]['vol_lambda'], .94, places=8)
            self.assertEqual(fits[0]['parameter_units'], 'standardized returns')
            self.assertIn('Next-period conditional risk', out.getvalue())
            self.assertIn('standardized_loglik', fits[0])
            for bad in [['--vol-lambda', '.94'], ['--standardize-vol', 'ewma', '--vol-lambda', '1']]:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main([str(path/'returns.csv'), *bad])


if __name__ == '__main__':
    unittest.main()
