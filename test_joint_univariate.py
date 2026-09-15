import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.fitting import fit_one, specification
from return_distributions.joint_cli import main
from return_distributions.joint_univariate import model_mapping, format_univariate_comparison
from return_distributions.multivariate import ALL_JOINT_MODELS


class JointUnivariateTests(unittest.TestCase):
    def test_compact_console_and_preserved_data(self):
        table = pd.DataFrame(dict(name=['normal', 'student-t'], scipy_distribution=['norm', 't'],
            n=[40, 40], ks_p=[np.nan, np.nan], k=[2, 3], aic=[-10., -12.],
            converged=[True, True], status=['ok', 'ok'], rank=[2, 1],
            joint_models=['normal', 'student-t'], symbol=['A', 'A'], window=[40, 40],
            first_date=['2020-01-01']*2, last_date=['2020-02-01']*2,
            return_type=['simple']*2, sample_policy=['joint-complete-case']*2))
        original = table.copy(deep=True)
        text = format_univariate_comparison(table)
        self.assertTrue(text.startswith('Status: all fits ok\n'))
        self.assertIn('Warnings: none', text)
        header = next(line.split() for line in text.splitlines() if line.split()[:1] == ['name'])
        self.assertEqual(header, ['name', 'k', 'aic', 'rank'])
        pd.testing.assert_frame_equal(table, original)
        table.loc[1, 'status'] = 'boundary'
        text = format_univariate_comparison(table)
        self.assertFalse(text.startswith('Status:'))
        header = next(line.split() for line in text.splitlines() if line.split()[:1] == ['name'])
        self.assertIn('status', header)
        self.assertNotIn('converged', header)
        table.loc[1, 'converged'] = False
        text = format_univariate_comparison(table)
        header = next(line.split() for line in text.splitlines() if line.split()[:1] == ['name'])
        self.assertIn('converged', header)

    def test_diagnostics_below_table_and_warnings_when_present(self):
        table = pd.DataFrame({'name': ['ged', 'generalized-t', 'normal'],
            'status': ['ok']*3, 'converged': [True]*3, 'aic': [-2., -3., -1.],
            'warnings': ['', None, np.nan],
            'optimizer_message': ['Optimization terminated successfully.',
                                  'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL', ''],
            'fit_restriction': ['power bounded', 'power bounded', None]})
        original = table.copy(deep=True)
        text = format_univariate_comparison(table)
        self.assertIn('Warnings: none', text)
        self.assertNotIn('optimizer_message', text)
        self.assertNotIn('Optimizer messages:', text)
        self.assertNotIn('fit_restriction', text)
        self.assertEqual(text.count('power bounded'), 1)
        self.assertIn('Fit restrictions:\n  ged, generalized-t: power bounded', text)
        pd.testing.assert_frame_equal(table, original)
        table.loc[1, ['status', 'converged', 'warnings', 'optimizer_message']] = [
            'not_converged', False, 'precision loss', 'Iteration limit reached']
        text = format_univariate_comparison(table)
        self.assertNotIn('Warnings: none', text)
        self.assertIn('warnings', next(line.split() for line in text.splitlines() if line.split()[:1] == ['name']))
        self.assertIn('precision loss', text)
        self.assertIn('Optimizer messages:\n  generalized-t: Iteration limit reached', text)

    def setUp(self):
        self.frame = pd.DataFrame(np.random.default_rng(4).normal(0, .01, (40, 2)),
                                  columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=40))
        self.frame.iloc[[3, 32], 0] = np.nan
        self.frame.iloc[[4, 35], 1] = np.nan

    def test_same_sample_windows_saved_results_and_shared_fitter(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fits.json'
            with patch('return_distributions.joint_cli.read_returns', return_value=self.frame), \
                 contextlib.redirect_stdout(io.StringIO()) as console:
                code = main(['unused.csv', '--models', 'normal', '--days', '12', '20',
                             '--location', '0', '--univariate', '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertIn('NOT implied marginals', console.getvalue())
            self.assertIn('Separately estimated univariate fits - A:', console.getvalue())
            saved = pd.read_csv(output.with_name('fits_univariate.csv'))
            self.assertEqual(len(saved), 4)
            for _, row in saved.iterrows():
                sample = self.frame.dropna().tail(int(row['window']))
                ref = fit_one(sample[row.symbol].to_numpy(), 'normal', location=0., max_iterations=2000)
                self.assertAlmostEqual(row.loglik, ref['loglik'])
                self.assertEqual(row.n, len(sample))
                self.assertEqual(row.first_date, str(sample.index[0].date()))
                self.assertEqual(row.last_date, str(sample.index[-1].date()))
                self.assertAlmostEqual(row['loc'], 0., places=15)
                self.assertEqual(row.joint_models, 'normal')

    def test_mapping_dedup_and_unsupported(self):
        mapped, skipped = model_mapping(ALL_JOINT_MODELS)
        self.assertEqual(set(skipped), {'slash-normal', 'skew-slash-normal', 'slash-t', 'skew-slash-t'})
        self.assertEqual(mapped['laplace'], ['laplace', 'laplace-mixture-symmetric'])
        self.assertEqual(mapped['nct'], ['noncentral-t'])
        self.assertEqual(mapped['azzalini-skew-t'], ['azzalini-skew-t', 'sdb-skew-t'])
        for name in mapped: specification(name)
        stub = dict(model='slash-normal', status='ok', observations=40, dimensions=2,
                    parameters=6, loglik=1., aic=10., bic=12.)
        with patch('return_distributions.joint_cli.read_returns', return_value=self.frame), \
             patch('return_distributions.joint_cli.fit_joint', return_value=stub), \
             patch('return_distributions.joint_univariate.fit_many') as fitter, \
             contextlib.redirect_stdout(io.StringIO()) as console:
            self.assertEqual(main(['unused.csv', '--models', 'slash-normal', '--univariate', '--no-save']), 0)
        fitter.assert_not_called()
        self.assertIn('skipped for slash-normal', console.getvalue())

    def test_no_save_and_numerical_failure(self):
        failure = pd.DataFrame([dict(name='normal', status='failed', error='test failure')])
        with patch('return_distributions.joint_cli.read_returns', return_value=self.frame), \
             patch('return_distributions.joint_univariate.fit_many', return_value=failure) as fitter, \
             patch.object(pd.DataFrame, 'to_csv', side_effect=AssertionError('must not save')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('must not mkdir')), \
             contextlib.redirect_stdout(io.StringIO()) as console:
            self.assertEqual(main(['unused.csv', '--models', 'normal', '--univariate', '--no-save']), 1)
        self.assertEqual(fitter.call_count, 2)
        self.assertIn('test failure', console.getvalue())
        self.assertIn('Joint fit comparison:', console.getvalue())

    def test_output_collision_and_default_unchanged(self):
        with patch('return_distributions.joint_cli.read_returns') as reader, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['fits_univariate.csv', '--models', 'normal', '--univariate', '--output', 'fits.json'])
        reader.assert_not_called()
        with patch('return_distributions.joint_cli.read_returns', return_value=self.frame), \
             patch('return_distributions.joint_cli.fit_univariate_sample') as fitter, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['unused.csv', '--models', 'normal', '--no-save']), 0)
        fitter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
