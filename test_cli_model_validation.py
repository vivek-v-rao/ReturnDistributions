import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.cli import main
from return_distributions.fitting import model_catalog, specification, DEFAULT_MODELS
from return_distributions.model_names import display_model_name


class ModelValidationTests(unittest.TestCase):
    def test_help_lists_available_models_without_loading_data(self):
        with patch('return_distributions.cli.read_returns') as reader, \
             contextlib.redirect_stdout(io.StringIO()) as console, \
             self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        reader.assert_not_called()
        help_text = console.getvalue().split('--models MODEL [MODEL ...]')[-1].split('--days')[0]
        self.assertIn('Project models (preferred names)', help_text)
        for name in model_catalog()['Project models and aliases']:
            self.assertIn(display_model_name(name), help_text)
            specification(name)
        for name in DEFAULT_MODELS:
            self.assertIn(display_model_name(name), help_text)
        self.assertNotIn('Additional SciPy', help_text)
        self.assertNotIn('wrapcauchy', help_text)
        for base in ['nig', 'gh', 'hyperbolic', 'variance-gamma', 'meixner', 'egb2', 'nts']:
            self.assertNotIn(base+'-skewed', help_text)
            self.assertIs(specification(base)[0], specification(base+'-skewed')[0])
        for distinct in ['ged-skewed', 'laplace-skewed', 'generalized-t-skewed']:
            self.assertIn(distinct, help_text)
        self.assertIn('fs-skew-normal', help_text)
        for hidden, preferred in [
            ('azzalini', 'azzalini-skew-t'), ('azzalini_skew_t', 'azzalini-skew-t'),
            ('fernandez-steel', 'fs-skew-t'), ('fs_skew_t', 'fs-skew-t'),
            ('generalized-t-symmetric', 'generalized-t'),
            ('laplace_asymmetric', 'laplace-skewed'), ('nef-ghs', 'meixner'),
        ]:
            listed = help_text.split('Project models (preferred names):')[1].split('Older spellings')[0]
            listed_names = ''.join(listed.split()).split(',')
            self.assertNotIn(hidden, listed_names)
            self.assertIn(preferred, listed_names)
            self.assertEqual(display_model_name(hidden), preferred)
            old_dist, old_names, old_fixed = specification(hidden)
            new_dist, new_names, new_fixed = specification(preferred)
            self.assertIs(old_dist, new_dist)
            self.assertEqual((old_names, old_fixed), (new_names, new_fixed))

    def test_unknown_names_rejected_before_io_or_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'existing.csv'
            output.write_text('preserve me', encoding='utf-8')
            with patch('return_distributions.cli.read_returns') as reader, \
                 patch('return_distributions.cli.fit_many') as fitter, \
                 contextlib.redirect_stderr(io.StringIO()) as error, \
                 self.assertRaises(SystemExit) as raised:
                main(['missing.csv', '--models', 'normal', 'laplace-typo',
                      'unknown-model', 'laplace-typo', '--output', str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('model(s): laplace-typo, unknown-model', error.getvalue())
            reader.assert_not_called()
            fitter.assert_not_called()
            self.assertEqual(output.read_text(encoding='utf-8'), 'preserve me')

    def test_existing_names_aliases_and_custom_models_pass(self):
        names = ['normal', 'norm', 'ged', 'ged-skewed', 'nig', 'laplace_asymmetric', 'laplace-skewed']
        with patch('return_distributions.cli.read_returns', side_effect=OSError('test read')) as reader, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['missing.csv', '--models', *names]), 1)
        reader.assert_called_once()

    def test_numerical_failures_remain_rows_and_other_fits_continue(self):
        sample = pd.DataFrame({'A': np.linspace(-.02, .02, 20)},
                              index=pd.bdate_range('2020-01-01', periods=20))
        def fit(x, name, **kwargs):
            if name == 'student-t':
                raise ArithmeticError('test numerical failure')
            return dict(name=name, status='ok', aic=0., bic=0.)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fits.csv'
            with patch('return_distributions.cli.read_returns', return_value=sample), \
                 patch('return_distributions.fitting.fit_one', side_effect=fit) as fitter, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['unused.csv', '--models', 'student-t', 'normal',
                                       '--output', str(output)]), 1)
            self.assertEqual(fitter.call_count, 2)
            saved = pd.read_csv(output).set_index('name')
            self.assertEqual(saved.loc['student-t', 'status'], 'failed')
            self.assertEqual(saved.loc['normal', 'status'], 'ok')


if __name__ == '__main__':
    unittest.main()
