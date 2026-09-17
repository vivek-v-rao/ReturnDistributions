"""Test screen-only runs across univariate, joint, and copula CLIs.
Ensure outputs are untouched while plotting and failure reporting still work.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.cli import main as univariate_main
from return_distributions.joint_cli import main as joint_main
from return_distributions.copula_cli import main as copula_main


class NoSaveTests(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame(np.random.default_rng(12).normal(0, .01, (120, 2)),
                                   columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=120))
        self.programs = [
            (univariate_main, 'cli', ['--models', 'normal'], 'fits.csv', 'KS is descriptive'),
            (joint_main, 'joint_cli', ['--models', 'normal', '--weights', 'A=.6', 'B=.4'], 'fits.json', 'Portfolio VaR'),
            (copula_main, 'copula_cli', ['--marginal-models', 'normal', '--copulas', 'gaussian'], 'fits.json', 'Copula fit comparison'),
        ]

    def test_no_writes_existing_or_new_paths_and_ignored_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root/'input.csv'
            sentinel.write_text('preserve existing input/output', encoding='utf-8')
            for main, module, options, filename, heading in self.programs:
                for output in [root/'absent'/filename, sentinel, Path('.')]:
                    args = [str(sentinel), *options, '--output', str(output), '--no-save']
                    if module == 'joint_cli': args += ['--portfolio-output', str(root/'risk'/'ignored.txt')]
                    with patch(f'return_distributions.{module}.read_returns', return_value=self.sample), \
                         patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir called')), \
                         patch.object(Path, 'write_text', side_effect=AssertionError('write called')), \
                         patch.object(pd.DataFrame, 'to_csv', side_effect=AssertionError('CSV called')), \
                         contextlib.redirect_stdout(io.StringIO()) as console:
                        self.assertEqual(main(args), 0)
                    self.assertIn(heading, console.getvalue())
                    self.assertIn('Overall elapsed:', console.getvalue())
                    self.assertNotIn('Wrote ', console.getvalue())
            self.assertEqual(list(root.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserve existing input/output')

    def test_saving_still_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for main, module, options, filename, heading in self.programs:
                output = root/module/filename
                with patch(f'return_distributions.{module}.read_returns', return_value=self.sample), \
                     contextlib.redirect_stdout(io.StringIO()) as console:
                    self.assertEqual(main(['unused.csv', *options, '--output', str(output)]), 0)
                self.assertTrue(output.exists())
                self.assertIn('Wrote ', console.getvalue())
                if module == 'joint_cli': self.assertTrue(output.with_name('fits_portfolio_risk.csv').exists())
                if module == 'copula_cli': self.assertTrue(output.with_name('fits_marginals.csv').exists())

    def test_plot_and_failure_status_unchanged(self):
        with patch('return_distributions.cli.read_returns', return_value=self.sample), \
             patch('return_distributions.cli.plot_fits') as plot, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(univariate_main(['unused.csv', '--models', 'normal', '--show-plot', '--no-save']), 0)
        self.assertEqual(plot.call_count, 2)
        with patch('return_distributions.joint_cli.read_returns', return_value=self.sample), \
             patch('return_distributions.joint_cli.fit_joint', side_effect=ValueError('numerical failure')), \
             contextlib.redirect_stdout(io.StringIO()) as console:
            self.assertEqual(joint_main(['unused.csv', '--models', 'normal', '--no-save']), 1)
        self.assertIn('numerical failure', console.getvalue())


if __name__ == '__main__':
    unittest.main()
