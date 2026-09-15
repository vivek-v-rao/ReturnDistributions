import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from return_distributions.cli import main
from return_distributions.univariate_analysis import all_project_models, prepare_samples, samples
from return_distributions.fitting import specification, fitted_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.vol_standardization import ewma_standardize


class AnalysisTests(unittest.TestCase):
    def test_catalog(self):
        models = all_project_models()
        self.assertEqual(len(models), len(set(models)))
        self.assertIn('generalized-t-skewed', models)
        self.assertNotIn('gamma', models)
        self.assertNotIn('fs_skew_t', models)
        for name in models: specification(name)

    def test_common_alignment(self):
        frame = pd.DataFrame({'A': np.arange(100)+1., 'B': np.arange(100)+2.}, index=pd.date_range('2020-01-01', periods=100))
        frame.loc[frame.index[60], 'B'] = np.nan
        cache, eligible = prepare_samples(frame, ['none', 'ewma'], [.94, .97], 5, 1e-8, True)
        batches = [list(samples(frame, s, [20], ['none', 'ewma'], [.94, .97], cache, eligible)) for s in frame]
        for group in batches:
            for batch in group: self.assertTrue(batch[3].index.equals(batches[0][0][3].index))
        self.assertEqual(len(cache), 2)

    def test_ewma_scores_risk_saved_units(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            x = np.random.default_rng(12).normal(.001, .01, 120)
            frame = pd.DataFrame({'A': x}, index=pd.date_range('2020-01-01', periods=120))
            frame.to_csv(p/'x.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(p/'x.csv'), '--input-type', 'returns', '--models', 'normal',
                    '--days', '50', '--standardize-vol', 'none', 'ewma', '--vol-lambda', '.94', '.97',
                    '--vol-warmup', '5', '--risk-levels', '.95', '.99', '--output', str(p/'fit.csv')])
            self.assertEqual(code, 0)
            table = pd.read_csv(p/'fit.csv')
            self.assertEqual(len(table), 3)
            self.assertEqual(table.first_date.nunique(), 1)
            self.assertEqual(table.last_date.nunique(), 1)
            risks = pd.read_csv(p/'fit_risk.csv')
            self.assertEqual(len(risks), 6)
            for _, row in table.loc[table.vol_standardization.eq('ewma')].iterrows():
                z, scales, nxt = ewma_standardize(frame, row.vol_lambda, 5)
                dist = fitted_distribution(row)
                ll = dist.logpdf(z.A.tail(50)).sum()-np.log(scales.A.tail(50)).sum()
                self.assertAlmostEqual(ll, row.loglik)
                risk = risks.loc[(risks.name=='normal') & np.isclose(risks.vol_lambda, row.vol_lambda)].iloc[0]
                self.assertAlmostEqual(risk['ES_99%'], portfolio_risk(dist, .99)['es']*nxt.A)

    def test_all_expansion_without_fitting(self):
        with patch('return_distributions.cli.read_returns', side_effect=ValueError('stop')) as reader:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['missing.csv', '--models', 'all', '--no-save']), 1)
            reader.assert_called_once()

    def test_invalid_options(self):
        for options in [['--vol-lambda', '.94'], ['--risk-levels', '1'],
                        ['--models', 'all', 'normal'], ['--standardize-vol', 'ewma', '--vol-lambda', '0']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['missing.csv', *options])


if __name__ == '__main__': unittest.main()
