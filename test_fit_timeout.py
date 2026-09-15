import contextlib
import io
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.fit_timeout import run_fit, FitTimeout
from return_distributions.fitting import fit_many, fit_one
from return_distributions.cli import main
from return_distributions.joint_cli import main as joint_main


def slow_fit():
    time.sleep(60)


def broken_fit():
    raise ValueError('worker failure example')


class FitTimeoutTests(unittest.TestCase):
    def test_process_timeout_cleanup_and_success(self):
        before = {p.pid for p in mp.active_children()}
        started = time.perf_counter()
        with self.assertRaises(FitTimeout) as caught:
            run_fit(slow_fit, fit_timeout=.25)
        self.assertGreaterEqual(caught.exception.elapsed, .25)
        self.assertLess(time.perf_counter()-started, 15.)
        self.assertEqual({p.pid for p in mp.active_children()}, before)
        x = np.random.default_rng(1).normal(size=30)
        result = run_fit(fit_one, x, 'normal', fit_timeout=30.)
        self.assertEqual(result['status'], 'ok')
        self.assertAlmostEqual(result['loc'], x.mean())
        with self.assertRaisesRegex(ValueError, 'worker failure example'):
            run_fit(broken_fit, fit_timeout=30.)
        self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_default_no_worker_and_model_continuation(self):
        x = np.random.default_rng(2).normal(size=30)
        with patch('return_distributions.fit_timeout.mp.get_context', side_effect=AssertionError('no worker')):
            self.assertEqual(fit_many(x, ['normal']).iloc[0].status, 'ok')
        normal = fit_one(x, 'normal')
        with patch('return_distributions.fit_timeout.run_fit', side_effect=[FitTimeout(.1,.12),normal]) as fitter:
            result = fit_many(x, ['student-t','normal'], fit_timeout=.1)
        self.assertEqual(fitter.call_count, 2)
        timed = result.loc[result.name.eq('student-t')].iloc[0]
        self.assertEqual(timed.status, 'timeout')
        self.assertTrue(np.isnan(timed['aic_rank']))
        self.assertTrue(np.isnan(timed['bic_rank']))
        self.assertEqual(result.iloc[0]['name'], 'normal')

    def test_cli_timeout_files_and_validation(self):
        sample = pd.DataFrame(np.random.default_rng(3).normal(size=(30,2)), columns=['A','B'],
                              index=pd.bdate_range('2020-01-01', periods=30))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'returns.csv'
            sample.to_csv(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source),'--input-type','returns','--models','normal','laplace',
                    '--fit-timeout','.001','--output',str(root/'uni.csv')]), 1)
            table = pd.read_csv(root/'uni.csv')
            self.assertEqual(len(table), 4)
            self.assertTrue(table.status.eq('timeout').all())
            self.assertTrue(table['aic_rank'].isna().all())
            self.assertTrue(table['bic_rank'].isna().all())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(joint_main([str(source),'--input-type','returns','--models','normal','generalized-t-skewed',
                    '--asset-orders','all','--univariate','--fit-timeout','.001','--weights','A=.6','B=.4',
                    '--js-distance','--output',str(root/'joint.json')]), 1)
            joint = pd.read_csv(root/'joint.csv')
            self.assertEqual(len(joint), 3)
            self.assertTrue(joint.status.eq('timeout').all())
            self.assertTrue(joint['aic_rank'].isna().all())
            self.assertTrue(joint['bic_rank'].isna().all())
            self.assertTrue(pd.read_csv(root/'joint_univariate.csv').status.eq('timeout').all())
            risk = pd.read_csv(root/'joint_portfolio_risk.csv')
            self.assertTrue(risk.loc[risk.model.ne('empirical')].status.eq('skipped').all())
            for entry in [main, joint_main]:
                for value in ['0','-1','nan','inf']:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        entry([str(source),'--fit-timeout',value])
            with patch('pandas.DataFrame.to_csv',side_effect=AssertionError('no save')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source),'--input-type','returns','--models','normal',
                    '--fit-timeout','.001','--no-save']), 1)


if __name__ == '__main__':
    unittest.main()
