"""Test copula fitting within the joint-distribution command-line workflow.
Cover shared samples, cached marginals, likelihood scores, risk, and validation.
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
from return_distributions.joint_cli import main
from return_distributions.fitting import fit_many


class JointCopulaTests(unittest.TestCase):
    def test_shared_blocks_cache_scores_and_risk(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            pd.DataFrame(np.random.default_rng(22).normal(0,.01,(220,2)),columns=['A','B'],
                         index=pd.date_range('2020-01-01',periods=220)).to_csv(p/'r.csv')
            args=[str(p/'r.csv'),'--input-type','returns','--models','normal','--copulas','gaussian',
                  '--marginal-models','normal','--univariate','--days','160','--subperiods','2',
                  '--standardize-vol','none','ewma','--return-scale','100','--vol-warmup','5',
                  '--weights','A=.6','B=.4','--risk-levels','.95','--simulations','2000','--mc-batches','4',
                  '--output',str(p/'fits.json')]
            with (contextlib.redirect_stdout(io.StringIO()), patch('return_distributions.joint_copula.fit_many',wraps=fit_many) as fitter,
                    patch('return_distributions.joint_univariate.fit_many',side_effect=AssertionError('Should reuse marginals'))):
                self.assertEqual(main(args),0)
                self.assertEqual(fitter.call_count,8)
            records=json.loads((p/'fits_copulas.json').read_text())
            joint=json.loads((p/'fits.json').read_text())
            self.assertEqual(len(records),4)
            for record, fit in zip(records,joint):
                self.assertEqual(record['first_date'],fit['first_date'])
                self.assertEqual(record['last_date'],fit['last_date'])
                self.assertEqual(record['observations'],80)
                self.assertAlmostEqual(record['summary']['joint_loglik'],fit['loglik'],places=5)
                self.assertNotIn('aic_rank',record['summary'])
            self.assertEqual(len(pd.read_csv(p/'fits_copula_risk.csv')),8)

    def test_invalid_models(self):
        for args in [['--copulas','gaussian','--marginal-models','bad'], ['--marginal-models','normal']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(['missing.csv',*args])


if __name__=='__main__': unittest.main()
