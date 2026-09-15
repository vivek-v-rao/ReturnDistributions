import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from return_distributions.cli import main as uni
from return_distributions.joint_cli import main as joint


class ReturnScaleTests(unittest.TestCase):
    def test_scores_parameters_and_risk(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            pd.DataFrame(np.random.default_rng(4).normal(0,.01,(120,2)), columns=['A','B'],
                         index=pd.date_range('2020-01-01',periods=120)).to_csv(p/'r.csv')
            for main, dimension in [(uni, 1), (joint, 2)]:
                outputs = []
                for factor in (1,100):
                    target = p/f'f{dimension}_{factor}.csv' if dimension==1 else p/f'f{dimension}_{factor}.json'
                    args = [str(p/'r.csv'), '--input-type','returns','--models','normal', '--days','50',
                            '--standardize-vol','none','ewma','--vol-warmup','5', '--return-scale',str(factor),
                            '--risk-levels','.99','--output',str(target)]
                    args += ['--symbols','A'] if dimension==1 else ['--weights','A=.6','B=.4']
                    with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(main(args),0)
                    table = pd.read_csv(target if dimension==1 else target.with_suffix('.csv'))
                    risk = pd.read_csv(target.with_name(target.stem+('_risk.csv' if dimension==1 else '_portfolio_risk.csv')))
                    outputs.append((table,risk))
                small,big = outputs[0][0],outputs[1][0]
                np.testing.assert_allclose(big.loglik-small.loglik,-50*dimension*np.log(100),atol=1e-7)
                np.testing.assert_allclose(big.aic-small.aic,100*dimension*np.log(100),atol=1e-7)
                self.assertTrue(big.return_scale.eq(100).all())
                if dimension==1:
                    self.assertAlmostEqual(big.iloc[0]['scale']/small.iloc[0]['scale'],100)
                    self.assertAlmostEqual(big.iloc[1]['scale']/small.iloc[1]['scale'],1)
                key = 'ES_99%' if dimension==1 else 'es_0.99'
                np.testing.assert_allclose(outputs[0][1][key],outputs[1][1][key],atol=1e-10)

    def test_invalid(self):
        for main in (uni,joint):
            for value in ('0','-1','nan','inf'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['missing.csv','--return-scale',value])


if __name__ == '__main__': unittest.main()
