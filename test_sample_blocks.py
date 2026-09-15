import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from return_distributions.cli import main as uni
from return_distributions.joint_cli import main as joint
from return_distributions.sample_blocks import blocks, endpoint_scale
from return_distributions.vol_standardization import ewma_standardize


class BlockTests(unittest.TestCase):
    def test_balanced(self):
        idx = pd.date_range('2020-01-01',periods=101)
        parts = list(blocks(idx, [None], [1,2,4]))
        self.assertEqual(len(parts),7)
        self.assertEqual([len(p[3]) for p in parts[-4:]], [26,25,25,25])
        self.assertEqual(list(np.concatenate([p[3] for p in parts[-4:]])),list(idx))

    def test_cli_blocks_and_endpoint_scales(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame(np.random.default_rng(2).normal(0,.01,(180,2)),columns=['A','B'],index=pd.date_range('2020-01-01',periods=180))
            frame.to_csv(path/'r.csv')
            for main, suffix in [(uni,'.csv'),(joint,'.json')]:
                out = path/('fits'+suffix)
                args = [str(path/'r.csv'),'--input-type','returns','--models','normal','--date-min','2020-02-01',
                        '--date-max','2020-05-31','--days','100','--subperiods','1','2','4',
                        '--standardize-vol','none','ewma','--vol-warmup','5','--output',str(out)]
                if main is uni: args += ['--symbols','A']
                with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(main(args),0)
                result = pd.read_csv(out.with_suffix('.csv'))
                self.assertEqual(len(result),14)
                self.assertTrue(result.aic_rank.max()<=2)
                self.assertTrue((result.last_date<='2020-05-31').all())
                for (_, _), group in result.groupby(['subperiods','block']):
                    self.assertEqual(group.first_date.nunique(),1)
                    self.assertEqual(group.last_date.nunique(),1)
                if main is uni:
                    for _, row in result.loc[result.vol_standardization.eq('ewma')].iterrows():
                        _, _, nxt = ewma_standardize(frame.loc[:row.last_date],.94,5)
                        self.assertAlmostEqual(row.next_volatility,nxt.A)

    def test_return_before_date_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            idx=pd.date_range('2020-01-01',periods=40)
            pd.DataFrame({'A':100*np.exp(np.arange(40)**1.1*.001)},index=idx).to_csv(p/'p.csv')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(uni([str(p/'p.csv'),'--models','normal','--date-min','2020-01-10','--date-max','2020-01-20','--output',str(p/'f.csv')]),0)
            row=pd.read_csv(p/'f.csv').iloc[0]
            self.assertEqual(row.n,11)
            self.assertEqual(row.first_date,'2020-01-10')

    def test_invalid(self):
        for main in (uni,joint):
            for args in [['--subperiods','0'],['--date-min','bad'],['--date-min','2021-01-01','--date-max','2020-01-01']]:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(['missing.csv',*args])


if __name__=='__main__': unittest.main()
