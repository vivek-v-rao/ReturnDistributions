import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from return_distributions.copula_portfolio import simulate_portfolio, empirical_es
from return_distributions.tail_risk import portfolio_risk
from return_distributions.portfolio_cli import main


def fixture():
    return dict(symbols=['A','B'],stage='two-stage',return_type='simple',marginal_mode='fitted',
        marginals=[dict(name='normal',loc=.001,scale=.02),dict(name='normal',loc=-.001,scale=.01)],
        copula=dict(model='gaussian',correlation=[[1,.4],[.4,1]],df=None,status='ok'))


class CopulaPortfolioTests(unittest.TestCase):
    def test_normal_reference_reproducibility(self):
        record = fixture()
        w = np.array([.6,-.4])
        row, draws = simulate_portfolio(record,w,simulations=40000,seed=9)
        repeated, other = simulate_portfolio(record,w,simulations=40000,seed=9)
        np.testing.assert_array_equal(draws,other)
        sigma = np.diag([.02,.01])@np.array([[1,.4],[.4,1]])@np.diag([.02,.01])
        ref = stats.norm(w@np.array([.001,-.001]),np.sqrt(w@sigma@w))
        self.assertAlmostEqual(row['es_0.99'],portfolio_risk(ref,.99)['es'],delta=.0008)
        self.assertEqual(row['tail_count_0.99'],400)
        self.assertTrue(np.isfinite(row['es_0.99_mc_se']))
        self.assertGreater(row['es_0.99'],row['var_0.99'])

    def test_marginal_not_copula_moments(self):
        record = fixture()
        record['copula'].update(model='student-t',df=.8)
        row,_ = simulate_portfolio(record,[.5,.5],simulations=2000)
        self.assertTrue(np.isfinite(row['es_0.99']))
        record['marginals'][0]=dict(name='student-t',df=.8,loc=0,scale=.02)
        row,_ = simulate_portfolio(record,[.5,.5],simulations=2000)
        self.assertTrue(np.isnan(row['es_0.99']))
        row,_ = simulate_portfolio(record,[0,1],simulations=2000)
        self.assertTrue(np.isfinite(row['es_0.99']))
        row,_ = simulate_portfolio(record,[0,0],simulations=2000)
        self.assertEqual(row['es_0.99'],0)

    def test_fractional_and_validation(self):
        self.assertAlmostEqual(empirical_es(np.array([-4.,-2.,0,2]),.375),10/3)
        record = fixture()
        record['return_type']='log'
        with self.assertRaises(ValueError): simulate_portfolio(record,[.5,.5])
        record['marginal_mode']='ranks'
        with self.assertRaises(ValueError): simulate_portfolio(record,[.5,.5],allow_log=True)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,output=Path(tmp)/'fits.json',Path(tmp)/'risk.csv'
            source.write_text(json.dumps([fixture()]))
            with contextlib.redirect_stdout(io.StringIO()):
                code=main([str(source),'--weights','A=.6','B=.4','--risk-levels','.95','.99','--simulations','2000','--output',str(output)])
            self.assertEqual(code,0)
            row=pd.read_csv(output).iloc[0]
            self.assertEqual(row['simulations'],2000)
            self.assertTrue(np.isfinite(row['es_0.95']))


if __name__ == '__main__': unittest.main()
