import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from return_distributions.joint_cli import main
from return_distributions.portfolio_cli import main as portfolio_main
from return_distributions.joint_risk_report import risk_report
from return_distributions.multivariate import fit_joint
from return_distributions.copula_portfolio import empirical_es


class JointRiskTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(np.random.default_rng(7).normal(0,.01,(40,2)),
            columns=['A','B'],index=pd.bdate_range('2020-01-01',periods=40))

    def test_empirical_signed_zero_and_skipped(self):
        sample=self.sample(); fit=fit_joint(sample.to_numpy(),'normal')
        fit.update(symbols=list(sample.columns),return_type='simple',window=len(sample))
        for weights in [{'A':.6,'B':.4},{'A':1.,'B':-.5},{'A':0.}]:
            result=risk_report([fit,dict(fit,model='failed-model',status='boundary')],sample,weights,[.95,.975])
            empirical=result.iloc[-1]
            values=sample.A*weights.get('A',0)+sample.B*weights.get('B',0)
            self.assertAlmostEqual(empirical['var_0.95'],-np.quantile(values,.05))
            self.assertAlmostEqual(empirical['es_0.975'],empirical_es(values.to_numpy(),.025))
            self.assertEqual(result.iloc[1].status,'skipped')
            self.assertTrue(pd.isna(result.iloc[1]['var_0.95']))
            if not any(weights.values()): self.assertEqual(result.iloc[0]['es_0.95'],0.)
        with self.assertRaises(ValueError): risk_report([fit],sample,{'C':1.},[.95])

    def test_windows_and_existing_cli_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'returns.csv'; output=root/'fits.json'
            sample=self.sample(); sample.iloc[5,0]=np.nan; sample.to_csv(source)
            args=[str(source),'--input-type','returns','--models','normal','--days','20','30',
                  '--weights','B=0.4','A=0.6','--risk-levels','.95','.975','--output',str(output)]
            with contextlib.redirect_stdout(io.StringIO()) as console:
                self.assertEqual(main(args),0)
            report=pd.read_csv(root/'fits_portfolio_risk.csv')
            self.assertEqual(len(report),4)
            self.assertEqual(set(report.observations),{20,30})
            self.assertIn('VAR_97.5%',console.getvalue())
            self.assertIn('empirical',console.getvalue())
            for window in [20,30]:
                expected=sample.dropna().tail(window)
                row=report[(report.window==window)&(report.model=='empirical')].iloc[0]
                self.assertEqual(row.first_date,str(expected.index[0].date()))
                self.assertAlmostEqual(row['var_0.95'],-np.quantile(expected.to_numpy()@np.array([.6,.4]),.05))
            existing=root/'existing.csv'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(portfolio_main([str(output),'--weights','A=.6','B=.4','--risk-levels','.95','.975','--output',str(existing)]),0)
            old=pd.read_csv(existing)
            new=report[report.model=='normal'].sort_values('window')
            np.testing.assert_allclose(new[['var_0.95','es_0.95','var_0.975','es_0.975']],
                old.sort_values('window')[['var_0.95','es_0.95','var_0.975','es_0.975']],rtol=1e-10)

    def test_validation_and_weights_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'returns.csv'; self.sample().to_csv(source)
            weights=root/'weights.csv'; pd.DataFrame({'symbol':['A','B'],'weight':[.6,.4]}).to_csv(weights,index=False)
            base=[str(source),'--input-type','returns','--models','normal','--output',str(root/'fits.json')]
            for extra in [['--risk-levels','.95'],['--weights','A=1','--risk-levels','nan'],
                          ['--weights','A=1','--return-type','log'],
                          ['--weights','A=1','--portfolio-output',str(source)],
                          ['--weights-file',str(weights),'--portfolio-output',str(weights)],
                          ['--weights','A=1','--portfolio-output',str(root/'fits.csv')]]:
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main(base+extra)
            with patch('return_distributions.joint_cli.fit_joint') as fitter,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(base+['--weights','C=1']),1)
                fitter.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(base+['--weights-file',str(weights),'--portfolio-output',str(root/'risk.csv')]),0)
            self.assertIn('es_0.995',pd.read_csv(root/'risk.csv'))


if __name__=='__main__': unittest.main()
