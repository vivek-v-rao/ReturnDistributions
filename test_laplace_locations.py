import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from return_distributions.joint_laplace_mixture import pilot_location
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.joint_cli import main
from return_distributions.portfolio_cli import main as portfolio_main


class LaplaceLocationTests(unittest.TestCase):
    def sample(self):
        return np.random.default_rng(25).normal(size=(160,2))*[.01,.02]+[.003,-.001]

    def test_pilot_methods(self):
        x=self.sample()
        median,_=pilot_location(x,'median')
        np.testing.assert_array_equal(median,np.median(x,axis=0))
        for bandwidth in ['scott','silverman',.7]:
            mu,info=pilot_location(x,'mode',bandwidth)
            kde=stats.gaussian_kde(x.T,bw_method=bandwidth)
            self.assertAlmostEqual(info['pilot_log_density'],kde.logpdf(mu)[0],places=9)
            for j in range(2):
                step=np.zeros(2);step[j]=1e-7
                self.assertLess(abs((kde.logpdf(mu+step)-kde.logpdf(mu-step))[0]/2e-7),.01)
            moved,_=pilot_location(x*[2.,3.]+[1.,-2.],'mode',bandwidth)
            np.testing.assert_allclose(moved,mu*[2.,3.]+[1.,-2.],atol=1e-6)

    def test_two_stage_metadata(self):
        x=self.sample()
        for method in ['median','mode']:
            fit=fit_joint(x,'asymmetric-laplace',laplace_location=method,max_iterations=300)
            self.assertEqual(fit['status'],'ok')
            self.assertEqual(fit['estimation_method'],'two-stage')
            self.assertIsNone(fit['aic']); self.assertIsNone(fit['bic'])
            self.assertEqual(fit['parameters'],7);self.assertEqual(fit['conditional_parameters'],5)
            self.assertAlmostEqual(fit['two_stage_aic'],14-2*fit['loglik'])
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=7)
            json.dumps(fit,allow_nan=False)
        # Componentwise medians can equal an observed vector; no jitter/floor.
        centered=np.vstack([x,-x,[0.,0.]])
        with self.assertRaisesRegex(ValueError,'singular'):
            fit_joint(centered,'asymmetric-laplace',laplace_location='median')
        with self.assertRaises(ValueError): fit_joint(x,'asymmetric-laplace',laplace_location='mode',laplace_mode_bandwidth=0)

    def test_cli_windows_ranks_and_portfolio(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'returns.csv'; output=root/'fits.json'
            pd.DataFrame(self.sample(),columns=['A','B'],index=pd.bdate_range('2020-01-01',periods=160)).to_csv(source)
            args=[str(source),'--input-type','returns','--models','normal','asymmetric-laplace',
                  '--laplace-location','zero','median','mode','zero','--laplace-mode-bandwidth','.7',
                  '--days','80','120','--weights','A=.6','B=.4','--risk-levels','.95','--output',str(output)]
            with contextlib.redirect_stdout(io.StringIO()) as console:
                self.assertEqual(main(args),0)
            fits=json.loads(output.read_text()); self.assertEqual(len(fits),8)
            summary=pd.read_csv(root/'fits.csv')
            pilots=summary[summary.estimation_method=='two-stage']
            self.assertEqual(len(pilots),4); self.assertTrue(pilots['aic_rank'].isna().all())
            self.assertTrue(pilots['bic_rank'].isna().all())
            self.assertTrue(pilots.two_stage_aic.notna().all())
            risk=pd.read_csv(root/'fits_portfolio_risk.csv')
            self.assertEqual(len(risk),10)
            self.assertEqual(set(risk.location_method.dropna()),{'zero','median','mode'})
            self.assertIn('two-stage',console.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(portfolio_main([str(output),'--weights','A=.6','B=.4','--risk-levels','.95','--output',str(root/'saved.csv')]),0)
            saved=pd.read_csv(root/'saved.csv')
            self.assertEqual(set(saved.location_method.dropna()),{'zero','median','mode'})

    def test_cli_validation(self):
        for extra in [['--laplace-location','median','--location','0'],
                      ['--models','normal','--laplace-location','mode'],
                      ['--laplace-location','median','--laplace-mode-bandwidth','.5'],
                      ['--laplace-location','mode','--laplace-mode-bandwidth','nan']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(['unused.csv','--models','asymmetric-laplace',*extra])


if __name__=='__main__': unittest.main()
