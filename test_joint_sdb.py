"""Test Sahu-Dey-Branco skew distributions and their symmetric limits.
Cover CDF accuracy, sampling, fitting, portfolio risk, and audit failures.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import stats
from return_distributions import fit_joint,joint_distribution,project_distribution,simulate_joint_portfolio
from return_distributions.joint_sdb import SDB_MODELS,orthant_logcdf
from return_distributions.multivariate import JOINT_MODELS,ALL_JOINT_MODELS
from return_distributions.simulation import preset
from return_distributions.skew_t import azzalini_skew_t
from return_distributions.portfolio_cli import main as portfolio_main


class SDBTests(unittest.TestCase):
    def test_univariate_and_symmetric_limits(self):
        x=np.linspace(-3,3,31)
        for model in SDB_MODELS:
            fit=dict(model=model,location=[.1],scatter=[[.49]],delta=[-.4],df=5.)
            scale=np.sqrt(.49+.16)
            reference=stats.skewnorm(-.4/.7,loc=.1,scale=scale) if model.endswith('normal') else azzalini_skew_t(5.,-.4/.7,loc=.1,scale=scale)
            dist=joint_distribution(fit)
            np.testing.assert_allclose(dist.pdf(x[:,None]),reference.pdf(x),rtol=1e-11)
            np.testing.assert_allclose(dist.mean(),reference.mean(),atol=1e-12)
            np.testing.assert_allclose(dist.cov(),reference.var(),atol=1e-12)
            fit=preset(model);fit['delta']=[0.,0.]
            dist=joint_distribution(fit);v=np.array([[.01,.02],[-.03,.01]])
            ref=stats.multivariate_normal(fit['location'],fit['scatter']) if model.endswith('normal') else stats.multivariate_t(fit['location'],fit['scatter'],fit['df'])
            np.testing.assert_allclose(dist.pdf(v),ref.pdf(v),rtol=1e-12)

    def test_cdf_and_normal_independence(self):
        c=np.array([[1.,.4],[.4,1.]])
        for df in [None,5.]:
            actual=np.exp(orthant_logcdf([[0.,0.]],c,df,4096,55))[0]
            self.assertAlmostEqual(actual,.25+np.arcsin(.4)/(2*np.pi),delta=2e-5)
            h=np.array([[-1.,.5],[2.,-1.]])
            ref=stats.multivariate_normal.cdf(h,cov=c,maxpts=1000000,abseps=1e-8) if df is None else stats.multivariate_t.cdf(h,shape=c,df=df,maxpts=1000000,random_state=22)
            np.testing.assert_allclose(np.exp(orthant_logcdf(h,c,df,8192,55)),ref,atol=8e-5)
        fit=dict(model='sdb-skew-normal',location=[0.,0.],scatter=[[1.,0.],[0.,1.]],delta=[1.,-2.])
        x=np.array([[1.,-1.],[.2,.5]])
        expected=stats.skewnorm.pdf(x[:,0],1.,scale=np.sqrt(2))*stats.skewnorm.pdf(x[:,1],-2.,scale=np.sqrt(5))
        np.testing.assert_allclose(joint_distribution(fit).pdf(x),expected,rtol=1e-12)
        c3=np.array([[1.,.2,.1],[.2,1.,-.3],[.1,-.3,1.]])
        expected3=.125+sum(np.arcsin([.2,.1,-.3]))/(4*np.pi)
        for df in [None,4.]:
            actual=np.exp(orthant_logcdf(np.zeros((1,3)),c3,df,8192,12))[0]
            self.assertAlmostEqual(actual,expected3,delta=5e-5)
            actual=np.exp(orthant_logcdf(np.zeros((1,5)),np.eye(5),df,64,12))[0]
            self.assertAlmostEqual(actual,1/32,places=12)

    def test_simulation_and_portfolio(self):
        for model in SDB_MODELS:
            fit=preset(model);dist=joint_distribution(fit)
            x=dist.rvs(100000,random_state=19)
            np.testing.assert_allclose(x.mean(0),dist.mean(),atol=.0002)
            np.testing.assert_allclose(np.cov(x.T),dist.cov(),rtol=.04,atol=1e-6)
            row,draws=simulate_joint_portfolio(fit,[.6,-.4],simulations=20000)
            again,other=simulate_joint_portfolio(fit,[.6,-.4],simulations=20000)
            np.testing.assert_array_equal(draws,other)
            self.assertEqual(row,again)
            self.assertTrue(np.isfinite(row['es_0.95_mc_se']))
            self.assertGreater(row['es_0.95'],row['var_0.95'])
            self.assertAlmostEqual(row['mean'],np.array([.6,-.4])@dist.mean())
            with self.assertRaises(ValueError):project_distribution(fit,[.6,-.4])
        fit=preset('sdb-skew-t');fit['df']=.8
        row,_=simulate_joint_portfolio(fit,[.5,.5],simulations=2000)
        self.assertTrue(np.isinf(row['es_0.95']))
        row,_=simulate_joint_portfolio(fit,[0.,0.],simulations=2000)
        self.assertEqual(row['es_0.95'],0.)
        fit['df']=1.5
        row,_=simulate_joint_portfolio(fit,[.5,.5],simulations=2000)
        self.assertTrue(np.isfinite(row['es_0.95']))
        self.assertTrue(np.isnan(row['es_0.95_mc_se']))

    def test_fitting_and_validation(self):
        for model in SDB_MODELS:
            self.assertNotIn(model,JOINT_MODELS);self.assertIn(model,ALL_JOINT_MODELS)
            x=joint_distribution(preset(model)).rvs(40,random_state=6)
            fit=fit_joint(x,model,location=0,max_iterations=1,sdb_points=64)
            self.assertEqual(fit['parameters'],5 if model.endswith('normal') else 6)
            self.assertNotEqual(fit['status'],'ok')
            np.testing.assert_allclose(fit['location'],0.,atol=1e-16)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=7)
            self.assertIn('cdf_audit_passed',fit)
            json.dumps(fit,allow_nan=False)
            for points in [63,100,32768]:
                with self.assertRaises(ValueError):fit_joint(x,model,sdb_points=points)
            fit['status']='cdf_unstable'
            with self.assertRaises(ValueError):simulate_joint_portfolio(fit,[.5,.5])
        with self.assertRaises(ValueError):fit_joint(np.random.default_rng(1).normal(size=(20,6)),'sdb-skew-t')

    def test_portfolio_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fits.json';out=Path(tmp)/'risk.csv'
            fit=preset('sdb-skew-t');fit.update(status='ok',symbols=['A','B'],return_type='simple')
            p.write_text(json.dumps([fit]))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(portfolio_main([str(p),'--weights','A=.6','B=.4','--simulations','2000','--risk-levels','.95','--output',str(out)]),0)
            self.assertIn('sdb-monte-carlo',out.read_text())

    def test_accepted_fit_and_failed_accuracy_audit(self):
        x=joint_distribution(preset('sdb-skew-normal')).rvs(60,random_state=10)
        fit=fit_joint(x,'sdb-skew-normal',sdb_points=64,max_iterations=80)
        self.assertEqual(fit['status'],'ok')
        self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=7)
        from scipy.optimize import OptimizeResult
        from return_distributions.joint_sdb import logdensity
        def fake_minimize(fun,initial,**kwargs):
            return OptimizeResult(x=initial,fun=fun(initial),success=True,nit=0,message='test seed')
        def inaccurate_density(x,mu,sigma,delta,df,points,seed):
            return logdensity(x,mu,sigma,delta,df,points,seed)+(.1 if points>64 else 0.)
        with patch('return_distributions.joint_sdb.optimize.minimize',side_effect=fake_minimize),patch('return_distributions.joint_sdb.logdensity',side_effect=inaccurate_density):
            fit=fit_joint(x,'sdb-skew-normal',sdb_points=64)
        self.assertTrue(fit['converged'])
        self.assertFalse(fit['cdf_audit_passed'])
        self.assertEqual(fit['status'],'cdf_unstable')


if __name__=='__main__':unittest.main()
