"""Test joint slash distributions, limiting cases, and portfolio tail risk.
Cover quantiles, fitting, and exclusion of failed numerical accuracy audits.
"""

import unittest
from unittest.mock import patch
import numpy as np
from scipy import integrate,stats
from return_distributions import fit_joint,joint_distribution,project_distribution,portfolio_risk
from return_distributions.joint_slash import SLASH_MODELS,ProjectedSlash
from return_distributions.multivariate import JOINT_MODELS,ALL_JOINT_MODELS
from return_distributions.simulation import preset


class SlashTests(unittest.TestCase):
    def test_density_and_location(self):
        for model in SLASH_MODELS:
            self.assertNotIn(model,JOINT_MODELS);self.assertIn(model,ALL_JOINT_MODELS)
            fit=preset(model,1);fit['quadrature_points']=512
            dist=joint_distribution(fit);p=project_distribution(fit,np.array([1.]))
            x=np.array([-.01,0.,.01])
            np.testing.assert_allclose(dist.pdf(x[:,None]),p.pdf(x),rtol=2e-5)
            # Correct CDF mixture and normalized endpoints, not the printed CDF.
            self.assertEqual(p.cdf(-np.inf),0.);self.assertEqual(p.cdf(np.inf),1.)
            np.testing.assert_allclose(p.cdf(x)+p.sf(x),1.,atol=1e-9)
            if model=='slash-normal':
                area=integrate.quad(lambda v:float(p.pdf(v)),fit['location'][0]-.2,fit['location'][0]+.2,epsabs=1e-6)[0]
                self.assertGreater(area,.999)
            shifted=dict(fit,location=(np.array(fit['location'])+.5).tolist())
            np.testing.assert_allclose(joint_distribution(shifted).pdf((x+.5)[:,None]),dist.pdf(x[:,None]),rtol=1e-10)

    def test_limits_and_moments(self):
        for model in SLASH_MODELS:
            fit=preset(model);dist=joint_distribution(fit)
            x=dist.rvs(100000,random_state=73)
            np.testing.assert_allclose(x.mean(0),dist.mean(),atol=.0002)
            np.testing.assert_allclose(np.cov(x.T),dist.cov(),rtol=.04,atol=1e-6)
            p=project_distribution(fit,np.array([.6,-.4]))
            self.assertAlmostEqual(p.mean(),np.array([.6,-.4])@dist.mean(),places=10)
            self.assertAlmostEqual(p.var(),np.array([.6,-.4])@dist.cov()@np.array([.6,-.4]),places=10)
        for base in [stats.norm(),stats.t(5)]:
            p=ProjectedSlash(.1,.7,base,np.inf,None if base.dist.name=='norm' else 5.)
            np.testing.assert_allclose(p.pdf([-.2,.1,.5]),base.pdf((np.array([-.2,.1,.5])-.1)/.7)/.7,rtol=1e-12)
            ref=stats.norm(loc=.1,scale=.7) if base.dist.name=='norm' else stats.t(5,loc=.1,scale=.7)
            self.assertAlmostEqual(portfolio_risk(p,.95)['es'],portfolio_risk(ref,.95)['es'],places=7)
        p=ProjectedSlash(0.,1.,stats.norm(),10000.,None)
        np.testing.assert_allclose(p.pdf([-2.,0.,2.]),stats.norm.pdf([-2.,0.,2.]),rtol=.001)
        fit=dict(model='slash-normal',location=[0.],scatter=[[1.]],q=10000.)
        np.testing.assert_allclose(joint_distribution(fit).pdf(np.array([-2.,0.,2.])[:,None]),p.pdf([-2.,0.,2.]),rtol=1e-9)

    def test_quantiles_and_es(self):
        for model in SLASH_MODELS:
            fit=preset(model);w=np.array([.6,-.4]);p=project_distribution(fit,w)
            sample=joint_distribution(fit).rvs(150000,random_state=19)@w
            qs=p.ppf([.01,.5,.99])
            np.testing.assert_allclose(p.cdf(qs),[.01,.5,.99],atol=1e-8)
            np.testing.assert_allclose(qs,np.quantile(sample,[.01,.5,.99]),atol=.0007)
            for confidence in [.1,.95]:
                threshold=np.quantile(sample,1-confidence)
                expected=-sample[sample<=threshold].mean()
                self.assertAlmostEqual(portfolio_risk(p,confidence)['es'],expected,delta=.0005)
        for q,df in [(.8,5.),(5.,.8),(1.5,5.),(5.,1.5)]:
            p=ProjectedSlash(0.,1.,stats.t(df),q,df)
            risk=portfolio_risk(p,.95)
            self.assertTrue(np.isnan(p.var()))
            self.assertEqual(np.isfinite(risk['es']),q>1 and df>1)

    def test_fitting(self):
        for model in SLASH_MODELS:
            x=joint_distribution(preset(model)).rvs(40,random_state=22)
            fit=fit_joint(x,model,location=0,max_iterations=1,slash_points=16)
            self.assertEqual(fit['parameters'],3+int(model.startswith('skew-'))*2+1+int(model.endswith('-t')))
            self.assertNotEqual(fit['status'],'ok')
            np.testing.assert_allclose(fit['location'],0.,atol=1e-16)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=6)
            self.assertIn('quadrature_audit_passed',fit)
        with self.assertRaises(ValueError):fit_joint(x,'slash-t',slash_points=0)

    def test_accuracy_failure_is_not_rankable(self):
        from scipy.optimize import OptimizeResult
        from return_distributions.joint_slash import logdensity
        x=joint_distribution(preset('slash-normal')).rvs(30,random_state=9)
        def fake_minimize(fun,initial,**kwargs):
            return OptimizeResult(x=initial,fun=fun(initial),success=True,message='test seed')
        def inaccurate(x,mu,chol,a,df,q,points):
            return logdensity(x,mu,chol,a,df,q,points)+(.1 if points>16 else 0.)
        with patch('return_distributions.joint_slash.optimize.minimize',side_effect=fake_minimize),patch('return_distributions.joint_slash.logdensity',side_effect=inaccurate):
            fit=fit_joint(x,'slash-normal',slash_points=16)
        self.assertTrue(fit['converged'])
        self.assertEqual(fit['status'],'quadrature_unstable')
        with self.assertRaises(ValueError):project_distribution(fit,[.5,.5])


if __name__=='__main__':unittest.main()
