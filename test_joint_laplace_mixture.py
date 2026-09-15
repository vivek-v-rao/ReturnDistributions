import json
import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.joint_laplace_mixture import LAPLACE_MIXTURE_MODELS, JointLaplaceMixture
from return_distributions.multivariate import fit_joint, joint_distribution, JOINT_MODELS, ALL_JOINT_MODELS
from return_distributions.projection import project_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.simulation import preset


class LaplaceMixtureTests(unittest.TestCase):
    def truth(self):
        return dict(model='asymmetric-laplace',location=[.001,-.002],gamma=[-.003,.002],
            scatter=[[.0001,.00002],[.00002,.0002]],vg_shape=1.,status='ok',return_type='simple')

    def test_density_and_singularity(self):
        fit=self.truth(); joint=JointLaplaceMixture(fit); point=np.array([.01,.003])
        expected=integrate.quad(lambda w: np.exp(-w)*stats.multivariate_normal.pdf(
            point,mean=joint.location+w*joint.gamma,cov=w*joint.scatter),0,np.inf,epsabs=1e-7)[0]
        self.assertAlmostEqual(joint.pdf(point)/expected,1.,places=8)
        self.assertTrue(np.isposinf(joint.logpdf(joint.location)))
        one=dict(model='asymmetric-laplace',location=[.002],gamma=[-.003],scatter=[[.0001]],status='ok')
        x=np.array([-.04,.002,.03])
        np.testing.assert_allclose(JointLaplaceMixture(one).pdf(x[:,None]),project_distribution(one,[1.]).pdf(x),rtol=1e-12)

    def test_moments_projection_and_risk(self):
        fit=self.truth(); joint=joint_distribution(fit)
        draws=joint.rvs(150000,random_state=24)
        np.testing.assert_allclose(draws.mean(axis=0),joint.mean(),atol=.00015)
        np.testing.assert_allclose(np.cov(draws.T),joint.cov(),atol=4e-6)
        for w in [np.array([.6,.4]),np.array([1.,-.5])]:
            projected=project_distribution(fit,w)
            self.assertEqual(projected.dist.name,'laplace_asymmetric')
            self.assertAlmostEqual(projected.mean(),w@joint.mean())
            self.assertAlmostEqual(projected.var(),w@joint.cov()@w)
            returns=draws@w; q=np.quantile(returns,.01)
            risk=portfolio_risk(projected,.99)
            self.assertAlmostEqual(risk['var'],-q,delta=.0006)
            self.assertAlmostEqual(risk['es'],-returns[returns<=q].mean(),delta=.0008)
        self.assertEqual(portfolio_risk(project_distribution(fit,[0.,0.]),.99)['es'],0.)
        symmetric=dict(fit,gamma=[0.,0.])
        p=project_distribution(symmetric,[.6,.4])
        self.assertAlmostEqual(p.args[0],1.)

    def test_fits_restrictions_and_roundtrip(self):
        x=JointLaplaceMixture(self.truth()).rvs(300,random_state=12)
        for model,count in [('asymmetric-laplace',5),('laplace-mixture-symmetric',3)]:
            fit=fit_joint(x,model,max_iterations=300)
            self.assertEqual(fit['status'],'ok')
            self.assertEqual(fit['parameters'],count)
            self.assertEqual(fit['location'],[0.,0.])
            self.assertEqual(fit['location_source'],'default-zero')
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=6)
            json.dumps(fit,allow_nan=False)
            if count==3: self.assertEqual(fit['gamma'],[0.,0.])
            limited=fit_joint(x,model,location=[.001,-.002],max_iterations=1)
            self.assertEqual(limited['status'],'not_converged')
            self.assertEqual(limited['location_source'],'provided')
            self.assertEqual(limited['parameters'],count)
        with self.assertRaisesRegex(ValueError,'singular'):
            fit_joint(np.vstack([x,[0.,0.]]),'asymmetric-laplace')
        with self.assertRaisesRegex(ValueError,'singular'):
            fit_joint(x,'asymmetric-laplace',location=x[0])

    def test_registration_and_presets(self):
        for model in LAPLACE_MIXTURE_MODELS:
            self.assertIn(model,ALL_JOINT_MODELS)
            self.assertNotIn(model,JOINT_MODELS)
            truth=preset(model)
            joint=joint_distribution(truth)
            np.testing.assert_allclose(joint.mean(),truth['mean'])
            np.testing.assert_allclose(joint.cov(),truth['covariance'])


if __name__=='__main__': unittest.main()
