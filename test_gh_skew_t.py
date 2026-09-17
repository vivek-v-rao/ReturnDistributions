"""Test GH skew-t symmetric limits, mixture densities, and moment thresholds.
Cover univariate/joint fitting, portfolio projection, and expected shortfall.
"""

import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.gh_skew_t import gh_skew_t, JointGHSkewT
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.fitting import fit_one, fitted_distribution
from return_distributions.simulation import preset
from return_distributions.projection import project_distribution
from return_distributions.tail_risk import portfolio_risk


class TestGHSkewT(unittest.TestCase):
    def test_symmetric_limit_and_density(self):
        x=np.array([-3.,-.2,0.,.5,4.])
        for nu in (.5,3.,10.,200.):
            np.testing.assert_allclose(gh_skew_t.logpdf(x,nu,0),stats.t.logpdf(x,nu),atol=1e-12)
            np.testing.assert_allclose(gh_skew_t.logpdf(x,nu,1e-12),stats.t.logpdf(x,nu),atol=1e-8)
        f=gh_skew_t(7.,-.4)
        self.assertAlmostEqual(integrate.quad(f.pdf,-np.inf,np.inf)[0],1.,places=7)
        np.testing.assert_allclose(f.pdf(x),gh_skew_t.pdf(-x,7.,.4),rtol=1e-12)
        self.assertEqual(f.pdf(np.inf),0.)

    def test_joint_mixture_and_projection(self):
        truth=preset('gh-skew-t',2); joint=JointGHSkewT(truth)
        point=np.array([.01,-.005])
        nu=joint.df
        expected=integrate.quad(lambda w: stats.invgamma.pdf(w,nu/2,scale=nu/2)*stats.multivariate_normal.pdf(point,mean=joint.location+w*joint.gamma,cov=w*joint.scatter),0,np.inf)[0]
        self.assertAlmostEqual(joint.pdf(point)/expected,1.,places=8)
        x=joint.rvs(200000,random_state=12)
        np.testing.assert_allclose(x.mean(axis=0),joint.mean(),atol=.00012)
        np.testing.assert_allclose(np.cov(x.T),joint.cov(),rtol=.03,atol=1e-6)
        w=np.array([.6,-.4]); projected=project_distribution(truth,w)
        self.assertAlmostEqual(projected.mean(),w@joint.mean())
        self.assertAlmostEqual(projected.var(),w@joint.cov()@w)
        self.assertAlmostEqual(projected.ppf(.05),np.quantile(x@w,.05),delta=.00025)
        risk=portfolio_risk(projected,.95)
        sample=x@w
        self.assertAlmostEqual(risk['es'],-sample[sample<=np.quantile(sample,.05)].mean(),delta=.0003)

    def test_tails_and_moments(self):
        for nu,b in [(1.5,.4),(1.5,-.4),(3.,-.4),(7.,0.)]:
            f=gh_skew_t(nu,b)
            np.testing.assert_allclose(f.cdf([-2,0,2])+f.sf([-2,0,2]),1.,atol=1e-8)
            np.testing.assert_allclose(f.cdf(f.ppf([.01,.5,.99])),[.01,.5,.99],atol=1e-8)
        self.assertTrue(np.isfinite(portfolio_risk(gh_skew_t(1.5,.4),.95)['es']))
        self.assertTrue(np.isinf(portfolio_risk(gh_skew_t(1.5,-.4),.95)['es']))
        self.assertTrue(np.isfinite(portfolio_risk(gh_skew_t(3.,-.4),.95)['es']))
        self.assertTrue(np.isnan(gh_skew_t.mean(1.5,.4)))
        self.assertTrue(np.isinf(gh_skew_t.var(3.,.4)))
        self.assertTrue(np.isfinite(gh_skew_t.var(3.,0.)))
        self.assertTrue(np.isnan(gh_skew_t.stats(5.,.4,moments='s')))
        self.assertTrue(np.isnan(gh_skew_t.stats(7.,.4,moments='k')))

    def test_fit_roundtrip(self):
        x=JointGHSkewT(preset('gh-skew-t',2)).rvs(35,random_state=5)
        fit=fit_joint(x,'gh-skew-t',location=0.,max_iterations=1)
        self.assertEqual(fit['status'],'not_converged')
        self.assertEqual(fit['parameters'],6)
        np.testing.assert_allclose(fit['location'],0.,atol=1e-15)
        self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(),fit['loglik'],places=8)
        row=fit_one(x[:,0],'gh-skew-t',max_iterations=1)
        self.assertEqual(row['k'],4)
        self.assertAlmostEqual(fitted_distribution(row).logpdf(x[:,0]).sum(),row['loglik'],places=8)


if __name__ == '__main__': unittest.main()
