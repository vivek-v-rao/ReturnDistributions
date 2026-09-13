import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.skew_t import fs_skew_t, azzalini_skew_t
from return_distributions import fit_one, fitted_distribution, fit_copula, refine_joint, CopulaJoint


class SkewTTests(unittest.TestCase):
    def test_symmetric_limit_and_probabilities(self):
        x=np.linspace(-4,4,25)
        for dist in [fs_skew_t,azzalini_skew_t]:
            np.testing.assert_allclose(dist.pdf(x,7,0),stats.t.pdf(x,7),atol=1e-12)
            np.testing.assert_allclose(dist.cdf(x,7,0),stats.t.cdf(x,7),atol=1e-12)
            for skew in [-1.,1.]:
                frozen=dist(7,skew)
                mass=integrate.quad(frozen.pdf,-np.inf,np.inf)[0]
                self.assertAlmostEqual(mass,1,places=8)
                q=np.array([.001,.05,.5,.95,.999])
                np.testing.assert_allclose(frozen.cdf(frozen.ppf(q)),q,atol=1e-8)
                np.testing.assert_allclose(frozen.sf(x),dist.cdf(-x,7,-skew),atol=1e-12)

    def test_moments_and_sampling(self):
        for dist in [fs_skew_t,azzalini_skew_t]:
            f=dist(9,.6)
            sample=f.rvs(size=120000,random_state=53)
            self.assertAlmostEqual(sample.mean(),f.mean(),delta=.015)
            self.assertAlmostEqual(sample.var(),f.var(),delta=.06)
            self.assertTrue(np.isnan(dist.mean(.8,.6)))
            self.assertTrue(np.isinf(dist.var(1.5,.6)))
            self.assertTrue(np.isnan(dist.stats(2.5,.6,moments='s')))

    def test_fit_and_refinement(self):
        for name,dist in [('fernandez-steel',fs_skew_t),('azzalini',azzalini_skew_t)]:
            x=dist.rvs(4,.5,size=(100,2),random_state=76)*.02
            margins=[fit_one(x[:,j],name) for j in range(2)]
            self.assertTrue(all(m['status']=='ok' for m in margins))
            self.assertTrue(all(m['k']==4 for m in margins))
            u=np.column_stack([fitted_distribution(m).cdf(x[:,j]) for j,m in enumerate(margins)])
            seed=dict(marginal_mode='fitted',marginals=margins,copula=fit_copula(u),clipped_entries=0)
            result=refine_joint(x,seed,max_iterations=1)
            self.assertGreaterEqual(result['summary']['joint_loglik']+1e-7,CopulaJoint(seed).logpdf(x).sum())

    def test_azzalini_cdf_zero_heavy_tails(self):
        for df in [.3,1,5,500,1e12]:
            self.assertAlmostEqual(azzalini_skew_t.cdf(0,df,2),.5-np.arctan(2)/np.pi,places=8)

    def test_other_skew_refinement(self):
        for name,shapes in [('jf_skew_t',dict(a=3.,b=4.)),('nct',dict(df=6.,nc=.5))]:
            row=dict(name=name,loc=0.,scale=.02,k=4,**shapes)
            dist=fitted_distribution(row)
            x=dist.rvs(size=(50,2),random_state=58)
            u=np.column_stack([dist.cdf(x[:,j]) for j in range(2)])
            seed=dict(marginal_mode='fitted',marginals=[row.copy(),row.copy()],copula=fit_copula(u),clipped_entries=0)
            result=refine_joint(x,seed,max_iterations=1)
            self.assertGreaterEqual(result['summary']['joint_loglik']+1e-7,CopulaJoint(seed).logpdf(x).sum())


if __name__=='__main__': unittest.main()
