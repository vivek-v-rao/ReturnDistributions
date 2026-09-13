import unittest
import numpy as np
from scipy import stats
from return_distributions.nts import nts, cf, invert
from return_distributions.fitting import fit_one, fit_many, fitted_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copulas import CopulaJoint, sample_copula


class TestNTS(unittest.TestCase):
    def test_nig_special_case(self):
        for lam,b in [(2.,.7),(.8,-.5),(3.,0.)]:
            scale=np.sqrt(2*lam)
            ref=stats.norminvgauss(scale*np.sqrt(b*b+2*lam),b*scale,scale=scale)
            x=np.array([-8.,-2.,0.,1.,5.,12.])
            np.testing.assert_allclose(nts.logpdf(x,.5,lam,b),ref.logpdf(x),atol=1e-8)
            np.testing.assert_allclose(nts.cdf(x,.5,lam,b),ref.cdf(x),atol=1e-9)
            np.testing.assert_allclose(nts.sf(x,.5,lam,b),ref.sf(x),atol=1e-9)
            np.testing.assert_allclose(nts.stats(.5,lam,b,moments='mvsk'),ref.stats(moments='mvsk'),rtol=1e-12,atol=1e-14)
        risk=portfolio_risk(nts(.5,2,.7,loc=.001,scale=.02),.99)
        scale=2.
        reference=stats.norminvgauss(scale*np.sqrt(.7**2+4),.7*scale,loc=.001,scale=.02*scale)
        expected=portfolio_risk(reference,.99)
        self.assertAlmostEqual(risk['var'],expected['var'],places=9)
        self.assertAlmostEqual(risk['es'],expected['es'],places=9)

    def test_limits_and_scores(self):
        t=np.array([0.,.1,1.,3.]); lam=3.; b=.6
        vg=(1-(1j*b*t-.5*t*t)/lam)**(-lam)
        np.testing.assert_allclose(cf(t,1e-7,lam,b),vg,atol=2e-8)
        np.testing.assert_allclose(cf(t,1-1e-7,lam,b),np.exp(1j*b*t-.5*t*t),atol=2e-8)
        x=np.array([-3.,0.,2.]); aa=.4; ll=2.; bb=.7; e=1e-5
        _,scores=invert(x,aa,ll,bb,gradient=True)
        differences=[(invert(x,aa+e,ll,bb)-invert(x,aa-e,ll,bb))/(2*e),
                     (invert(x,aa,ll*np.exp(e),bb)-invert(x,aa,ll*np.exp(-e),bb))/(2*e),
                     (invert(x,aa,ll,bb+e)-invert(x,aa,ll,bb-e))/(2*e),
                     (invert(x+e,aa,ll,bb)-invert(x-e,aa,ll,bb))/(2*e)]
        np.testing.assert_allclose(scores,differences,rtol=1e-6,atol=1e-8)

    def test_sampling_and_quantiles(self):
        for a in [.25,.7]:
            f=nts(a,2,.5)
            p=np.array([1e-5,.05,.5,.95,1-1e-5])
            np.testing.assert_allclose(f.cdf(f.ppf(p)),p,atol=2e-9)
            np.testing.assert_allclose(f.sf(f.isf(p)),p,atol=2e-9)
            x=f.rvs(size=30000,random_state=21)
            self.assertLess(abs(x.mean()-f.mean()),6*f.std()/np.sqrt(len(x)))
            np.testing.assert_allclose(np.exp(1j*x[:,None]*np.array([.5,1.5])).mean(axis=0),cf(np.array([.5,1.5]),a,2,.5),atol=.02)
        self.assertTrue(np.isnan(nts.pdf(0,1,2,0)))
        self.assertEqual(nts.pdf(np.inf,.5,2,0),0.)
        self.assertEqual(nts.cdf(-np.inf,.5,2,0),0.)
        self.assertEqual(nts.sf(np.inf,.5,2,0),0.)
        self.assertEqual(nts.rvs(.5,2,0,size=0).size,0)

    def test_fits_reconstruction_and_counts(self):
        x=nts.rvs(.4,2,.4,size=40,random_state=11)*.01
        for name,k in [('nts-symmetric',4),('nts',5)]:
            row=fit_one(x,name,max_iterations=1)
            self.assertEqual(row['k'],k)
            self.assertEqual(row['status'],'not_converged')
            self.assertAlmostEqual(fitted_distribution(row).logpdf(x).sum(),row['loglik'],places=7)
            if k==4: self.assertEqual(row['b'],0.)
        table=fit_many(x,['nts','nts-skewed'],location=0.,max_iterations=1)
        self.assertEqual(len(table),1)
        self.assertEqual(table.k.iloc[0],4)
        self.assertAlmostEqual(table['loc'].iloc[0],0.)
        fixed=fit_one(x,'nts-symmetric',location=0.,max_iterations=1)
        self.assertEqual(fixed['k'],3)

    def test_copula_marginal_integration(self):
        row=dict(name='nts',alpha=.4,lam=2.,b=.5,loc=.001,scale=.02)
        copula=dict(model='gaussian',correlation=[[1.,0.],[0.,1.]],df=None)
        joint=CopulaJoint(dict(marginal_mode='fitted',marginals=[row,row],copula=copula))
        draws=joint.rvs(4,random_state=3)
        marginal=fitted_distribution(row)
        np.testing.assert_allclose(marginal.cdf(draws),sample_copula(copula,4,3),atol=1e-9)
        np.testing.assert_allclose(joint.logpdf(draws),marginal.logpdf(draws).sum(axis=1),atol=1e-9)


if __name__=='__main__': unittest.main()
