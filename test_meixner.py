import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.meixner import meixner as m
from return_distributions.fitting import fit_one,fit_many,fitted_distribution
from return_distributions.tail_risk import portfolio_risk


class TestMeixner(unittest.TestCase):
    def test_hyperbolic_secant_and_reflection(self):
        x=np.array([-15.,-2.,0.,.3,10.])
        np.testing.assert_allclose(m.pdf(x,.5,0,scale=np.pi),stats.hypsecant.pdf(x),rtol=1e-12)
        np.testing.assert_allclose(m.cdf(x,.5,0,scale=np.pi),stats.hypsecant.cdf(x),atol=1e-10)
        np.testing.assert_allclose(m.pdf(x,.7,1.),m.pdf(-x,.7,-1.),rtol=1e-12)
        self.assertEqual(m.pdf(np.inf,1.,0),0.)
        self.assertTrue(np.isnan(m.pdf(0,1.,np.pi)))

    def test_density_moments_and_quantiles(self):
        for delta,b in [(.2,-1.),(1.,0.),(4.,1.3),(.05,2.9)]:
            f=m(delta,b); mean,var,skew,kurt=f.stats(moments='mvsk')
            # Mode-separated quadrature is tested by CDF+SF; independent moments
            # use integration in mean/SD units on ordinary shape cases.
            np.testing.assert_allclose(f.cdf([mean-2*np.sqrt(var),mean,mean+np.sqrt(var)])+f.sf([mean-2*np.sqrt(var),mean,mean+np.sqrt(var)]),1.,atol=2e-9)
            p=np.array([.001,.05,.5,.95,.999])
            np.testing.assert_allclose(f.cdf(f.ppf(p)),p,atol=2e-9)
            np.testing.assert_allclose(f.sf(f.isf(p)),p,atol=2e-9)
            if delta>.1:
                sd=np.sqrt(var)
                for power,expected in [(0,1.),(1,0.),(2,1.),(3,skew),(4,kurt+3)]:
                    value=integrate.quad(lambda z:z**power*f.pdf(mean+sd*z)*sd,-np.inf,np.inf,epsabs=1e-8)[0]
                    self.assertAlmostEqual(value,float(expected),places=6)

    def test_es_sampling_and_fits(self):
        # Hyperbolic-secant special case independently checks lower-tail ES.
        f=m(.5,0,loc=.001,scale=.02)
        risk=portfolio_risk(f,.95)
        reference=stats.hypsecant(loc=.001,scale=.02/np.pi)
        with np.errstate(over='ignore'):
            expected=-integrate.quad(lambda x:x*reference.pdf(x),-np.inf,reference.ppf(.05))[0]/.05
        self.assertAlmostEqual(risk['es'],expected,places=8)
        sample=f.rvs(12,random_state=31)
        np.testing.assert_allclose(f.cdf(sample),np.random.RandomState(31).uniform(size=12),atol=1e-9)
        x=stats.hypsecant.rvs(size=60,random_state=2)*.01
        for name in ['meixner-symmetric','meixner']:
            row=fit_one(x,name,max_iterations=1)
            self.assertEqual(row['status'],'not_converged')
            self.assertEqual(row['k'],3 if name.endswith('symmetric') else 4)
            self.assertAlmostEqual(fitted_distribution(row).logpdf(x).sum(),row['loglik'],places=7)
        table=fit_many(x,['meixner','nef-ghs','meixner-skewed'],location=0.,max_iterations=1)
        self.assertEqual(len(table),1)
        self.assertEqual(table.name.iloc[0],'meixner-skewed')
        self.assertEqual(table.k.iloc[0],3)
        self.assertAlmostEqual(table['loc'].iloc[0],0.)


if __name__=='__main__': unittest.main()
