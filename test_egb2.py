"""Test EGB2 densities, logistic limits, reflection, and tail calculations.
Cover moments, simulation, fitting aliases, and copula integration.
"""

import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.egb2 import egb2
from return_distributions.fitting import fit_one, fit_many, fitted_distribution
from return_distributions.tail_risk import portfolio_risk
from return_distributions.copulas import fit_copula, sample_copula


class TestEGB2(unittest.TestCase):
    def test_logistic_and_reflection(self):
        x=np.array([-100.,-3.,0.,.7,20.,100.])
        for method in ['pdf','logpdf','cdf','sf']:
            np.testing.assert_allclose(getattr(egb2,method)(x,1,1),
                                       getattr(stats.logistic,method)(x),rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(egb2.pdf(x,.4,2),egb2.pdf(-x,2,.4))
        np.testing.assert_allclose(egb2.stats(1,1,moments='mvsk'),[0,np.pi**2/3,0,1.2])
        self.assertTrue(np.isnan(egb2.pdf(0,0,1)))
        self.assertEqual(egb2.pdf(np.inf,1,1),0.)

    def test_tails_moments_quantiles(self):
        for a,b in [(.01,2),(.3,1),(5.,.5),(200.,200.)]:
            f=egb2(a,b)
            p=np.array([1e-12,.001,.05,.5,.95,.999,1-1e-12])
            np.testing.assert_allclose(f.cdf(f.ppf(p)),p,rtol=1e-9,atol=1e-14)
            np.testing.assert_allclose(f.sf(f.isf(p)),p,rtol=1e-9,atol=1e-14)
            mean,var,skew,kurt=f.stats(moments='mvsk'); sd=np.sqrt(var)
            for power,expected in [(0,1.),(1,0.),(2,1.),(3,skew),(4,kurt+3)]:
                value=integrate.quad(lambda z:z**power*f.pdf(mean+sd*z)*sd,
                                     -np.inf,np.inf,epsabs=1e-8)[0]
                self.assertAlmostEqual(value,float(expected),places=6)

    def test_es_sampling_and_copula(self):
        f=egb2(1,1,loc=.001,scale=.02)
        q=.05
        expected=-.001-.02*(q*np.log(q)+(1-q)*np.log1p(-q))/q
        self.assertAlmostEqual(portfolio_risk(f,1-q)['es'],expected,places=9)
        x=egb2.rvs(.3,1.2,size=400,random_state=12)
        np.testing.assert_allclose(egb2.cdf(x,.3,1.2),np.random.RandomState(12).uniform(size=400),atol=1e-12)
        u=np.column_stack([egb2.cdf(x,.3,1.2),np.random.default_rng(1).uniform(size=400)])
        copula=fit_copula(u)
        sample=sample_copula(copula,size=100,random_state=2)
        simulated=egb2.ppf(sample,.3,1.2)
        self.assertTrue(np.isfinite(simulated).all())
        np.testing.assert_allclose(egb2.cdf(simulated,.3,1.2),sample,atol=1e-12)

    def test_fits_and_alias(self):
        x=egb2.rvs(.5,1.5,loc=.001,scale=.01,size=500,random_state=24)
        for name,k in [('egb2-symmetric',3),('egb2',4)]:
            row=fit_one(x,name,max_iterations=500)
            self.assertTrue(row['converged'])
            self.assertEqual(row['k'],k)
            self.assertAlmostEqual(fitted_distribution(row).logpdf(x).sum(),row['loglik'],places=7)
            if k==3: self.assertEqual(row['a'],row['b'])
            limited=fit_one(x,name,max_iterations=1)
            self.assertEqual(limited['status'],'not_converged')
        table=fit_many(x,['egb2','egb2-skewed'],location=0.,max_iterations=1)
        self.assertEqual(len(table),1)
        self.assertEqual(table.k.iloc[0],3)
        self.assertAlmostEqual(table['loc'].iloc[0],0.)
        fixed=fit_one(x,'egb2-symmetric',location=0.,max_iterations=1)
        self.assertEqual(fixed['k'],2)


if __name__=='__main__': unittest.main()
