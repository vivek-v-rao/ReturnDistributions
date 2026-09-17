"""Test Champernowne densities, special cases, simulation, and expected shortfall.
Exercise fitting and command-line integration.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import integrate, stats
from return_distributions.champernowne import champernowne as c
from return_distributions.fitting import fit_one, fitted_distribution, specification
from return_distributions.cli import main
from return_distributions.tail_risk import portfolio_risk


class TestChampernowne(unittest.TestCase):
    def test_special_cases(self):
        x=np.array([-50.,-4.,0.,.1,3.,50.])
        for lam,reference in [(0.,stats.hypsecant),(1.,stats.logistic)]:
            np.testing.assert_allclose(c.pdf(x,lam),reference.pdf(x),rtol=1e-12)
            np.testing.assert_allclose(c.cdf(x,lam),reference.cdf(x),atol=1e-14)
            np.testing.assert_allclose(c.sf(x,lam),reference.sf(x),atol=1e-14)
            p=np.array([1e-12,.01,.5,.99,1-1e-12])
            # SciPy hypsecant's upper-tail tan expression loses precision;
            # compare using the symmetric lower-tail quantile instead.
            expected=np.where(p>.5,-reference.ppf(1-p),reference.ppf(p))
            np.testing.assert_allclose(c.ppf(p,lam),expected,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(c.stats(lam,moments='mvsk'),reference.stats(moments='mvsk'),atol=1e-12)
        self.assertEqual(specification('hyperbolic-secant')[0].name,'hypsecant')
        self.assertEqual(specification('hyperbolic-secant')[2],{})

    def test_general_shape(self):
        for lam in [-.95,-.5,0.,1-1e-10,1.,1+1e-10,3.,1000.]:
            f=c(lam)
            self.assertAlmostEqual(integrate.quad(f.pdf,-np.inf,np.inf)[0],1.,places=8)
            variance=integrate.quad(lambda x: x*x*f.pdf(x),-np.inf,np.inf)[0]
            self.assertAlmostEqual(variance/f.var(),1.,places=8)
            fourth=integrate.quad(lambda x:x**4*f.pdf(x),-np.inf,np.inf)[0]
            self.assertAlmostEqual(fourth/f.var()**2-3,float(f.stats(moments='k')),places=7)
            p=np.array([1e-12,.01,.25,.5,.8,.99,1-1e-12])
            np.testing.assert_allclose(f.cdf(f.ppf(p)),p,rtol=1e-9,atol=1e-14)
            np.testing.assert_allclose(f.sf(f.isf(p)),p,rtol=1e-9,atol=1e-14)
            self.assertTrue(np.isfinite(f.logpdf(10000)))
        self.assertTrue(np.isnan(c.pdf(0,-1)))
        self.assertEqual(c.cdf(-np.inf,0),0.)
        self.assertEqual(c.cdf(np.inf,0),1.)

    def test_simulation_and_es(self):
        f=c(-.5,loc=.001,scale=.01)
        x=f.rvs(150000,random_state=41)
        self.assertAlmostEqual(x.mean(),f.mean(),delta=.0001)
        self.assertAlmostEqual(x.var()/f.var(),1.,delta=.025)
        risk=portfolio_risk(f,.95)
        self.assertAlmostEqual(risk['es'],-x[x<=np.quantile(x,.05)].mean(),delta=.0003)
        self.assertTrue(np.isfinite(c.rvs(0.,random_state=4)))

    def test_fits_and_cli(self):
        x=c.rvs(-.3,loc=.001,scale=.01,size=250,random_state=4)
        row=fit_one(x,'champernowne',max_iterations=1000)
        self.assertEqual(row['status'],'ok')
        self.assertEqual(row['k'],3)
        self.assertAlmostEqual(fitted_distribution(row).logpdf(x).sum(),row['loglik'],places=7)
        fixed=fit_one(x,'champernowne',location=0.,max_iterations=500)
        self.assertEqual(fixed['k'],2)
        self.assertAlmostEqual(fixed['loc'],0.)
        self.assertEqual(fit_one(x,'champernowne',max_iterations=1)['status'],'not_converged')
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'returns.csv'; output=Path(directory)/'fits.csv'
            pd.DataFrame({'A':x},index=pd.date_range('2020-01-01',periods=len(x))).to_csv(source)
            with contextlib.redirect_stdout(io.StringIO()):
                status=main([str(source),'--input-type','returns','--models','champernowne','hyperbolic-secant','logistic','--output',str(output)])
            self.assertEqual(status,0)
            self.assertEqual(set(pd.read_csv(output).name),{'champernowne','hyperbolic-secant','logistic'})


if __name__ == '__main__': unittest.main()
