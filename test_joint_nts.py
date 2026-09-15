import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats,optimize

from return_distributions.joint_nts import JointNTS,mixing_rule
from return_distributions.nts import nts
from return_distributions.multivariate import fit_joint,joint_distribution,ALL_JOINT_MODELS
from return_distributions.projection import project_distribution,PointMass
from return_distributions.tail_risk import portfolio_risk
from return_distributions.simulation import preset
from return_distributions.joint_cli import main


def truth(d=2,alpha=.6,lam=2.):
    return dict(model='nts-skewed',status='ok',location=np.zeros(d).tolist(),
                scatter=np.eye(d).tolist(),gamma=np.linspace(-.3,.2,d).tolist(),alpha=alpha,lam=lam)


class JointNTSTests(unittest.TestCase):
    def test_failed_quadrature_audit_is_unranked(self):
        from return_distributions.joint_nts import mixing_rule as original_rule
        def altered_rule(alpha,lam,points=256,tail=1e-12):
            w,lw,mass = original_rule(alpha,lam,points,tail)
            return w,lw+(.002 if tail<1e-12 else 0.),mass
        def optimizer(fun,initial,**kwargs):
            return optimize.OptimizeResult(x=initial,fun=fun(initial),jac=np.zeros(len(initial)),
                                           success=True,message='test fixture',nit=0)
        x = np.random.default_rng(9).normal(size=(20,2))
        with patch('return_distributions.joint_nts.mixing_rule',side_effect=altered_rule), \
             patch('return_distributions.joint_nts.optimize.minimize',side_effect=optimizer):
            fit = fit_joint(x,'nts-skewed')
        self.assertEqual(fit['parameters'],9)
        self.assertEqual(fit['status'],'quadrature_failed')
        self.assertFalse(fit['nts_audit_passed'])
        with self.assertRaises(ValueError): project_distribution(fit,[.6,.4])

    def test_mixing_and_univariate_density(self):
        for alpha in [.25,.5,.75]:
            for lam in [.5,2.,10.]:
                w,lw,mass = mixing_rule(alpha,lam)
                weights = np.exp(lw)
                self.assertAlmostEqual(mass,1.,places=7)
                self.assertAlmostEqual(weights@w,1.,places=7)
                self.assertAlmostEqual(weights@(w-1)**2,(1-alpha)/lam,places=7)
            f = JointNTS(truth(1,alpha))
            x = np.array([-4.,-1.,0.,1.,4.])
            np.testing.assert_allclose(f.logpdf(x[:,None]),nts.logpdf(x,alpha,2.,-.3),atol=2e-7)

    def test_nig_special_case_and_portfolio(self):
        f = truth(3,.5,2.)
        f['scatter'] = [[1.,.3,-.1],[.3,.8,.2],[-.1,.2,1.2]]
        dist = JointNTS(f)
        weights = np.array([.6,-.4,.3])
        projected = project_distribution(f,weights)
        scale = np.sqrt(weights@dist.scatter@weights)
        beta = weights@dist.gamma/scale
        delta = np.sqrt(2*dist.lam)
        reference = stats.norminvgauss(np.sqrt(beta*beta+2*dist.lam)*delta,beta*delta,
                                      loc=weights@dist.location,scale=scale*delta)
        grid = np.linspace(-3,3,9)
        np.testing.assert_allclose(projected.logpdf(grid),reference.logpdf(grid),atol=1e-7)
        self.assertAlmostEqual(portfolio_risk(projected,.95)['es'],portfolio_risk(reference,.95)['es'],places=6)
        # Joint GH representation: W'=W/(2*lam), chi'=1, psi'=(2*lam)^2.
        from return_distributions.joint_gh import JointGH
        gh = JointGH(dict(location=f['location'],scatter=(dist.scatter*4).tolist(),gamma=(dist.gamma*4).tolist(),
                          **{'lambda':-.5,'psi':16.,'chi':1.}))
        x = np.random.default_rng(1).normal(size=(12,3))
        np.testing.assert_allclose(dist.logpdf(x),gh.logpdf(x),atol=1e-7)
        self.assertIsInstance(project_distribution(f,[0.,0.,0.]),PointMass)
        with self.assertRaises(ValueError): project_distribution(dict(f,status='quadrature_failed'),weights)

    def test_sampling_permutation_and_projection_cf(self):
        fit = preset('nts',dimensions=3)
        dist = joint_distribution(fit)
        draws = dist.rvs(80000,random_state=31)
        np.testing.assert_allclose(draws.mean(axis=0),dist.mean(),atol=.0002)
        np.testing.assert_allclose(np.cov(draws.T),dist.cov(),rtol=.05,atol=2e-6)
        weights = np.array([.6,-.2,.5])
        projected = project_distribution(fit,weights)
        self.assertAlmostEqual(projected.mean(),weights@dist.mean(),places=12)
        self.assertAlmostEqual(projected.var(),weights@dist.cov()@weights,places=12)
        from return_distributions.nts import cf
        scale = np.sqrt(weights@dist.scatter@weights)
        for t in [0.,1.,20.]:
            expected = np.exp(1j*t*(weights@dist.location))*cf(t*scale,dist.alpha,dist.lam,(weights@dist.gamma)/scale)
            self.assertAlmostEqual(abs(dist.cf(t*weights)-expected),0.,places=12)
        # The characteristic function also checks the normal and VG limits
        # without forcing numerical mixing integration at degenerate boundaries.
        t = np.array([2.,-1.,3.])
        normal_limit = JointNTS(dict(fit,alpha=1-1e-7))
        expected = np.exp(1j*(t@dist.mean())-.5*t@dist.scatter@t)
        self.assertAlmostEqual(abs(normal_limit.cf(t)-expected),0.,places=8)
        vg_limit = JointNTS(dict(fit,alpha=1e-7))
        z = 1j*(t@dist.gamma)-.5*t@dist.scatter@t
        expected = np.exp(1j*(t@dist.location)-dist.lam*np.log1p(-z/dist.lam))
        self.assertAlmostEqual(abs(vg_limit.cf(t)-expected),0.,places=8)
        order = [2,0,1]
        permuted = dict(fit,location=dist.location[order].tolist(),gamma=dist.gamma[order].tolist(),
                        scatter=dist.scatter[np.ix_(order,order)].tolist())
        np.testing.assert_allclose(dist.logpdf(draws[:20]),JointNTS(permuted).logpdf(draws[:20,order]),atol=1e-10)

    def test_fit_roundtrip_cli_and_limits(self):
        x = JointNTS(truth()).rvs(40,random_state=6)*.01
        fit = fit_joint(x,'nts-symmetric',location=0.,max_iterations=1)
        self.assertEqual(fit['parameters'],5)
        self.assertEqual(fit['status'],'not_converged')
        self.assertTrue(fit['nts_audit_passed'])
        np.testing.assert_allclose(fit['gamma'],0.)
        restored = joint_distribution(json.loads(json.dumps(fit,allow_nan=False)))
        self.assertAlmostEqual(restored.logpdf(x).sum(),fit['loglik'],places=7)
        self.assertIn('nts-skewed',ALL_JOINT_MODELS)
        sample = pd.DataFrame(x,columns=['A','B'],index=pd.bdate_range('2020-01-01',periods=len(x)))
        fixture = dict(fit,status='ok')
        with patch('return_distributions.joint_cli.read_returns',return_value=sample), \
             patch('return_distributions.joint_cli.fit_joint',return_value=fixture), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['unused.csv','--models','nts-symmetric','--no-save']),0)
        self.assertIn('NTS quadrature audit',output.getvalue())
        with self.assertRaises(ValueError): fit_joint(x,'nts',nts_points=32)


if __name__ == '__main__': unittest.main()
