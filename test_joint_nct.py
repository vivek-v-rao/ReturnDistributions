import unittest
import numpy as np
from scipy import integrate, stats
from return_distributions.multivariate import fit_joint, joint_distribution
from return_distributions.joint_nct import logdensity
from return_distributions.projection import project_distribution
from return_distributions.simulation import preset
from return_distributions.tail_risk import portfolio_risk


class JointNCTTests(unittest.TestCase):
    def test_scalar_and_central_limits(self):
        for df in (.3, 1.5, 6., 80.):
            for nc in (-7., -.5, 0., 2.):
                x = np.array([-8., -1., 0., .5, 4., 20.])
                fit = dict(model='noncentral-t', location=[.1], scatter=[[1.44]], delta=[1.2*nc], df=df)
                np.testing.assert_allclose(joint_distribution(fit).pdf(x[:, None]),
                                           stats.nct.pdf(x, df, nc, loc=.1, scale=1.2), rtol=2e-5, atol=1e-13)
        fit = preset('noncentral-t'); fit['delta'] = [0., 0.]
        x = np.array([[.01, -.02], [.02, .03]])
        np.testing.assert_allclose(joint_distribution(fit).pdf(x),
                                   stats.multivariate_t.pdf(x, loc=fit['location'], shape=fit['scatter'], df=6.), rtol=1e-12)

    def test_mixture_integral(self):
        chol = np.array([[1., 0.], [.3, .8]])
        mu = np.array([.1, -.2]); a = np.array([.8, -1.1]); delta = chol@a
        for df in (.4, 3., 20.):
            for x in (mu, np.array([-2., 1.]), np.array([3., -4.])):
                def f(s):
                    return stats.multivariate_normal.pdf(s*(x-mu), mean=delta, cov=chol@chol.T)*s**2*stats.chi.pdf(s*np.sqrt(df), df)*np.sqrt(df)
                value = integrate.quad(f, 0, np.inf, epsabs=1e-12)[0]
                self.assertAlmostEqual(float(np.exp(logdensity(x, mu, chol, a, df))[0]), value, delta=1e-10)

    def test_large_df_projection(self):
        fit = dict(model='noncentral-t', location=[0.], scatter=[[1.]], delta=[.8], df=100000.)
        p = project_distribution(fit, [1.])
        x = np.array([-2., 0., 2.])
        np.testing.assert_allclose(p.pdf(x), stats.norm.pdf(x, loc=.8), rtol=.001)
        self.assertAlmostEqual(portfolio_risk(p, .95)['es'], portfolio_risk(stats.norm(loc=.8), .95)['es'], delta=.001)

    def test_projection_moments_and_es(self):
        fit = preset('noncentral-t'); dist = joint_distribution(fit)
        x = dist.rvs(150000, random_state=93)
        np.testing.assert_allclose(x.mean(0), dist.mean(), atol=.0002)
        np.testing.assert_allclose(np.cov(x.T), dist.cov(), rtol=.04, atol=1e-6)
        for w in ([.6, .4], [-.4, 1.3]):
            p = project_distribution(fit, w); sample = x@w
            np.testing.assert_allclose(p.mean(), np.array(w)@dist.mean(), atol=1e-12)
            np.testing.assert_allclose(p.var(), np.array(w)@dist.cov()@w, rtol=1e-12)
            np.testing.assert_allclose(np.quantile(sample, [.05, .5, .95]), p.ppf([.05, .5, .95]), atol=.0005)
            self.assertAlmostEqual(portfolio_risk(p, .95)['es'], -sample[sample <= np.quantile(sample, .05)].mean(), delta=.001)
        fit['df'] = 1.5
        self.assertTrue(np.isfinite(portfolio_risk(project_distribution(fit, [.5, .5]))['es']))
        fit['df'] = .9
        self.assertTrue(np.isinf(portfolio_risk(project_distribution(fit, [.5, .5]))['es']))

    def test_fit_and_fixed_location(self):
        x = joint_distribution(preset('noncentral-t')).rvs(200, random_state=112)
        for loc in (None, 0.):
            fit = fit_joint(x, 'noncentral-t', location=loc, max_iterations=300)
            self.assertEqual(fit['parameters'], 8 if loc is None else 6)
            self.assertAlmostEqual(joint_distribution(fit).logpdf(x).sum(), fit['loglik'], places=6)
            self.assertTrue(np.linalg.eigvalsh(fit['scatter']).min() > 0)
            if loc is not None: np.testing.assert_allclose(fit['location'], 0., atol=1e-16)
        failed = fit_joint(x, 'noncentral-t', max_iterations=1)
        self.assertNotEqual(failed['status'], 'ok')
        with self.assertRaises(ValueError): project_distribution(failed, [.5, .5])


if __name__ == '__main__': unittest.main()
