import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from return_distributions.copulas import copula_logpdf, fit_copula, sample_copula, CopulaJoint
from return_distributions.fitting import fit_one
from return_distributions.copula_cli import main


class CopulaTests(unittest.TestCase):
    def test_independence_and_uniforms(self):
        u = np.random.default_rng(8).uniform(.001, .999, (200, 3))
        np.testing.assert_allclose(copula_logpdf(u, 'gaussian', np.eye(3)), 0, atol=1e-12)
        fitted = fit_copula(u, 'gaussian')
        self.assertEqual(fitted['parameters'], 3)
        self.assertTrue(fitted['converged'])
        for name in ['gaussian', 'student-t']:
            fit = dict(model=name, correlation=[[1, .5], [.5, 1]], df=5.)
            draws = sample_copula(fit, 20000, 29)
            np.testing.assert_allclose(draws.mean(axis=0), .5, atol=.01)
            np.testing.assert_allclose(draws.var(axis=0), 1/12, atol=.005)

    def test_fitting(self):
        u = sample_copula(dict(model='student-t', correlation=[[1,.6],[.6,1]], df=4.), 400, 71)
        for name in ['gaussian', 'student-t']:
            fit = fit_copula(u, name)
            self.assertTrue(fit['converged'])
            self.assertEqual(fit['parameters'], 1 if name == 'gaussian' else 2)
            self.assertAlmostEqual(fit['loglik'], copula_logpdf(u, name, fit['correlation'], fit['df']).sum())
            self.assertAlmostEqual(fit['correlation'][0][1], .6, delta=.15)
            np.testing.assert_allclose(np.diag(fit['correlation']), 1.)

    def test_joint_normal_identity(self):
        rng = np.random.default_rng(7)
        margins = [fit_one(rng.normal(size=200), 'normal') for _ in range(2)]
        corr = np.array([[1., .3],[.3,1.]])
        record = dict(marginal_mode='fitted', marginals=margins, copula=dict(model='gaussian', correlation=corr.tolist(), df=None))
        model = CopulaJoint(record)
        x = rng.normal(size=(30,2))
        scales = np.array([m['scale'] for m in margins])
        expected = stats.multivariate_normal.logpdf(x, mean=[m['loc'] for m in margins], cov=corr*np.outer(scales,scales))
        np.testing.assert_allclose(model.logpdf(x), expected, atol=1e-10)
        self.assertEqual(model.rvs(5, 4).shape, (5,2))

    def test_cli_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'returns.csv'
            pd.DataFrame(np.random.default_rng(7).normal(size=(90,2)), columns=['A','B'], index=pd.bdate_range('2020-01-01',periods=90)).to_csv(source)
            for mode in ['fitted', 'ranks']:
                output = Path(tmp)/(mode+'.json')
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main([str(source), '--input-type','returns','--marginal-mode',mode,'--marginal-models','normal','--copulas','gaussian','--output',str(output)])
                self.assertEqual(code, 0)
                record = json.loads(output.read_text())[0]
                self.assertEqual('joint_loglik' in record['summary'], mode == 'fitted')

    def test_invalid(self):
        with self.assertRaises(ValueError): fit_copula(np.zeros((10,2)))
        u = np.tile(np.linspace(.1,.9,20)[:,None], (1,2))
        with self.assertRaises(ValueError): fit_copula(u)


if __name__ == '__main__':
    unittest.main()
