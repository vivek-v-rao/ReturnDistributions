import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from return_distributions import fit_joint, joint_distribution
from return_distributions.joint_cli import main


class JointTests(unittest.TestCase):
    def setUp(self):
        self.x = stats.multivariate_t.rvs([.01, -.02], [[.04, .015], [.015, .09]],
                                         df=5, size=300, random_state=91)

    def test_normal(self):
        fit = fit_joint(self.x, 'normal')
        np.testing.assert_allclose(fit['location'], self.x.mean(axis=0))
        np.testing.assert_allclose(fit['covariance'], np.cov(self.x.T, bias=True))
        self.assertEqual(fit['parameters'], 5)
        self.assertAlmostEqual(joint_distribution(fit).logpdf(self.x).sum(), fit['loglik'])
        fixed = fit_joint(self.x, 'normal', location=0)
        np.testing.assert_allclose(fixed['covariance'], self.x.T@self.x/len(self.x))
        self.assertEqual(fixed['parameters'], 3)

    def test_t(self):
        fit = fit_joint(self.x, 'student-t', df_starts=[5.])
        self.assertTrue(fit['converged'])
        self.assertGreater(fit['df'], 2)
        self.assertLess(fit['df'], 15)
        self.assertEqual(fit['parameters'], 6)
        self.assertAlmostEqual(joint_distribution(fit).logpdf(self.x).sum(), fit['loglik'], places=6)
        np.testing.assert_allclose(fit['covariance'], np.array(fit['scatter'])*fit['df']/(fit['df']-2))
        self.assertGreater(np.linalg.eigvalsh(fit['scatter']).min(), 0)
        self.assertAlmostEqual(fit['aic'], 12-2*fit['loglik'])
        json.dumps(fit, allow_nan=False)

    def test_invalid_and_fixed(self):
        for x in [self.x[:, 0], np.ones((30, 2)), np.column_stack([self.x[:, 0]]*2), self.x[:2], self.x*np.nan]:
            with self.assertRaises(ValueError):
                fit_joint(x)
        fit = fit_joint(self.x, 'student-t', location=0, df_starts=[5.], max_iterations=1)
        np.testing.assert_allclose(fit['location'], [0, 0], atol=1e-16)
        self.assertEqual(fit['parameters'], 4)
        self.assertFalse(fit['converged'])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'returns.csv', Path(tmp)/'joint.json'
            frame = pd.DataFrame(self.x, columns=['A', 'B'], index=pd.bdate_range('2020-01-01', periods=len(self.x)))
            frame.iloc[-3, 0] = np.nan
            frame.to_csv(source)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(source), '--input-type', 'returns', '--models', 'normal', '--days', '63', '126', '--output', str(output)])
            self.assertEqual(code, 0)
            rows = json.loads(output.read_text())
            self.assertEqual([r['observations'] for r in rows], [63, 126])
            self.assertEqual(rows[0]['symbols'], ['A', 'B'])
            self.assertTrue(output.with_suffix('.csv').exists())


if __name__ == '__main__':
    unittest.main()
