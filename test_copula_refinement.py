import unittest
import copy
import numpy as np
from scipy import stats
from return_distributions import fit_one, fitted_distribution, fit_copula, CopulaJoint, refine_joint


class RefinementTests(unittest.TestCase):
    def seed(self, x, family='normal', copula='gaussian'):
        marginals = [fit_one(x[:,j], family) for j in range(x.shape[1])]
        u = np.column_stack([fitted_distribution(row).cdf(x[:,j]) for j,row in enumerate(marginals)])
        fit = fit_copula(u, copula)
        return dict(marginal_mode='fitted', marginals=marginals, copula=fit, clipped_entries=0)

    def test_normal_mle(self):
        x = np.random.default_rng(47).multivariate_normal([.001,-.002], [[.0004,.00015],[.00015,.0003]], size=150)
        seed = self.seed(x)
        frozen = copy.deepcopy(seed)
        result = refine_joint(x, seed)
        # Two-stage normal estimates already attain joint normal MLE.
        expected = stats.multivariate_normal.logpdf(x, mean=x.mean(axis=0), cov=np.cov(x.T,bias=True)).sum()
        self.assertAlmostEqual(result['summary']['joint_loglik'], expected, places=5)
        self.assertAlmostEqual(CopulaJoint(result).logpdf(x).sum(), expected, places=5)
        self.assertEqual(seed, frozen)

    def test_t_refinement_and_failed_fallback(self):
        rng = np.random.default_rng(14)
        x = stats.multivariate_t.rvs([0,0], [[1,.7],[.7,1]],df=5,size=130,random_state=rng)*.02
        seed = self.seed(x,'student-t','student-t')
        baseline = CopulaJoint(seed).logpdf(x).sum()
        result = refine_joint(x, seed)
        self.assertGreaterEqual(result['summary']['joint_loglik']+1e-7, baseline)
        self.assertTrue(result['refinement']['accepted'])
        self.assertAlmostEqual(CopulaJoint(result).logpdf(x).sum(), result['summary']['joint_loglik'], places=6)
        self.assertEqual(result['summary']['total_parameters'], 8)
        limited = refine_joint(x,seed,max_iterations=1)
        self.assertFalse(limited['refinement']['accepted'])
        self.assertEqual(limited['marginals'],seed['marginals'])

    def test_reject_rank_and_clip(self):
        with self.assertRaises(ValueError): refine_joint(np.ones((10,2)),dict(marginal_mode='ranks'))
        seed = self.seed(np.random.default_rng(6).normal(size=(100,2)))
        seed['clipped_entries']=1
        with self.assertRaises(ValueError): refine_joint(np.ones((10,2)),seed)

    def test_other_marginal_constraints(self):
        x = stats.t.rvs(6,size=(90,2),random_state=62)*.02
        for family in ['laplace','ged','hyperbolic-symmetric','hyperbolic-skewed','nig-symmetric','nig-skewed']:
            seed = self.seed(x,family)
            result = refine_joint(x,seed,max_iterations=2)
            self.assertGreaterEqual(result['summary']['joint_loglik']+1e-7,CopulaJoint(seed).logpdf(x).sum())
            for row in result['marginals']:
                if family.startswith('hyperbolic'): self.assertEqual(row['p'],1)
                if family.endswith('symmetric'): self.assertEqual(row['b'],0)
                self.assertGreater(row['scale'],0)


if __name__ == '__main__':
    unittest.main()
