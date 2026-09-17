"""Test mixture component weights, means, standard deviations, and correlations.
Cover undefined Student-t moments and the distinction between NIG mean and location.
"""

import contextlib
import io
import unittest

from return_distributions.joint_cli import print_mixture_components


class ComponentOutputTests(unittest.TestCase):
    def output(self, model='normal', df=None, **extra):
        record = dict(model=model, df=df, location=[.01, -.02], scatter=[[.0001, .0001], [.0001, .0004]], **extra)
        fit = dict(model=model, df=df, symbols=['A', 'B'], status='ok',
                   mixture_weights=[.25, .75], component_fits=[record, record], **extra)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_mixture_components(fit)
        return stream.getvalue()

    def test_normal_weights_and_moments(self):
        out = self.output()
        self.assertIn('weight: 0.250000 (25.00%)', out)
        self.assertIn('weight: 0.750000 (75.00%)', out)
        self.assertLess(out.index('weight: 0.750000'), out.index('weight: 0.250000'))
        self.assertIn('0.020000', out)
        self.assertEqual(out.count('Component return correlation:'), 2)
        self.assertIn('0.500', out)

    def test_student_t_actual_sd_and_undefined_moments(self):
        self.assertIn('0.014142', self.output('student-t', 4.))
        out = self.output('student-t', .8)
        self.assertIn('mean undefined', out)
        self.assertIn('correlation unavailable', out)
        self.assertNotIn('Component return correlation:', out)

    def test_nig_mean_is_not_location(self):
        out = self.output('nig-skewed', psi=4., chi=1., **{'lambda': -.5}, gamma=[.02, 0.])
        # E[W] = 1/sqrt(psi) = .5, so the first mean is .01 + .5*.02.
        self.assertIn('0.020000', out)
        self.assertIn('Component return correlation:', out)


if __name__ == '__main__':
    unittest.main()
