import unittest
import pandas as pd
from return_distributions.model_names import canonical_model, display_model_label
from return_distributions.table_format import aligned_table
from return_distributions.joint_cli import format_fit_comparison


class DisplayFamilyTests(unittest.TestCase):
    def test_names_and_aliases(self):
        for name in ('nig', 'gh', 'hyperbolic', 'variance-gamma', 'meixner', 'egb2'):
            self.assertEqual(display_model_label(name+'-skewed'), name)
            self.assertEqual(display_model_label(name+'-skewed [2 components]'), name+' [2 components]')
            self.assertEqual(canonical_model(name), name+'-skewed')
        for name in ('ged-skewed', 'generalized-t-skewed', 'gh-skew-t', 'nig-symmetric', 'azzalini-skew-t'):
            self.assertEqual(display_model_label(name), name)

    def test_tables_do_not_mutate_identifiers(self):
        frame = pd.DataFrame(dict(model=['nig-skewed', 'gh-skewed'], aic=[1., 2.]))
        for render in (aligned_table, format_fit_comparison):
            text = render(frame)
            self.assertNotIn('-skewed', text)
            self.assertIn('nig', text)
        self.assertEqual(frame.model.tolist(), ['nig-skewed', 'gh-skewed'])


if __name__ == '__main__': unittest.main()
