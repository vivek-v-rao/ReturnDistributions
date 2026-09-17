"""Test distribution alias resolution, deduplication, and canonical output.
Cover univariate and joint APIs, fitting, and CLI integration.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from return_distributions.model_names import MODEL_ALIASES, unique_models
from return_distributions.fitting import specification, fit_many, fit_one
from return_distributions.multivariate import fit_joint, ALL_JOINT_MODELS
from return_distributions.simulation import preset
from return_distributions.joint_cli import main


class TestModelAliases(unittest.TestCase):
    def test_univariate_laplace_alias_fit(self):
        x = np.random.default_rng(8).normal(size=100)
        alias = fit_one(x, 'laplace-skewed')
        canonical = fit_one(x, 'laplace_asymmetric')
        self.assertEqual(alias['name'], 'laplace_asymmetric')
        self.assertAlmostEqual(alias['loglik'], canonical['loglik'])
        self.assertEqual(unique_models(['laplace-skewed', 'laplace_asymmetric']),
                         ['laplace_asymmetric'])

    def test_resolution_and_api(self):
        x = np.random.default_rng(3).normal(size=(20,2))
        for alias, canonical in MODEL_ALIASES.items():
            a, names, fixed = specification(alias)
            b, expected_names, expected_fixed = specification(canonical)
            self.assertIs(a, b)
            self.assertEqual((names,fixed), (expected_names,expected_fixed))
            if canonical in ALL_JOINT_MODELS:
                self.assertEqual(preset(alias)['model'], canonical)
                target = 'fit_generalized_t' if canonical == 'generalized-t' else ('fit_nts_joint' if canonical.startswith('nts-') else 'fit_gh')
                with patch('return_distributions.multivariate.'+target, return_value={}) as fitter:
                    fit_joint(x, alias)
                    if canonical == 'generalized-t':
                        fitter.assert_called_once()
                    else:
                        self.assertEqual(fitter.call_args.args[1], canonical)
        row = fit_one(x[:,0], 'variance-gamma', max_iterations=1)
        self.assertEqual(row['name'], 'variance-gamma-skewed')

    def test_deduplication(self):
        names = ['nig', 'nig-skewed', 'nig-symmetric', 'gh', 'gh-skewed']
        expected = ['nig-skewed', 'nig-symmetric', 'gh-skewed']
        self.assertEqual(unique_models(names), expected)
        with patch('return_distributions.fitting.fit_one', side_effect=lambda x,name,**kw: dict(name=name,status='ok',aic=1)) as fitter:
            result = fit_many(np.arange(10.), names)
            self.assertEqual(fitter.call_count, 3)
            self.assertEqual(set(result.name), set(expected))

    def test_cli_canonical_output(self):
        frame = pd.DataFrame(np.random.default_rng(4).normal(size=(20,2)),
                             index=pd.date_range('2020-01-01',periods=20), columns=['A','B'])
        joint_aliases={a:c for a,c in MODEL_ALIASES.items() if c in ALL_JOINT_MODELS}
        names = [n for pair in joint_aliases.items() for n in pair]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fits.json'
            with patch('return_distributions.joint_cli.read_returns', return_value=frame), patch('return_distributions.joint_cli.fit_joint', side_effect=ValueError('test stub')) as fitter, contextlib.redirect_stdout(io.StringIO()):
                status = main(['unused.csv','--models',*names,'--output',str(output)])
            self.assertEqual(status,1)
            self.assertEqual(fitter.call_count,len(joint_aliases))
            self.assertEqual([r['model'] for r in json.loads(output.read_text())],list(joint_aliases.values()))


if __name__ == '__main__':
    unittest.main()
