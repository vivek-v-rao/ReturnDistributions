import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from return_distributions.joint_cli import main
from return_distributions.multivariate import ALL_JOINT_MODELS, JOINT_MODELS


class AllModelsTests(unittest.TestCase):
    def test_all_and_default_expansion(self):
        sample=pd.DataFrame(np.random.default_rng(1).normal(size=(20,2)),
            columns=['A','B'],index=pd.bdate_range('2020-01-01',periods=20))
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'fits.json'
            for selection,expected in [(['--models','all'],ALL_JOINT_MODELS),([],JOINT_MODELS)]:
                # Check dispatch and saved canonical names without costly fits.
                with patch('return_distributions.joint_cli.read_returns',return_value=sample),patch('return_distributions.joint_cli.fit_joint',side_effect=ValueError('test stub')) as fitter,contextlib.redirect_stdout(io.StringIO()) as console:
                    self.assertEqual(main(['unused.csv','--output',str(output),*selection]),1)
                self.assertEqual([c.args[1] for c in fitter.call_args_list],list(expected))
                self.assertEqual([r['model'] for r in json.loads(output.read_text())],list(expected))
                if selection:
                    self.assertIn('Expanded models',console.getvalue())
                    self.assertIn('can be slow',console.getvalue())
                else: self.assertNotIn('Expanded models',console.getvalue())

    def test_mixed_all_rejected_before_loading(self):
        for names in [['all','normal'],['nig','all'],['all','all']]:
            with patch('return_distributions.joint_cli.read_returns') as reader,contextlib.redirect_stderr(io.StringIO()) as err,self.assertRaises(SystemExit) as raised:
                main(['unused.csv','--models',*names])
            self.assertEqual(raised.exception.code,2)
            self.assertIn('must be used alone',err.getvalue())
            reader.assert_not_called()


if __name__=='__main__': unittest.main()
