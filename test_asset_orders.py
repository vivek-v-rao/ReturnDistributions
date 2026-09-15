import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from return_distributions.joint_cli import main, fitting_tasks, select_best_orders
from return_distributions.joint_risk_report import risk_report


class AssetOrderTests(unittest.TestCase):
    def test_best_selection_success_ties_and_no_success(self):
        def record(order, ll, status='ok'):
            return dict(model='generalized-t-skewed', asset_order=order, loglik=ll, status=status)
        for first, second, winner in [(1.,2.,1),(2.,1.,0),(2.,2.,0)]:
            rows = [record('A B',first), record('B A',second), record('bad',100.,'timeout'),
                    dict(model='normal', status='failed')]
            with contextlib.redirect_stdout(io.StringIO()): select_best_orders(rows)
            self.assertTrue(rows[winner]['selected_for_display'])
            self.assertFalse(rows[1-winner]['selected_for_display'])
            self.assertFalse(rows[2]['selected_for_display'])
            self.assertTrue(rows[3]['selected_for_display'])
        rows = [record('A B',None,'failed'),record('B A',np.nan)]
        with contextlib.redirect_stdout(io.StringIO()) as output: select_best_orders(rows)
        self.assertTrue(all(not r['selected_for_display'] for r in rows))
        self.assertIn('none', output.getvalue())

    def test_best_console_filters_but_saved_audit_is_complete(self):
        sample = pd.DataFrame(np.random.default_rng(4).normal(size=(30,2))*.01,
            columns=['A','B'], index=pd.bdate_range('2020-01-01',periods=30))
        normal = dict(model='normal', status='ok', dimensions=2, observations=30, parameters=5,
            loglik=10., aic=-10., bic=-3., location=[0.,0.], scatter=(np.eye(2)*.0001).tolist(),
            covariance=(np.eye(2)*.0001).tolist(), correlation=np.eye(2).tolist())
        first = dict(normal, model='generalized-t-skewed', parameters=9, power=2.,q=4.,skewness=[.1,.2],loglik=11.,aic=-4.)
        second = dict(first, loglik=14.,aic=-10.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('return_distributions.joint_cli.read_returns',return_value=sample), \
                 patch('return_distributions.joint_cli.fit_joint',side_effect=[normal,first,second]) as fitter, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['unused.csv','--models','normal','generalized-t-skewed',
                    '--asset-orders','all','--best-asset-order','--js-distance','--weights','A=.6','B=.4',
                    '--simulations','2000','--output',str(root/'fits.json')]),0)
            self.assertEqual(fitter.call_count,3)
            text = output.getvalue()
            fit_table = text.split('Joint fit comparison:')[1].split('Generalized t:')[0]
            self.assertIn('B A',fit_table)
            self.assertNotIn('A B',fit_table)
            js_table = text.split('Jensen-Shannon distance (base 2; range 0--1):')[1].split('Numerical uncertainty')[0]
            self.assertIn('order: B A',js_table)
            self.assertNotIn('order: A B',js_table)
            risk_table = text.split('Positive-loss percentages per input return period (one day for daily data).')[1].split('simulated risk estimates')[0]
            self.assertNotIn(' A B ',risk_table)
            records = json.loads((root/'fits.json').read_text())
            self.assertEqual([f['selected_for_display'] for f in records],[True,False,True])
            summary = pd.read_csv(root/'fits.csv')
            self.assertEqual(len(summary),3)
            self.assertEqual(summary.selected_for_display.tolist(),[True,False,True])
            risk = pd.read_csv(root/'fits_portfolio_risk.csv')
            self.assertEqual(risk.selected_for_display.tolist(),[True,False,True,True])
            js = pd.read_csv(root/'fits_distances.csv')
            self.assertEqual(len(js),9)
            self.assertEqual(js.selected_for_display.sum(),4)
            row = js.loc[js.model_p.eq('normal') & js.model_q.str.contains('order: A B',regex=False)].iloc[0]
            self.assertTrue(row.selected_p)
            self.assertFalse(row.selected_q)
            reverse = js.loc[js.model_q.eq('normal') & js.model_p.str.contains('order: A B',regex=False)].iloc[0]
            self.assertFalse(reverse.selected_p)
            self.assertTrue(reverse.selected_q)
        with patch('return_distributions.joint_cli.read_returns') as reader, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['unused.csv','--best-asset-order'])
        reader.assert_not_called()

    def test_tasks_and_safety(self):
        names = ['normal', 'generalized-t-skewed', 'generalized-t', 'normal']
        tasks = list(fitting_tasks(names, ['C', 'A', 'B'], asset_orders='all'))
        self.assertEqual(len(tasks), 8)
        self.assertEqual([t[2] for t in tasks if t[0]=='generalized-t-skewed'],
                         list(itertools.permutations(['C', 'A', 'B'])))
        self.assertEqual(len(list(fitting_tasks(names, ['A', 'B']))), 3)
        with self.assertRaisesRegex(ValueError, '120 permutations'):
            fitting_tasks(names, list('ABCDE'), asset_orders='all')
        self.assertEqual(len(list(fitting_tasks(names, list('ABCDE'), asset_orders='all', max_asset_orders=120))), 122)
        self.assertEqual(len(list(fitting_tasks(['normal'], list('ABCDEFG'), asset_orders='all'))), 1)
        self.assertEqual(len(list(fitting_tasks(['asymmetric-laplace'], ['A','B'], ['zero','median'], 'all'))), 2)

    def test_cli_permutation_data_labels_files_js_and_univariate(self):
        sample = pd.DataFrame(np.random.default_rng(7).normal(size=(40, 2))*.01,
            columns=['B', 'A'], index=pd.bdate_range('2020-01-01', periods=40))
        def fitter(x, model, **kwargs):
            fit = dict(model=model, status='ok', observations=len(x), dimensions=2,
                parameters=5, loglik=10., aic=-10., bic=-5., location=x.mean(axis=0).tolist(),
                scatter=(np.eye(2)*.0001).tolist(), covariance=(np.eye(2)*.0001).tolist(),
                correlation=np.eye(2).tolist())
            if model=='generalized-t-skewed':
                fit.update(power=2., q=4., skewness=[.1, -.2])
            return fit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('return_distributions.joint_cli.read_returns', return_value=sample), \
                 patch('return_distributions.joint_cli.fit_joint', side_effect=fitter) as fits, \
                 patch('return_distributions.joint_cli.fit_univariate_sample', return_value=[]) as univ, \
                 contextlib.redirect_stdout(io.StringIO()) as console:
                code = main(['unused.csv', '--models', 'normal', 'generalized-t-skewed',
                    '--days', '20', '30', '--asset-orders', 'all', '--univariate', '--js-distance',
                    '--weights', 'A=.6', 'B=.4', '--simulations', '2000', '--output', str(root/'fits.json')])
            self.assertEqual(code, 0, console.getvalue())
            self.assertEqual(fits.call_count, 6)
            self.assertEqual(univ.call_count, 2)
            for offset, window in [(0,20),(3,30)]:
                np.testing.assert_array_equal(fits.call_args_list[offset].args[0], sample.tail(window).to_numpy())
                np.testing.assert_array_equal(fits.call_args_list[offset+1].args[0], sample.tail(window).to_numpy())
                np.testing.assert_array_equal(fits.call_args_list[offset+2].args[0], sample[['A','B']].tail(window).to_numpy())
            records = json.loads((root/'fits.json').read_text())
            self.assertEqual([f['symbols'] for f in records[:3]], [['B','A'],['B','A'],['A','B']])
            self.assertEqual(records[2]['asset_order'], 'A B')
            summary = pd.read_csv(root/'fits.csv')
            self.assertEqual(summary.asset_order.tolist()[:3], ['order-invariant','B A','A B'])
            risk = pd.read_csv(root/'fits_portfolio_risk.csv')
            self.assertEqual(len(risk), 8)
            distances = pd.read_csv(root/'fits_distances.csv')
            self.assertEqual(len(distances), 18)
            self.assertTrue(distances.model_p.str.contains('order: A B', regex=False).any())
            self.assertIn('risk range across 2/2 successful orderings', console.getvalue())

    def test_risk_weights_follow_symbols_and_failed_order(self):
        sample = pd.DataFrame([[.1,.2],[.2,.1]], columns=['A','B'], index=pd.bdate_range('2020-01-01', periods=2))
        records = [dict(model='generalized-t-skewed', status='ok', symbols=order, asset_order=' '.join(order))
                   for order in [['A','B'],['B','A']]]
        records.append(dict(records[0], status='failed'))
        with patch('return_distributions.joint_risk_report.simulate_joint_portfolio',
                   return_value=({'var_0.95':.1,'es_0.95':.2}, None)) as simulate:
            report = risk_report(records, sample, {'A':.6,'B':.4}, [.95])
        np.testing.assert_array_equal(simulate.call_args_list[0].args[1], [.6,.4])
        np.testing.assert_array_equal(simulate.call_args_list[1].args[1], [.4,.6])
        self.assertEqual(report.iloc[2].status, 'skipped')

    def test_cap_before_any_fit_and_no_save(self):
        sample = pd.DataFrame(np.random.default_rng(1).normal(size=(30,5)),
            columns=list('ABCDE'), index=pd.bdate_range('2020-01-01', periods=30))
        with patch('return_distributions.joint_cli.read_returns', return_value=sample), \
             patch('return_distributions.joint_cli.fit_joint') as fit, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(['unused.csv','--models','normal','generalized-t-skewed',
                                  '--asset-orders','all','--no-save']), 1)
        fit.assert_not_called()
        self.assertIn('--max-asset-orders 120', err.getvalue())
        with patch('return_distributions.joint_cli.read_returns', return_value=sample.iloc[:,:2]), \
             patch('return_distributions.joint_cli.fit_joint', side_effect=ValueError('failed fit')) as fit, \
             patch('pandas.DataFrame.to_csv', side_effect=AssertionError('must not write')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['unused.csv','--models','generalized-t-skewed',
                                  '--asset-orders','all','--no-save']), 1)
        self.assertEqual(fit.call_count, 2)


if __name__ == '__main__':
    unittest.main()
