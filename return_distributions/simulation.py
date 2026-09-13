"""Reproducible joint-family parameter recovery and model-selection study."""
import argparse
from .model_names import canonical_model, unique_models
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .multivariate import JOINT_MODELS, ALL_JOINT_MODELS, fit_joint, joint_distribution
from .joint_gh import GH_MODELS, mixture_moments
from .joint_power import covariance_factor


def preset(model, dimensions=2):
    """Moderate-tail examples with the same population mean/covariance."""
    model = canonical_model(model)
    d = dimensions
    sd = np.linspace(.01, .02, d)
    covariance = np.outer(sd, sd)*(.4**np.abs(np.arange(d)[:, None]-np.arange(d)))
    mu = np.full(d, .0003)
    truth = dict(model=model, location=mu.tolist(), mean=mu.tolist(), covariance=covariance.tolist())
    scatter = covariance.copy()
    if model == 'gh-skew-t':
        nu=10.; ew=nu/(nu-2); vw=2*nu**2/((nu-2)**2*(nu-4))
        gamma=sd*np.where(np.arange(d)%2,1.,-1.)*.2
        scatter=(covariance-vw*np.outer(gamma,gamma))/ew
        truth.update(df=nu,gamma=gamma.tolist(),location=(mu-ew*gamma).tolist())
    elif model.startswith('variance-gamma-'):
        shape = d/2+2.
        gamma = np.zeros(d) if model.endswith('symmetric') else sd*.2
        scatter = covariance-np.outer(gamma, gamma)/shape
        truth.update(vg_shape=shape, gamma=gamma.tolist(), location=(mu-gamma).tolist())
    elif 'slash' in model:
        from scipy import special
        q=6.;df=None if model.endswith('normal') else 8.
        delta=sd*np.where(np.arange(d)%2,1.,-1.)*.2 if model.startswith('skew-') else np.zeros(d)
        b=np.sqrt(2/np.pi) if df is None else np.sqrt(df/np.pi)/special.poch((df-1)/2,.5)
        factor=1. if df is None else df/(df-2)
        scatter=(covariance+(q/(q-1)*b)**2*np.outer(delta,delta))/(q/(q-2)*factor)
        solved=np.linalg.solve(scatter,delta)
        alpha=np.sqrt(scatter.diagonal())*solved/np.sqrt(1-delta@solved)
        truth.update(q=q,df=df,alpha=alpha.tolist(),location=(mu-q/(q-1)*b*delta).tolist())
    elif model.startswith('sdb-'):
        from scipy import special
        normal = model == 'sdb-skew-normal'
        df = None if normal else 6.
        delta = sd*np.where(np.arange(d)%2, 1., -1.)*.3
        factor = 1. if normal else df/(df-2)
        b = np.sqrt(2/np.pi) if normal else np.sqrt(df/np.pi)/special.poch((df-1)/2,.5)
        scatter = (covariance+b*b*np.outer(delta,delta))/factor-(1-2/np.pi)*np.diag(delta**2)-2/np.pi*np.outer(delta,delta)
        truth.update(df=df,delta=delta.tolist(),location=(mu-b*delta).tolist())
    elif model == 'student-t':
        truth['df'] = 6.
        scatter *= 4/6
    elif model == 'noncentral-t':
        from scipy import special
        df = 6.
        delta = sd*np.where(np.arange(d) % 2, 1., -1.)*.4
        b = np.sqrt(df/2)/special.poch((df-1)/2, .5)
        factor = df/(df-2)
        scatter = (covariance-(factor-b*b)*np.outer(delta, delta))/factor
        truth.update(df=df, delta=delta.tolist(), location=(mu-b*delta).tolist())
    elif model == 'azzalini-skew-t':
        from scipy import special
        df = 6.
        delta = sd*np.where(np.arange(d) % 2, 1., -1.)*.2
        b = np.sqrt(df/np.pi)/special.poch((df-1)/2, .5)
        scatter = (covariance+b*b*np.outer(delta, delta))*(df-2)/df
        solved = np.linalg.solve(scatter, delta)
        alpha = np.sqrt(scatter.diagonal())*solved/np.sqrt(1-delta@solved)
        truth.update(df=df, alpha=alpha.tolist(), location=(mu-b*delta).tolist())
    elif model in {'laplace', 'ged'}:
        p = 1. if model == 'laplace' else 1.25
        truth['power'] = p
        scatter /= covariance_factor(d, p)
    elif model in GH_MODELS:
        lam = .7 if model.startswith('gh-') else ((d+1)/2 if model.startswith('hyperbolic') else -.5)
        psi = 2.
        ew, vw = mixture_moments(lam, psi)
        direction = sd*np.where(np.arange(d) % 2, 1., -1.)
        gamma = np.zeros(d) if model.endswith('symmetric') else direction*np.sqrt(.15/(vw*(direction@np.linalg.solve(covariance, direction))))
        scatter = (covariance-vw*np.outer(gamma, gamma))/ew
        truth.update(gamma=gamma.tolist(), psi=psi, chi=1., **{'lambda': lam}, location=(mu-ew*gamma).tolist())
    truth['scatter'] = scatter.tolist()
    return truth


def parameters(fit):
    """Free distribution parameters and useful derived moments, in input units."""
    result = {}
    for key in ['df', 'q', 'power', 'psi', 'lambda', 'vg_shape']:
        if fit.get(key) is not None:
            result[key] = float(fit[key])
    for key in ['location', 'gamma', 'alpha', 'delta', 'mean']:
        if fit.get(key) is not None:
            result.update({f'{key}[{i}]': float(v) for i, v in enumerate(fit[key])})
    for key in ['scatter', 'covariance']:
        if fit.get(key) is not None:
            matrix = np.asarray(fit[key])
            result.update({f'{key}[{i},{j}]': float(matrix[i, j]) for i in range(len(matrix)) for j in range(i+1)})
    return result


def summarize(fits, recovery, candidates):
    """Selection only on replications with ALL requested fits successful."""
    selections = []
    for (truth_id, n), group in fits.groupby(['truth_id', 'sample_size'], sort=False):
        reps = list(group.groupby('replication'))
        for criterion in ['aic', 'bic']:
            credit = {m: 0. for m in candidates}
            eligible = 0
            for _, run in reps:
                if len(run) != len(candidates) or not run.status.eq('ok').all() or not np.isfinite(run[criterion]).all():
                    continue
                eligible += 1
                tied = run.loc[np.isclose(run[criterion], run[criterion].min(), rtol=0, atol=1e-8), 'model']
                for model in tied:
                    credit[model] += 1/len(tied)
            for model in candidates:
                selections.append(dict(truth_id=truth_id, sample_size=n, criterion=criterion, model=model,
                                       replications=len(reps), eligible=eligible, excluded=len(reps)-eligible,
                                       selection_credit=credit[model], selection_rate=credit[model]/eligible if eligible else np.nan))
    estimates = []
    if not recovery.empty:
        for (truth_id, n, parameter), rows in recovery.groupby(['truth_id', 'sample_size', 'parameter'], sort=False):
            valid = rows.loc[np.isfinite(rows.estimate)]
            errors = valid.estimate-valid.truth
            estimates.append(dict(truth_id=truth_id, sample_size=n, parameter=parameter,
                                  truth=rows.truth.iloc[0], estimates=len(valid), unavailable=len(rows)-len(valid),
                                  mean_estimate=valid.estimate.mean(), bias=errors.mean(),
                                  sd=valid.estimate.std(ddof=1), rmse=np.sqrt(np.mean(errors**2)) if len(valid) else np.nan))
    return pd.DataFrame(selections), pd.DataFrame(estimates)


def run_study(truths, candidates, sample_sizes, repetitions, seed=12345, max_iterations=2000, progress=None):
    candidates = unique_models(candidates)
    rows, recovery, full = [], [], []
    total = len(truths)*len(sample_sizes)*repetitions*len(candidates)
    for truth_id, truth in enumerate(truths):
        frozen = joint_distribution(truth)
        for n in sample_sizes:
            for rep in range(repetitions):
                # Streams do not depend on candidate order, sample-size order or failures.
                rng = np.random.default_rng(np.random.SeedSequence([seed, truth_id, n, rep]))
                x = np.asarray(frozen.rvs(size=n, random_state=rng)).reshape(n, -1)
                for model in candidates:
                    started = time.perf_counter()
                    try:
                        fit = fit_joint(x, model, max_iterations=max_iterations)
                    except Exception as exc:
                        fit = dict(model=model, status='failed', error=str(exc), fit_sec=time.perf_counter()-started)
                    meta = dict(truth_id=truth_id, generating_model=truth['model'], sample_size=n, replication=rep+1)
                    full.append(dict(**meta, fit=fit))
                    rows.append(dict(**meta, model=model, **{k: fit.get(k, np.nan) for k in ['aic', 'bic', 'loglik', 'status', 'fit_sec', 'error']}))
                    if model == truth['model']:
                        values = parameters(fit) if fit['status'] == 'ok' else {}
                        for key, value in parameters(truth).items():
                            recovery.append(dict(**meta, parameter=key, truth=value, estimate=values.get(key, np.nan), status=fit['status']))
                    if progress:
                        progress(len(rows), total)
    fits, recovery = pd.DataFrame(rows), pd.DataFrame(recovery)
    selection, accuracy = summarize(fits, recovery, candidates)
    return fits, recovery, selection, accuracy, full


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generators', type=canonical_model, choices=ALL_JOINT_MODELS, nargs='+', default=list(JOINT_MODELS))
    parser.add_argument('--models', type=canonical_model, choices=ALL_JOINT_MODELS, nargs='+', default=list(JOINT_MODELS))
    parser.add_argument('--truth-file', type=Path, help='Saved joint-fit JSON list, used instead of presets; filtered by --generators')
    parser.add_argument('--dimensions', type=int, default=2, help='Preset dimensions; ignored for saved truths')
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[500])
    parser.add_argument('--replications', type=int, default=10)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--output-dir', type=Path, default=Path('simulation_output'))
    args = parser.parse_args(argv)
    if args.dimensions < 1 or args.replications < 1 or args.seed < 0 or args.max_iterations < 1 or min(args.sample_sizes) < 8:
        parser.error('Require positive dimensions/replications/iterations, nonnegative seed and sample sizes >= 8')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions.simulation', *(sys.argv[1:] if argv is None else argv)]))
    try:
        if args.truth_file:
            truths = json.loads(args.truth_file.read_text(encoding='utf-8'))
            if not isinstance(truths, list) or not all(isinstance(t, dict) for t in truths):
                raise ValueError('Truth file must contain a list of joint fit objects')
            truths = [t for t in truths if t.get('model') in args.generators]
            if any(t.get('status', 'ok') != 'ok' for t in truths):
                raise ValueError('Saved truths must have status ok, not failed or boundary')
        else:
            truths = [preset(m, args.dimensions) for m in dict.fromkeys(args.generators)]
        if not truths:
            raise ValueError('No generating distributions selected')
        for truth in truths:
            if truth.get('chi', 1) != 1:
                raise ValueError('GH truth chi must equal 1')
            d = len(truth['location'])
            if min(args.sample_sizes) < max(8, d+2):
                raise ValueError('Sample sizes must be at least max(8, dimensions+2)')
            joint_distribution(truth)  # Validate before starting a long study.
        paths = {name: args.output_dir/name for name in ['config.json', 'truths.json', 'fits.json', 'fits.csv', 'recovery.csv', 'selection.csv', 'recovery_summary.csv']}
        if args.truth_file and args.truth_file.resolve() in {p.resolve() for p in paths.values()}:
            raise ValueError('Output would overwrite the truth input; select a different output directory')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths['config.json'].write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2), encoding='utf-8')
        paths['truths.json'].write_text(json.dumps(truths, indent=2, allow_nan=False), encoding='utf-8')
        last = [-1]
        def progress(done, total):
            bucket = int(done*20/total)
            if bucket != last[0]:
                elapsed = time.perf_counter()-start
                print(f'Simulation fits: {done}/{total} ({100*done/total:.0f}%); {elapsed:.1f}s; ETA {elapsed*(total-done)/done:.1f}s', flush=True)
                last[0] = bucket
        fits, recovery, selection, accuracy, full = run_study(truths, list(dict.fromkeys(args.models)),
            list(dict.fromkeys(args.sample_sizes)), args.replications, args.seed, args.max_iterations, progress)
        for name, frame in [('fits.csv', fits), ('recovery.csv', recovery), ('selection.csv', selection), ('recovery_summary.csv', accuracy)]:
            frame.to_csv(paths[name], index=False)
        paths['fits.json'].write_text(json.dumps(full, indent=2, allow_nan=False), encoding='utf-8')
        print('\nTruth IDs: ' + ', '.join(f'{i}={t["model"]}' for i, t in enumerate(truths)))
        print('\nFit status counts:')
        print(fits.groupby(['truth_id', 'sample_size', 'model', 'status']).size().to_string())
        print('\nAIC/BIC selection (complete successful comparisons only; ties split credit):')
        print(selection.to_string(index=False, float_format=lambda v: f'{v:.4g}', na_rep='n/a'))
        print('\nCorrect-family recovery (successful interior fits only):')
        print(accuracy.to_string(index=False, float_format=lambda v: f'{v:.5g}', na_rep='n/a') if not accuracy.empty else 'No generating family was included among candidate models.')
        print(f'\nOutputs: {args.output_dir}\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        print('Selection rates are conditional on all candidate fits succeeding; inspect excluded counts. Recovery is conditional on own-family success.')
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
