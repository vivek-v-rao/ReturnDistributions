"""Joint return distribution fitting on common observations."""
import argparse
import itertools
import math
from .model_names import canonical_model, unique_models, display_model_label
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data import read_returns
from .multivariate import fit_joint, JOINT_MODELS, ALL_JOINT_MODELS
from .portfolio_cli import read_weights
from .joint_risk_report import risk_report, print_risk_report, weight_vector
from .joint_laplace_mixture import LAPLACE_MIXTURE_MODELS, mode_bandwidth
from .table_format import aligned_table, is_time_column
from .joint_univariate import model_mapping, fit_univariate_sample
from .distribution_distances import add_distance_options, selected_metrics, compare_univariate, compare_joint
from .fit_timeout import run_fit, FitTimeout, positive_seconds
from .vol_standardization import ewma_standardize, annotate_fit
from .joint_finite_mixture import fit_finite_mixture, FINITE_MIXTURE_MODELS, JointFiniteMixture
from .fit_ranks import criterion_ranks
from .sample_blocks import add_block_options, validate_blocks, date_mask, blocks, rank_groups, endpoint_scale

ORDER_DEPENDENT_MODELS = frozenset({'generalized-t-skewed', 'ged-skewed', 'fs-skew-normal'})


def select_best_orders(fits):
    """Select one finite, accepted likelihood per order-dependent family/window.

    Called separately for each window. Exact ties retain first attempted order.
    Invariant references stay visible, including their failures.
    """
    for fit in fits:
        fit['selected_for_display'] = fit['model'] not in ORDER_DEPENDENT_MODELS
    rows = []
    for model in dict.fromkeys(f['model'] for f in fits if f['model'] in ORDER_DEPENDENT_MODELS):
        group = [f for f in fits if f['model'] == model]
        good = [f for f in group if f.get('status') == 'ok' and
                f.get('loglik') is not None and np.isfinite(f['loglik'])]
        good.sort(key=lambda f: f['loglik'], reverse=True)
        if good: good[0]['selected_for_display'] = True
        rows.append(dict(model=model, best_order=good[0]['asset_order'] if good else 'none',
                         successful=len(good), attempted=len(group),
                         loglik=good[0]['loglik'] if good else np.nan,
                         runner_up_gap=good[0]['loglik']-good[1]['loglik'] if len(good)>1 else np.nan))
    print('\nBest asset order by accepted log likelihood (per family/window):')
    print(aligned_table(pd.DataFrame(rows)) if rows else 'No order-dependent models selected.')
    print('Exact ties retain the first attempted order. Order-selection uncertainty is not counted by ordinary AIC/BIC.')


def fitting_tasks(models, symbols, laplace_locations=None, asset_orders='given', max_asset_orders=24):
    """Validate the factorial workload before lazily enumerating fit tasks."""
    models = unique_models(models)
    symbols = tuple(symbols)
    if len(set(symbols)) != len(symbols): raise ValueError('Duplicate asset symbols')
    expand = asset_orders == 'all' and bool(ORDER_DEPENDENT_MODELS.intersection(models))
    count = math.factorial(len(symbols)) if expand else 1
    if count > max_asset_orders:
        raise ValueError(f'All asset orders requires {count} permutations per order-dependent model/window; '
                         f'limit is {max_asset_orders}. Set --max-asset-orders {count} to permit this workload.')
    return ((model, method, order) for model in models
            for method in ((laplace_locations or [None]) if model in LAPLACE_MIXTURE_MODELS else [None])
            for order in (itertools.permutations(symbols) if expand and model in ORDER_DEPENDENT_MODELS else [symbols]))


def rank_mixture_comparison(summary):
    """Rank finite criteria, retaining status as the qualification on each fit."""
    windows = rank_groups(summary)
    ranked = criterion_ranks(summary, [*windows, summary['model']], True, '_family')
    return criterion_ranks(ranked, windows, True, '_all')


def print_mixture_components(fit):
    """Report component moments, not scatter or skewed-family locations."""
    symbols = fit['symbols']
    units = 'EWMA-standardized return units' if fit.get('vol_standardization') == 'ewma' else 'return units'
    print(f'\nMixture components ({units}; per input period, not annualized; status: {fit["status"]}):')
    ordered_components = sorted(zip(fit['mixture_weights'], fit['component_fits']),
                                key=lambda item: item[0], reverse=True)
    for number, (weight, record) in enumerate(ordered_components, 1):
        component = JointFiniteMixture(dict(fit, components=1, mixture_weights=[1.], component_fits=[record]))
        mean, covariance = component.mean(), component.cov()
        sd = np.sqrt(np.diag(covariance))
        print(f'\nComponent {number}; weight: {weight:.6f} ({100*weight:.2f}%)')
        table = pd.DataFrame(dict(symbol=symbols, mean=mean, standard_deviation=sd))
        print(table.to_string(index=False, float_format=lambda v: f'{v:.6f}', na_rep='n/a'))
        if not np.isfinite(mean).all():
            print('Component mean undefined (Student-t df <= 1).')
        if np.isfinite(covariance).all() and np.all(sd > 0):
            corr = covariance / np.outer(sd, sd)
            print('\nComponent return correlation:')
            print(pd.DataFrame(corr, index=symbols, columns=symbols).to_string(float_format=lambda v: f'{v:.3f}'))
        else:
            print('\nComponent return correlation unavailable: finite, positive variances are required (Student-t requires df > 2).')


def aggregate_fit_comparison(summary):
    """Compact all-family comparison, honoring any best-order display selection."""
    selected = summary.copy()
    if 'selected_for_display' in selected:
        selected = selected.loc[selected.selected_for_display.eq(True)].copy()
    if 'components' not in selected:
        selected['components'] = 1
    columns = ['model', 'components', 'window', 'subperiods', 'block', 'first_date', 'last_date', 'observations',
               'vol_standardization', 'vol_lambda', 'asset_order', 'location_method',
               'location_source', 'estimation_method', 'parameters', 'loglik', 'aic', 'bic',
               'status', 'fit_sec', 'error']
    selected = selected[[c for c in columns if c in selected]]
    return criterion_ranks(selected, rank_groups(selected), include_unconverged=True)


def format_fit_comparison(summary):
    """Compact console view only; retain the full saved summary schema."""
    if summary.empty:
        return 'No fits.'
    shared = []
    columns = []

    def display(value):
        if pd.isna(value):
            return 'n/a'
        if isinstance(value, (float, np.floating)):
            return f'{value:.5g}'
        return str(value)

    for name in summary.columns:
        if name == 'fit_restriction':
            continue
        values = summary[name]
        if name == 'window' and values.isna().all():
            shared.append('window: all available history')
        elif name != 'model' and '_rank' not in name and values.isna().all():
            continue
        elif name != 'model' and '_rank' not in name and values.notna().all() and values.nunique() == 1:
            value = values.iloc[0]
            rendered = f'{value:.3f}' if is_time_column(name) else display(value)
            shared.append(f'{name}: {rendered}')
        else:
            columns.append(name)
    lead = ['model', 'components', 'vol_standardization', 'vol_lambda', 'asset_order', 'parameters', '#param', 'loglik', 'aic', 'bic', 'rank', 'aic_rank', 'bic_rank', 'aic_rank_family', 'bic_rank_family', 'aic_rank_all', 'bic_rank_all', 'fit_sec']
    columns = [c for c in lead if c in columns] + [c for c in columns if c not in lead]
    lines = ['; '.join(shared)] if shared else []
    lines.append(aligned_table(summary[columns]))
    return '\n'.join(lines)


def main(argv=None):
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--asset-orders', choices=['given', 'all'], default='given',
                        help='Fit every asset permutation only for order-dependent families (generalized-t-skewed, ged-skewed, fs-skew-normal)')
    parser.add_argument('--best-asset-order', action='store_true',
                        help='With --asset-orders all, display only the highest-likelihood accepted order per family/window; save all orders with selection flags')
    parser.add_argument('--max-asset-orders', type=int, default=24,
                        help='Safety limit per order-dependent model/window; increase explicitly to permit more permutations (default 24)')
    parser.add_argument('--input-type', choices=['prices', 'returns'], default='prices')
    parser.add_argument('--return-type', choices=['simple', 'log'], default='simple')
    parser.add_argument('--return-scale', type=positive_seconds, default=1., help='Multiply input returns by this positive factor (default 1)')
    parser.add_argument('--days', type=int, nargs='+')
    parser.add_argument('--standardize-vol', choices=['none', 'ewma'], nargs='+', help='Fit raw and/or lagged EWMA-standardized returns on common dates; default none')
    decay_group = parser.add_mutually_exclusive_group()
    decay_group.add_argument('--vol-lambda', type=float, nargs='+', help='One or more EWMA decays per observation; default 0.94. Raw fits run once.')
    decay_group.add_argument('--vol-halflife', type=positive_seconds, help='Alternative EWMA half-life in input observations')
    parser.add_argument('--vol-warmup', type=int, default=63, help='Initial observations used only to seed EWMA; default 63')
    parser.add_argument('--vol-floor', type=positive_seconds, default=1e-8, help='Minimum EWMA standard deviation in return units; default 1e-8')
    parser.add_argument('--models', type=canonical_model, choices=(*ALL_JOINT_MODELS,'all'), nargs='+', default=list(JOINT_MODELS), help='Use all alone for every supported joint model, including experimental models. Aliases nig, gh, hyperbolic, variance-gamma select skewed variants')
    parser.add_argument('--sdb-cdf-points', type=int, default=512, help='Experimental SDB fitting grid: power of two, 64--16384; audit uses 4x points')
    parser.add_argument('--sdb-seed', type=int, default=12345)
    parser.add_argument('--nts-quadrature-points', type=int, default=256,
                        help='NTS mixing/angle quadrature order, 64--1024; final audit doubles it and tightens tail bounds')
    parser.add_argument('--slash-quadrature-points',type=int,default=64,help='Slash fitting quadrature order, 16--256; audit uses 4x points')
    parser.add_argument('--location', type=float, help='Fix every location component to this value; Laplace-mixture models require fixed location and default to zero')
    parser.add_argument('--laplace-location',choices=['zero','median','mode'],nargs='+',help='Laplace-mixture location methods to compare; default zero. Median/mode are two-stage fits.')
    parser.add_argument('--laplace-mode-bandwidth',type=mode_bandwidth,help='Joint Gaussian KDE bandwidth: scott (default), silverman, or a positive factor')
    parser.add_argument('--max-iterations', type=int, default=2000)
    parser.add_argument('--max-components', type=int, default=1, help='Fit 1 through K components for selected mixture families (1--8; default 1)')
    parser.add_argument('--mixture-models', type=canonical_model, choices=(*FINITE_MIXTURE_MODELS, 'all'), nargs='+',
                        help='Families for 2+ components; all selects supported families from --models (also the default). Explicit families add missing single-component baselines')
    parser.add_argument('--mixture-starts', type=int, default=5, help='Mixture optimizer starts, including the one-component baseline (default 5)')
    parser.add_argument('--mixture-min-weight', type=float, default=.01, help='Minimum mixture component weight (default .01)')
    parser.add_argument('--mixture-eigen-floor', type=positive_seconds, default=1e-4, help='Scatter eigenvalue floor after sample-SD scaling (default 1e-4)')
    parser.add_argument('--mixture-seed', type=int, default=12345)
    parser.add_argument('--fit-timeout', type=positive_seconds, metavar='SECONDS',
                        help='Hard limit per model/window/order/location fit (also separate univariate fits); startup included; default unlimited')
    parser.add_argument('--output', type=Path, default=Path('joint_distribution_fits.json'))
    parser.add_argument('--no-save', action='store_true', help='Screen output only; skip all result files and ignore output paths')
    parser.add_argument('--univariate', action='store_true', help='Also fit each asset separately on the same complete-case sample; save <output stem>_univariate.csv unless --no-save')
    parser.add_argument('--copulas', nargs='+', choices=['gaussian', 'student-t'], help='Also fit two-stage copulas on the same sample')
    parser.add_argument('--marginal-models', nargs='+', help='Candidate copula marginal families; default student-t')
    parser.add_argument('--marginal-criterion', choices=['aic', 'bic'], default='aic')
    parser.add_argument('--cdf-clip', type=float, default=1e-10)
    weights_group=parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights',nargs='+',metavar='SYMBOL=WEIGHT')
    weights_group.add_argument('--weights-file',type=Path)
    parser.add_argument('--risk-levels',type=float,nargs='+',help='Portfolio VaR/ES confidence levels; default with weights: .95 .975 .99 .995')
    parser.add_argument('--portfolio-output',type=Path,help='Risk CSV; default <output stem>_portfolio_risk.csv')
    parser.add_argument('--allow-log-linear-combination',action='store_true')
    parser.add_argument('--simulations',type=int,default=100000,help='Draws for simulation-based portfolio risk (SDB, skewed generalized-t/GED, FS skew-normal)')
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--mc-batches',type=int,default=20)
    add_distance_options(parser)
    add_block_options(parser)
    args = parser.parse_args(argv)
    validate_blocks(parser, args)
    if args.marginal_models and not args.copulas: parser.error('--marginal-models requires --copulas')
    args.copulas = list(dict.fromkeys(args.copulas or []))
    args.marginal_models = unique_models(args.marginal_models or ['student-t'])
    if not np.isfinite(args.cdf_clip) or not 0 < args.cdf_clip < .01: parser.error('Require 0 < --cdf-clip < .01')
    if args.copulas:
        from .fitting import specification
        for name in args.marginal_models:
            try: specification(name)
            except ValueError as exc: parser.error(str(exc))
    normalizations = list(dict.fromkeys(args.standardize_vol or ['none']))
    ewma_enabled = 'ewma' in normalizations
    if not ewma_enabled and (args.vol_lambda is not None or args.vol_halflife is not None
                                    or args.vol_warmup != 63 or args.vol_floor != 1e-8):
        parser.error('Volatility filter options require --standardize-vol ewma')
    decays = list(dict.fromkeys([float(np.exp(np.log(.5)/args.vol_halflife))] if args.vol_halflife is not None
                               else (args.vol_lambda or [.94])))
    if any(not np.isfinite(v) or not 0 < v < 1 for v in decays) or args.vol_warmup < 2:
        parser.error('Require 0 < --vol-lambda < 1 and --vol-warmup >= 2')
    variants = [(mode, decay) for mode in normalizations for decay in (decays if mode == 'ewma' else [None])]
    if not 64 <= args.nts_quadrature_points <= 1024:
        parser.error('--nts-quadrature-points must be 64--1024')
    if args.best_asset_order and args.asset_orders != 'all':
        parser.error('--best-asset-order requires --asset-orders all')
    if args.max_asset_orders < 1: parser.error('--max-asset-orders must be positive')
    metrics = selected_metrics(args)
    if (args.ks_distance or args.kl_divergence) and not args.univariate:
        parser.error('--ks-distance and --kl-divergence require --univariate in the joint program')
    all_models='all' in args.models
    if all_models:
        if args.models!=['all']:
            parser.error('--models all must be used alone, without other model names')
        args.models=unique_models(ALL_JOINT_MODELS)
    if (not 1 <= args.max_components <= 8 or args.mixture_starts < 2 or args.mixture_seed < 0
            or not np.isfinite(args.mixture_min_weight) or not 0 < args.mixture_min_weight < 1/args.max_components):
        parser.error('Require 1--8 components, >=2 mixture starts, nonnegative seed, and 0 < minimum weight < 1/max-components')
    if args.mixture_models and args.max_components == 1:
        parser.error('--mixture-models requires --max-components >=2')
    if args.mixture_models and 'all' in args.mixture_models:
        if args.mixture_models != ['all']:
            parser.error('--mixture-models all must be used alone, without other model names')
        args.mixture_models = None
    mixture_models = unique_models(args.mixture_models if args.mixture_models is not None
                                   else [m for m in args.models if m in FINITE_MIXTURE_MODELS])
    added_baselines = [m for m in mixture_models if m not in args.models]
    args.models = unique_models([*args.models, *added_baselines])
    if args.max_components > 1 and not mixture_models:
        parser.error('--max-components >1 requires a supported family in --models or --mixture-models')
    if args.laplace_location is not None:
        if args.location is not None: parser.error('--laplace-location cannot be combined with --location')
        if not any(m in LAPLACE_MIXTURE_MODELS for m in args.models):
            parser.error('--laplace-location requires a Laplace-mixture model')
        args.laplace_location=list(dict.fromkeys(args.laplace_location))
    if args.laplace_mode_bandwidth is not None and 'mode' not in (args.laplace_location or []):
        parser.error('--laplace-mode-bandwidth requires --laplace-location mode')
    has_weights=bool(args.weights or args.weights_file)
    if not has_weights and (args.risk_levels or (args.portfolio_output and not args.no_save) or args.allow_log_linear_combination):
        parser.error('Portfolio risk options require --weights or --weights-file')
    if has_weights and args.return_type=='log' and not args.allow_log_linear_combination:
        parser.error('Portfolio risk requires simple returns; use --allow-log-linear-combination only for a weighted sum of log returns')
    levels=list(dict.fromkeys(args.risk_levels or [.95,.975,.99,.995]))
    if any(not np.isfinite(c) or not 0<c<1 for c in levels):
        parser.error('Risk levels must be strictly between 0 and 1')
    if args.simulations<1000 or args.seed<0 or args.mc_batches<2 or args.simulations//args.mc_batches<50:
        parser.error('Require >=1000 simulations, >=2 batches, >=50 draws/batch and a nonnegative seed')
    if not args.no_save:
        summary_path = args.output.with_suffix('.csv')
        risk_path=args.portfolio_output or args.output.with_name(args.output.stem+'_portfolio_risk.csv')
        if has_weights and risk_path.suffix.lower()!='.csv':
            parser.error('--portfolio-output must end in .csv')
        if args.output.suffix.lower() != '.json':
            parser.error('--output must end in .json; a matching summary CSV is also written')
        inputs={p.resolve() for p in [args.file,args.weights_file] if p is not None}
        outputs=[args.output.resolve(),summary_path.resolve()]+([risk_path.resolve()] if has_weights else [])
        if args.max_components > 1:
            mixture_path = args.output.with_name(args.output.stem+'_mixtures.csv')
            outputs.append(mixture_path.resolve())
        if args.univariate:
            univariate_path = args.output.with_name(args.output.stem+'_univariate.csv')
            outputs.append(univariate_path.resolve())
        if metrics:
            distance_path = args.output.with_name(args.output.stem+'_distances.csv')
            outputs.append(distance_path.resolve())
        if inputs.intersection(outputs) or len(set(outputs))!=len(outputs):
            parser.error('Output must not overwrite input')
    if args.max_iterations < 1 or args.days and min(args.days) < 8:
        parser.error('Iterations must be positive; windows must be at least 8')
    print('Command: ' + ' '.join([sys.executable, '-m', 'return_distributions.joint_cli', *(sys.argv[1:] if argv is None else argv)]))
    copula_paths = {}
    if args.copulas and not args.no_save:
        copula_paths = {key: args.output.with_name(args.output.stem+suffix) for key, suffix in
                        [('fits','_copulas.json'), ('summary','_copulas.csv'), ('marginals','_copula_marginals.csv'), ('risk','_copula_risk.csv')]}
        if args.file.resolve() in {p.resolve() for p in copula_paths.values()}:
            parser.error('Copula output must not overwrite input')
    if all_models:
        print(f'Expanded models ({len(args.models)}): '+', '.join(map(display_model_label, args.models)))
        print('Warning: all includes optional/experimental models; SDB and slash fits can be slow.',flush=True)
    if args.max_components > 1:
        print(f'Finite mixtures: {", ".join(map(display_model_label, mixture_models))}; 1--{args.max_components} components; {args.mixture_starts} starts.')
        if added_baselines: print('Added single-component baselines: '+', '.join(map(display_model_label, added_baselines)))
        if 'student-t' in mixture_models: print('Student-t mixtures share estimated df [0.25,200].')
        if 'ged' in mixture_models: print('GED mixtures share estimated power [0.25,10]; direct numerical likelihood optimization.')
        if 'nig-skewed' in mixture_models:
            print('NIG mixtures share psi (log bounds [-12,12]), with chi=1 and lambda=-0.5; separate locations, scatter and gamma. Gamma bounded to [-10,10] in sample-SD units.')
        print(f'Minimum weight {args.mixture_min_weight:g}; scatter eigenvalue floor {args.mixture_eigen_floor:g} in sample-SD-scaled coordinates. Constraint-bound fits are excluded from risk; finite AIC/BIC values remain in mixture-table ranks.')
        print('Mixture AIC/BIC are descriptive: mixtures are nonregular; multiple starts do not guarantee a global maximum.')
    if any(m in LAPLACE_MIXTURE_MODELS for m in args.models):
        print('Laplace-mixture location choices: '+', '.join(args.laplace_location or [f'fixed {0. if args.location is None else args.location:g}'])+'. Other families retain their usual location settings.',flush=True)
        if any(m in ('median','mode') for m in (args.laplace_location or [])):
            print('Median/mode are two-stage fits: ordinary AIC/BIC and rank are omitted; descriptive two_stage_aic/bic count the pilot coordinates, not KDE complexity.',flush=True)
    try:
        weights=read_weights(args.weights,args.weights_file) if has_weights else None
        frame = read_returns(args.file, args.symbols, args.input_type, args.return_type)
        frame = frame * args.return_scale
        print(f'Return scale: {args.return_scale:g}; raw parameters and likelihoods use scaled return units.')
        if len(frame.columns) < 2:
            raise ValueError('Select at least two assets for the joint CLI')
        # Check the workload before any fitting, including invariant families.
        fitting_tasks(args.models, frame.columns, args.laplace_location, args.asset_orders, args.max_asset_orders)
        if args.asset_orders == 'all':
            dependent = ORDER_DEPENDENT_MODELS.intersection(args.models)
            if dependent:
                print(f'Asset orders: {math.factorial(len(frame.columns))} permutations per order-dependent model/window; '
                      'order-invariant families fit once per window/location choice.')
                print('Selecting the best ordering adds model-selection uncertainty not counted by ordinary AIC/BIC.')
            else:
                print('Asset orders: no order-dependent models selected; each family fit once per window/location choice.')
        raw_frame = frame
        vol_cache = {}
        eligible_mask = raw_frame.notna().all(axis=1)
        if ewma_enabled:
            for decay in decays:
                standardized, vol_scales, next_vol = ewma_standardize(raw_frame, decay, args.vol_warmup, args.vol_floor)
                vol_cache[decay] = (standardized, vol_scales, next_vol)
                eligible_mask &= standardized.notna().all(axis=1)
                print(f'Lagged zero-mean EWMA: lambda={decay}; half-life={np.log(.5)/np.log(decay):.3f} observations; warmup={args.vol_warmup}; SD floor={args.vol_floor:g}.')
            print('Warmup excluded; missing returns reset each asset filter. Decay is per input observation, not calendar day.')
            print('EWMA fit parameters describe standardized returns; raw fit parameters use scaled returns. All likelihood/AIC/BIC scores use scaled return units.')
            print('Compare raw/normalized scores only on identical dates. Selecting decay or warmup adds uncounted selection uncertainty.')
            print('Location options use each fit\'s input units; next-period EWMA risk uses its corresponding lambda\'s SDs.')
        complete = raw_frame.loc[eligible_mask]
        complete = complete.loc[date_mask(complete.index, args.date_min, args.date_max)]
        if weights is not None:
            weight_vector(frame.columns,weights)
            print('Weights used without normalization: '+', '.join(f'{s}={w:g}' for s,w in weights.items()))
            print(f'Net exposure: {sum(weights.values()):g}; gross exposure: {sum(abs(w) for w in weights.values()):g}. No cash return or financing costs added.')
            if args.return_type=='log': print('Warning: weighted sum of asset log returns, NOT the portfolio log return.')
        data_elapsed = time.perf_counter()-start
        results = []
        reports=[]
        univariate_tables = []
        distance_rows = []
        copula_records, copula_summaries, copula_audit, copula_risks = [], [], [], []
        mapped = {}
        if args.univariate:
            mapped, skipped = model_mapping(args.models)
            print('Univariate comparisons are separately estimated fits, NOT implied marginals of the joint fits.')
            print('Use univariate AIC/BIC only within the same asset/window, not against joint scores.')
            for name, sources in mapped.items():
                print('Univariate mapping: '+', '.join(sources)+' -> '+name)
            for model in skipped:
                print(f'Univariate comparison skipped for {model}: no corresponding univariate fitter is registered.')
            if any(m in LAPLACE_MIXTURE_MODELS for m in args.models):
                print('Univariate Laplace counterparts estimate location freely unless --location is supplied; joint Laplace location pilots/default-zero are not applied.')
        print('Same complete-case observations for every joint model; no pairwise deletion.')
        if len(variants) > 1:
            print('All raw/EWMA/lambda variants use identical dates, excluding warmup and all filter-ineligible observations; raw fits run once per window.')
        for (window, count, block, selected), (normalization, decay) in itertools.product(blocks(complete.index, args.days, args.subperiods), variants):
            eligible = complete.loc[selected]
            print(f'\nPartition: {count}; block: {block}/{count}')
            if normalization == 'ewma':
                standardized, vol_scales, next_vol = vol_cache[decay]
                sample = standardized.loc[eligible.index]
                next_vol = endpoint_scale(raw_frame, vol_scales, selected[-1], decay, args.vol_floor)
                print('Block-end next-period EWMA SDs (scaled return units): '+', '.join(f'{s}={v:.6g}' for s,v in next_vol.items()))
            else:
                sample = eligible
            if sample.empty:
                raise ValueError('No common returns')
            first, last = str(sample.index[0].date()), str(sample.index[-1].date())
            print(f'\n{window or "All"} periods: {len(sample)} common returns; {first} to {last}')
            if args.standardize_vol:
                print(f'Volatility normalization: {normalization}; '+(f'lambda={decay}; standardized parameter units; conditional risk' if normalization == 'ewma' else 'scaled return parameter units; unconditional risk'))
            if count == 1 and window and len(sample) < window:
                print('Warning: fewer common returns than requested')
            window_fits=[]
            tasks=fitting_tasks(args.models, sample.columns, args.laplace_location, args.asset_orders, args.max_asset_orders)
            expanded_tasks = ((model, method, order, count) for model, method, order in tasks
                              for count in (range(1, args.max_components+1) if model in mixture_models else [1]))
            for model,location_method,order,components in expanded_tasks:
                fit_sample = sample.loc[:, list(order)]
                label=model+(f' [{location_method}]' if location_method is not None else '')
                if args.asset_orders == 'all' and model in ORDER_DEPENDENT_MODELS:
                    label += ' [order: '+ ' '.join(order)+']'
                if components > 1: label += f' [{components} components]'
                print(f'Fitting joint {display_model_label(label)} ({len(sample.columns)} assets)...', flush=True)
                extra={}
                if model in {'nts-symmetric','nts-skewed'}:
                    extra['nts_points'] = args.nts_quadrature_points
                if model in LAPLACE_MIXTURE_MODELS:
                    extra=dict(laplace_location=location_method,
                               laplace_mode_bandwidth=args.laplace_mode_bandwidth if location_method=='mode' else None)
                try:
                    fit_started = time.perf_counter()
                    if components > 1:
                        baseline = next(f for f in window_fits if f['model'] == model and f.get('components', 1) == 1)
                        fit = run_fit(fit_finite_mixture, fit_sample.to_numpy(), model, components,
                                      fit_timeout=args.fit_timeout, baseline=baseline, location=args.location,
                                      max_iterations=args.max_iterations, starts=args.mixture_starts,
                                      min_weight=args.mixture_min_weight, eigen_floor=args.mixture_eigen_floor,
                                      seed=args.mixture_seed)
                    else:
                        fit = run_fit(fit_joint, fit_sample.to_numpy(), model, fit_timeout=args.fit_timeout,
                                        location=args.location, max_iterations=args.max_iterations,
                                        sdb_points=args.sdb_cdf_points, sdb_seed=args.sdb_seed, slash_points=args.slash_quadrature_points,**extra)
                except (ValueError, ArithmeticError, FitTimeout) as exc:
                    fit = dict(model=model, status='timeout' if isinstance(exc, FitTimeout) else 'failed',
                               error=str(exc), observations=len(sample), converged=False, fit_sec=time.perf_counter()-fit_started)
                    print(f'{display_model_label(label)}: {fit["status"]}: {exc}', flush=True)
                    if model in LAPLACE_MIXTURE_MODELS:
                        method=location_method or ('zero' if args.location is None else 'provided')
                        fit.update(location_method=method,fit_label=f'{model} [{method}]')
                fit.update(symbols=list(order), window=window, first_date=first, last_date=last, return_type=args.return_type)
                fit['return_scale'] = args.return_scale
                fit.update(subperiods=count, block=block)
                if args.max_components > 1: fit['components'] = components
                if components > 1: fit['fit_label'] = label
                if normalization == 'ewma':
                    annotate_fit(fit, vol_scales.loc[sample.index, list(order)], next_vol.loc[list(order)],
                                 decay, args.vol_warmup, args.vol_floor)
                elif args.standardize_vol:
                    fit.update(vol_standardization='none', parameter_units='original returns', likelihood_units='original returns')
                fit['parameter_units'] = 'standardized returns' if normalization == 'ewma' else 'scaled returns'
                fit['likelihood_units'] = 'scaled returns'
                if args.asset_orders == 'all':
                    fit['asset_order'] = ' '.join(order) if model in ORDER_DEPENDENT_MODELS else 'order-invariant'
                    if model in ORDER_DEPENDENT_MODELS: fit['fit_label'] = label
                if len(variants) > 1:
                    variant_label = f'ewma lambda={decay}' if normalization == 'ewma' else 'none'
                    fit['fit_label'] = fit.get('fit_label', label)+f' [vol: {variant_label}]'
                results.append(fit)
                window_fits.append(fit)
                if components > 1 and 'component_fits' in fit:
                    print_mixture_components(fit)
                if 'scatter' in fit and not (args.best_asset_order and model in ORDER_DEPENDENT_MODELS):
                    print('Location: ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(order, fit['location'])))
                    corr = fit.get('correlation') or fit['scatter_correlation']
                    if 'alpha' in fit and np.ndim(fit['alpha']) > 0:
                        print('Skewness alpha: ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(order, fit['alpha'])))
                    if model in ORDER_DEPENDENT_MODELS:
                        print('Cholesky-coordinate log-skewness (asset-order dependent): ' + ', '.join(
                            f'{s}={v:.6g}' for s, v in zip(order, fit['skewness'])))
                    if model in (*LAPLACE_MIXTURE_MODELS,'nts-symmetric','nts-skewed') and 'gamma' in fit:
                        print('Skew gamma ('+('standardized units' if normalization == 'ewma' else 'return units')+'): '+', '.join(f'{s}={v:.6g}' for s,v in zip(order,fit['gamma'])))
                    if 'delta' in fit:
                        label = 'SDB skew loadings' if model.startswith('sdb-') else 'Noncentrality delta'
                        print(label+' ('+('standardized units' if normalization == 'ewma' else 'return units')+'): ' + ', '.join(f'{s}={v:.6g}' for s, v in zip(order, fit['delta'])))
                    if 'cdf_audit_passed' in fit:
                        print(f"CDF accuracy audit: passed={fit['cdf_audit_passed']}; loglik change={fit['cdf_loglik_change']:.5g}; max row log-density change={fit['cdf_max_logpdf_change']:.5g}")
                    if 'quadrature_audit_passed' in fit:
                        print(f"Slash quadrature audit: passed={fit['quadrature_audit_passed']}; loglik change={fit['quadrature_loglik_change']:.5g}; max row log-density change={fit['quadrature_max_logpdf_change']:.5g}")
                    if 'nts_audit_passed' in fit:
                        print(f"NTS quadrature audit: passed={fit['nts_audit_passed']}; loglik change={fit['nts_loglik_change']:.5g}; max row log-density change={fit['nts_max_logpdf_change']:.5g}")
                    print('\nFitted ' + ('scatter correlation (Pearson correlation undefined)' if fit['covariance'] is None else 'return correlation') + ':')
                    print(pd.DataFrame(corr, index=order, columns=order).to_string(float_format=lambda v: f'{v:.3f}'))
            if args.best_asset_order:
                select_best_orders(window_fits)
                for fit in window_fits:
                    if not fit['selected_for_display'] or fit['model'] not in ORDER_DEPENDENT_MODELS or 'scatter' not in fit:
                        continue
                    print('\nSelected fit: '+fit['fit_label'])
                    print('Location: '+', '.join(f'{s}={v:.6g}' for s,v in zip(fit['symbols'],fit['location'])))
                    print('Cholesky-coordinate log-skewness: '+', '.join(f'{s}={v:.6g}' for s,v in zip(fit['symbols'],fit['skewness'])))
                    corr = fit.get('correlation') or fit['scatter_correlation']
                    print('\nFitted '+('return correlation:' if fit['covariance'] is not None else 'scatter correlation (Pearson correlation undefined):'))
                    print(pd.DataFrame(corr,index=fit['symbols'],columns=fit['symbols']).to_string(float_format=lambda v:f'{v:.3f}'))
            if weights is not None:
                report=risk_report(window_fits,sample,weights,levels,allow_log=args.allow_log_linear_combination,
                                   simulations=args.simulations,seed=args.seed,batches=args.mc_batches)
                shown = report
                if args.best_asset_order:
                    shown = report.loc[report.selected_for_display.fillna(True).eq(True)].copy()
                print_risk_report(shown,levels)
                reports.append(report)
            marginal_cache = None
            if args.copulas:
                from .joint_copula import fit_block
                metadata = dict(window=window, subperiods=count, block=block, first_date=first, last_date=last,
                    observations=len(sample), return_type=args.return_type, return_scale=args.return_scale,
                    vol_standardization=normalization, vol_lambda=decay)
                cr, cs, ca, risk, marginal_cache = fit_block(sample, args, metadata,
                    scales=vol_scales.loc[sample.index] if normalization == 'ewma' else None,
                    next_scale=next_vol if normalization == 'ewma' else None, weights=weights)
                copula_records.extend(cr); copula_summaries.extend(cs); copula_audit.extend(ca); copula_risks.extend(risk)
            if mapped:
                univariate_extra = ({'vol_scales': vol_scales.loc[sample.index]} if normalization == 'ewma' else {})
                tables = fit_univariate_sample(sample, mapped, window,
                    location=args.location, max_iterations=args.max_iterations, return_type=args.return_type,
                    fit_timeout=args.fit_timeout, cached_fits=marginal_cache, **univariate_extra)
                if args.standardize_vol:
                    for table in tables:
                        table['vol_standardization'] = normalization
                        table['vol_lambda'] = np.nan if decay is None else decay
                univariate_tables.extend(tables)
                for table in tables:
                    table['return_scale'] = args.return_scale
                    table['subperiods'] = count
                    table['block'] = block
                if metrics:
                    for table in tables:
                        symbol = table.iloc[0]['symbol']
                        print(f'\nUnivariate distribution comparisons: {symbol}; {window or "all"} periods')
                        rows = compare_univariate(table, metrics)
                        distance_rows.extend(dict(r, scope='univariate', symbol=symbol, window=window or 'all',
                            first_date=first, last_date=last, observations=len(sample), return_type=args.return_type,
                            vol_standardization=normalization, vol_lambda=decay, subperiods=count, block=block) for r in rows)
        if args.js_distance:
            for window, count, block in dict.fromkeys((f['window'], f['subperiods'], f['block']) for f in results):
                window_fits = [f for f in results if (f['window'], f['subperiods'], f['block']) == (window, count, block)]
                first_fit = window_fits[0]
                js_options = {'display_selected': True} if args.best_asset_order else {}
                if ewma_enabled:
                    print('\nJS in scaled return units: raw unconditional laws versus next-period rescaled EWMA laws.')
                    js_options['return_units'] = True
                rows = compare_joint(window_fits, args.simulations, args.seed, args.mc_batches, **js_options)
                distance_rows.extend(dict(r, scope='joint', symbol='', window=window or 'all',
                    first_date=first_fit['first_date'], last_date=first_fit['last_date'], subperiods=count, block=block,
                    observations=first_fit['observations'], return_type=args.return_type) for r in rows)
        keys = ['model', 'window', 'observations', 'dimensions', 'parameters', 'first_date', 'last_date', 'loglik', 'aic', 'bic', 'df', 'q', 'vg_shape', 'power', 'lambda', 'psi', 'status', 'fit_sec', 'error', 'return_scale']
        if args.max_components > 1: keys.append('components')
        keys += ['subperiods', 'block']
        if args.standardize_vol:
            keys += ['vol_standardization', 'vol_lambda', 'vol_warmup', 'vol_floor', 'parameter_units', 'likelihood_units']
        if args.asset_orders == 'all': keys += ['asset_order']
        if args.best_asset_order: keys += ['selected_for_display']
        if any(r['model'] in {'nts-symmetric','nts-skewed'} for r in results): keys += ['nts_alpha','lam','nts_audit_passed']
        if any(r['model'] in {'generalized-t', 'generalized-t-skewed'} for r in results):
            keys.append('tail_index')
        if any(r['model'] in LAPLACE_MIXTURE_MODELS for r in results):
            keys+=['location_fixed','location_source','location_method','estimation_method','two_stage_aic','two_stage_bic','conditional_parameters','fit_restriction']
        summary = pd.DataFrame([{k: r.get(k) for k in keys} for r in results])
        summary = criterion_ranks(summary, rank_groups(summary))
        print('\nJoint fit comparison:')
        shown_summary = summary
        if args.best_asset_order:
            shown_summary = summary.loc[summary.selected_for_display].drop(columns='selected_for_display').copy()
        if args.max_components > 1:
            shown_summary = shown_summary.loc[shown_summary.components.eq(1)].drop(columns='components').copy()
        shown_summary = criterion_ranks(shown_summary, rank_groups(shown_summary))
        print(format_fit_comparison(shown_summary))
        if args.max_components > 1:
            mixture_columns = ['model', 'components', 'window', 'subperiods', 'block', 'first_date', 'last_date', 'observations',
                               'vol_standardization', 'vol_lambda', 'parameters', 'loglik', 'aic', 'bic', 'status', 'fit_sec', 'error']
            mixture_summary = summary.loc[summary.model.isin(mixture_models),
                                          [c for c in mixture_columns if c in summary]].copy()
            mixture_summary = rank_mixture_comparison(mixture_summary)
            print('\nFinite mixture comparison (single-component baselines reused, not refitted):')
            print('AIC/BIC ranks within each window, across component counts and volatility settings: *_rank_family within distribution; *_rank_all across distributions. Every finite criterion is included, regardless of convergence; ties share the minimum rank. Check status.')
            print(format_fit_comparison(mixture_summary.rename(columns={'parameters': '#param'})))
        print('\nAggregate multivariate fit comparison (single-component and mixture fits):')
        print('AIC/BIC ranks within each window across all displayed families, component counts, and volatility settings. Every finite criterion is included, regardless of convergence; check status. Ties share the minimum rank.')
        aggregate_summary = aggregate_fit_comparison(summary)
        print(format_fit_comparison(aggregate_summary.rename(columns={'parameters': '#param'})))
        if not args.no_save:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
            summary.to_csv(summary_path, index=False)
            if copula_records:
                from .copula_cli import clean_json
                copula_paths['fits'].write_text(json.dumps(clean_json(copula_records), indent=2, allow_nan=False), encoding='utf-8')
                pd.DataFrame(copula_summaries).to_csv(copula_paths['summary'], index=False)
                pd.DataFrame(copula_audit).to_csv(copula_paths['marginals'], index=False)
                if copula_risks: pd.DataFrame(copula_risks).to_csv(copula_paths['risk'], index=False)
                for key, path in copula_paths.items():
                    if key != 'risk' or copula_risks: print(f'Wrote {path}')
            print(f'Wrote {args.output}\nWrote {summary_path}')
            if args.max_components > 1:
                mixture_summary.to_csv(mixture_path, index=False)
                print(f'Wrote {mixture_path}')
            if metrics:
                pd.DataFrame(distance_rows).to_csv(distance_path, index=False)
                print(f'Wrote {distance_path}')
            if univariate_tables:
                pd.concat(univariate_tables, ignore_index=True).to_csv(univariate_path, index=False)
                print(f'Wrote {univariate_path}')
            if reports:
                risk_path.parent.mkdir(parents=True,exist_ok=True)
                pd.concat(reports,ignore_index=True).to_csv(risk_path,index=False)
                print(f'Wrote {risk_path}')
        if any(r['status'] != 'ok' for r in copula_summaries+copula_risks):
            print('Some copula fits or risk estimates failed; see separate copula results.')
        fitted_models = {r['model'] for r in results}
        if fitted_models & {'nts-symmetric','nts-skewed'}:
            print('NTS: shared mean-one tempered-stable mixing; covariance=scatter+(1-alpha)/lam*gamma*gamma\'. Location is not the mean when gamma is nonzero. All moments exist. Numerical MLE; failed quadrature audits are unranked. Bounds: alpha [0.1,0.9], lam [0.1,30]. Portfolio projections use univariate NTS.')
        if fitted_models & {'generalized-t', 'generalized-t-skewed'}:
            print('Generalized t: R**power/q ~ BetaPrime(dimension/power,q); tail index=power*q. Mean requires index>1; covariance requires index>2. General portfolio projections retain the original dimension. Bounds: power [0.25,10], q [0.1,1000]; large-q boundary may indicate GED limit.')
        if 'generalized-t-skewed' in fitted_models:
            print('Skewed generalized t: two-piece Cholesky-coordinate construction; asset ordering matters. Skewness bounds [-3,3]; positive values lengthen the positive side. Location is the mode, not the mean. Portfolio risk uses Monte Carlo, not a univariate generalized-t projection.')
        if fitted_models & {'ged-skewed', 'fs-skew-normal'}:
            print('Skewed GED/FS normal: two-piece Cholesky coordinates; asset ordering matters. GED power bounds [0.25,10]; FS normal fixes power=2. Skewness bounds [-3,3]; positive values lengthen the positive side. All moments exist; location is the mode, not the mean. Portfolio risk uses Monte Carlo. General multivariate marginals need not be univariate skewed GED.')
        if 'student-t' in fitted_models:
            print('Student-t scatter is not covariance; covariance is unavailable when df <= 2.')
        if summary.status.eq('boundary').any():
            print('Boundary fits are unranked in the detailed table; finite AIC/BIC values are included in mixture-table ranks.' if args.max_components > 1 else 'Boundary fits are unranked.')
        if any(r['model'] in LAPLACE_MIXTURE_MODELS for r in results):
            print('Laplace mixtures: W~Exp(1); origin is not the mean when gamma is nonzero. Externally fixed-location AIC/BIC exclude location parameters. Median/mode have descriptive two-stage scores only, not ordinary AIC/BIC ranks. Distinct from power-exponential laplace.')
        for prefix, label, convention in [
            ('hyperbolic', 'Hyperbolic', 'lambda=(dimension+1)/2'),
            ('nig', 'NIG', 'lambda=-1/2'),
            ('gh', 'Generalized hyperbolic', 'lambda estimated'),
        ]:
            if fitted_models.intersection({prefix+'-symmetric', prefix+'-skewed'}):
                note = f'{label} mixture: chi=1; {convention}.'
                if prefix+'-skewed' in fitted_models:
                    note += f' Skewed {label} location is not its mean.'
                print(note)
        if any(r['model']=='gh-skew-t' for r in results):
            print('GH skew-t: inverse-gamma mixture; nonzero skew requires df>2 for a finite mean and df>4 for covariance. ES depends on the projected loss-tail direction. Fit bounds: 0.25<=df<=200; boundary fits excluded.')
        if any(r['model'].startswith('variance-gamma-') for r in results):
            print('VG mixture: Gamma(shape=vg_shape, rate=vg_shape); E[W]=1. Restricted MLE: dimension/2+0.05 <= vg_shape <= 100; boundary fits excluded from portfolio risk.')
        if any(r['model'] in {'gh-symmetric','gh-skewed'} for r in results):
            print('Generalized GH estimates lambda in [-20,20]; boundary fits are unranked and excluded from portfolio risk.')
        if any(r['model'].startswith('sdb-') for r in results):
            print('Experimental SDB: numerically approximated likelihood; failed CDF audits are unranked. Portfolio risk uses simulation, not an ordinary univariate skew-t projection.')
        if any('slash' in r['model'] for r in results):
            print('Slash mixtures: location is outside random scaling. Mean/ES require q>1 and df>1 (t); covariance requires q>2 and df>2. Failed quadrature audits and boundary fits are unranked.')
        if 'laplace' in fitted_models:
            print('Laplace is elliptical power-exponential: exp(-sqrt(q)); power=1. Marginals need not be Laplace.')
        if 'ged' in fitted_models:
            print('GED is elliptical power-exponential: exp(-q^(power/2)). Marginals need not be GED.')
        print(f'Data elapsed: {data_elapsed:.3f} seconds\nFitting/output elapsed: {time.perf_counter()-start-data_elapsed:.3f} seconds\nOverall elapsed: {time.perf_counter()-start:.3f} seconds')
        return 0 if (summary.status.eq('ok').all() and all(r.status.eq('ok').all() for r in reports)
                     and all(t.status.eq('ok').all() for t in univariate_tables)
                     and all(r['status']=='ok' for r in distance_rows+copula_summaries+copula_risks)) else 1
    except (ValueError, KeyError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
