"""Fit multiple SciPy real-line dists to 1D data; return ranked MLE
params and GOF metrics (loglik, AIC, BIC, KS)."""
import time

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------
# which groups to include
# ------------------------------------------------------------
pd.set_option('display.float_format', '{:.4f}'.format)
INCLUDE_CORE = True
INCLUDE_EXTRA = False   # set to False if you want the fast set only

# ------------------------------------------------------------
# distributions and their parameter names in scipy.stats.fit() order:
# [shape params..., 'loc', 'scale']
# ------------------------------------------------------------

CORE_DISTS = {
    # real-line support
    'norm':                 ['loc', 'scale'],
    'skewnorm':             ['a', 'loc', 'scale'],
    'laplace':              ['loc', 'scale'],
    'laplace_asymmetric':   ['kappa', 'loc', 'scale'],
    'logistic':             ['loc', 'scale'],
    'hypsecant':            ['loc', 'scale'],  # hyperbolic secant
#    'tukeylambda':          ['lam', 'loc', 'scale'],
    't':                    ['df', 'loc', 'scale'],  # Student's t
    'jf_skew_t':            ['p', 'q', 'loc', 'scale'],  # Jones & Faddy skew-t
    'gennorm':              ['beta', 'loc', 'scale'],  # generalized normal / GED
    'johnsonsu':            ['a', 'b', 'loc', 'scale'],  # Johnson SU
    'norminvgauss':         ['a', 'b', 'loc', 'scale'],  # Normal Inverse Gaussian
    'genhyperbolic':        ['p', 'a', 'b', 'loc', 'scale'],  # generalized hyperbolic

    # positive support (0, +inf)
    'lognorm':              ['s', 'loc', 'scale'],  # lognormal
    'loglaplace':           ['c', 'loc', 'scale'],
    'fisk':                 ['c', 'loc', 'scale'],  # log-logistic
    'betaprime':            ['a', 'b', 'loc', 'scale'],
    'gamma':                ['a', 'loc', 'scale'],
#   'weibull_min':          ['c', 'loc', 'scale'],
    'invgauss':             ['mu', 'loc', 'scale'],  # inverse Gaussian
    'loggamma':             ['c', 'loc', 'scale'],
}

EXTRA_DISTS = {
    'nct':           ['df', 'nc', 'loc', 'scale'],  # non-central Student’s t
    'rdist':         ['df', 'loc', 'scale'],
    'kappa4':        ['h', 'k', 'loc', 'scale'],
    'levy_stable':   ['alpha', 'beta', 'loc', 'scale'],
}

# Distributions with support (0, +inf) to exclude unless return_type == "gross"
POSITIVE_SUPPORT_DISTS = {
    'lognorm', 'loglaplace', 'fisk', 'betaprime', 'gamma', 'weibull_min', 'invgauss', 'loggamma'
}

def build_dist_catalog(include_core=True, include_extra=True, max_dist=None):
    d = {}
    if include_core:
        d.update(CORE_DISTS)
    if include_extra:
        d.update(EXTRA_DISTS)
    if max_dist is not None:
        d = dict(list(d.items())[:max_dist])
    return d

# ------------------------------------------------------------
# moment helpers
# ------------------------------------------------------------
def _skew_kurt_analytic(dist, params):
    """Try analytic skew and excess kurtosis via SciPy .stats(); return (s, k_excess) or (np.nan, np.nan)."""
    with np.errstate(all='ignore'):
        try:
            m, v, s, k = dist.stats(*params, moments='mvsk')
        except Exception:
            return np.nan, np.nan
    if np.all(np.isfinite([m, v, s, k])) and v > 0:
        return float(s), float(k)  # SciPy returns excess kurtosis for 'k'
    return np.nan, np.nan

def _skew_kurt_numeric(dist, params, eps=1e-6, n=4096):
    """Numerical fallback via quantile integration on (eps, 1-eps). Returns (skew, excess_kurtosis)."""
    u = np.linspace(eps, 1.0 - eps, n)
    with np.errstate(all='ignore', under='ignore', over='ignore', invalid='ignore'):
        xq = dist.ppf(u, *params)
    xq = xq[np.isfinite(xq)]
    if xq.size < 10:
        return np.nan, np.nan
    mu = float(np.mean(xq))
    c = xq - mu
    m2 = float(np.mean(c**2))
    if not np.isfinite(m2) or m2 <= 0:
        return np.nan, np.nan
    m3 = float(np.mean(c**3))
    m4 = float(np.mean(c**4))
    skew = m3 / (m2**1.5)
    kurt_ex = m4 / (m2**2) - 3.0
    return (float(skew) if np.isfinite(skew) else np.nan,
            float(kurt_ex) if np.isfinite(kurt_ex) else np.nan)

def _skew_kurt(dist, params):
    """Best-effort skew and excess kurtosis: analytic, else numeric fallback."""
    s, k = _skew_kurt_analytic(dist, params)
    if not np.isfinite(s) or not np.isfinite(k):
        s, k = _skew_kurt_numeric(dist, params)
    return s, k

# ------------------------------------------------------------
# fitting helpers
# ------------------------------------------------------------
def fit_one(x: np.ndarray, name: str, param_names: list[str]) -> dict:
    """Fit a single scipy.stats distribution by MLE and compute GOF metrics + moments."""
    dist = getattr(stats, name)
    params = dist.fit(x)         # tuple: shape..., loc, scale

    # log-likelihood and ICs
    ll = float(np.sum(dist.logpdf(x, *params)))
    n = x.size
    k = len(params)
    aic = 2.0 * k - 2.0 * ll
    bic = np.log(n) * k - 2.0 * ll

    # KS
    ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)

    # Moments (skew, excess kurt)
    if name == 't':
        nu = params[0]
        skew = 0.0 if nu > 3.0 else np.nan
        kurt = 6.0 / (nu - 4.0) if nu > 4.0 else np.nan
    else:
        skew, kurt = _skew_kurt(dist, params)

    out = {
        'name': name, 'n': n, 'k': k,
        'loglik': ll, 'aic': aic, 'bic': bic,
        'ks': float(ks_stat), 'ks_p': float(ks_p),
        'skew': float(skew) if np.isfinite(skew) else np.nan,
        'kurt': float(kurt) if np.isfinite(kurt) else np.nan,
    }
    for nm, val in zip(param_names, params):
        out[nm] = float(val)
    return out

def fit_many(x: np.ndarray, dists: dict[str, list[str]]) -> pd.DataFrame:
    """Fit all distributions in dists and return a tidy dataframe sorted by AIC."""
    rows = []
    for name, pnames in dists.items():
        t0 = time.perf_counter()
        try:
            row = fit_one(x, name, pnames)
            row['fit_sec'] = float(time.perf_counter() - t0)
            rows.append(row)
        except Exception as e:
            rows.append({'name': name, 'fit_sec': float(time.perf_counter() - t0), 'error': str(e)})
    df = pd.DataFrame(rows)

    # order columns: put skew/kurt BEFORE 'df'
    base_cols = ['name', 'n', 'k', 'loglik', 'aic', 'bic', 'ks', 'ks_p', 'skew', 'kurt']
    param_cols = [c for c in df.columns if c not in base_cols + ['error']]

    preferred = [c for c in [
        'df','a','b','p','q','alpha','beta','h','k','c','lam','nc','s','mu','kappa','loc','scale'
    ] if c in param_cols]
    rest = [c for c in param_cols if c not in preferred and c != 'fit_sec']
    cols = base_cols + preferred + sorted(rest)
    if 'fit_sec' in df.columns:
        cols.append('fit_sec')
    df = df[[c for c in cols if c in df.columns]]

    if 'aic' in df.columns:
        df_ok = df[df.get('aic').notna()].sort_values(['aic', 'bic'], ascending=[True, True], kind='mergesort')
        df_err = df[df.get('aic').isna()]
        df = pd.concat([df_ok, df_err], axis=0, ignore_index=True)
    return df.reset_index(drop=True)

# ------------------------------------------------------------
# plotting helper (KDE + top-N PDFs + always-plot list)
# ------------------------------------------------------------
def plot_top_densities(x: np.ndarray,
                       results_df: pd.DataFrame,
                       catalog: dict[str, list[str]],
                       title: str,
                       nplot_dist: int,
                       kde_bw=None,          # None (Scott) or float/callable per scipy.stats.gaussian_kde
                       ngrid: int = 512,
                       plot_dist_always=None,
                       log_scale: bool = False):
    """Plot KDE + PDFs of top nplot_dist fits by AIC, optionally on the log-density scale."""
    import matplotlib.pyplot as plt
    from scipy import stats as _st

    dfok = results_df[results_df.get('aic').notna()]
    res_top = dfok.head(nplot_dist)

    # ensure always-plot dists are included (if successfully fitted)
    if plot_dist_always:
        df_always = dfok[dfok['name'].isin(list(plot_dist_always))]
        res = pd.concat([res_top, df_always], ignore_index=True)
        res = res.drop_duplicates(subset='name', keep='first')
    else:
        res = res_top

    xfin = x[np.isfinite(x)]
    if xfin.size == 0:
        return

    lo = np.nanpercentile(xfin, 0.5)
    hi = np.nanpercentile(xfin, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.min(xfin), np.max(xfin)
    xx = np.linspace(lo, hi, ngrid)

    # KDE (and log KDE if requested)
    kde = _st.gaussian_kde(xfin, bw_method=kde_bw)
    yy_kde = kde(xx)
    if log_scale:
        yy_kde = np.log(np.clip(yy_kde, np.finfo(float).tiny, None))
    plt.figure(figsize=(8, 4))
    plt.plot(xx, yy_kde, linestyle='--', linewidth=2, label='log kde' if log_scale else 'kde')

    # PDFs or log-PDFs
    for _, row in res.iterrows():
        name = row['name']
        pnames = catalog.get(name, [])
        try:
            params = [row[p] for p in pnames if p in row and pd.notna(row[p])]
            dist = getattr(_st, name)
            if log_scale:
                yy = dist.logpdf(xx, *params)
            else:
                yy = dist.pdf(xx, *params)
            plt.plot(xx, yy, label=f'log {name}' if log_scale else name)
        except Exception as e:
            print(f"plot skip {name}: {e}")

    plt.title(title)
    plt.ylabel('log density' if log_scale else 'density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    t_script_start = time.perf_counter()

    scale_ret = 1.0
    prices_file = "spy_tlt_vxx.csv"
    max_stocks = None   # max number of symbols to include (besides Date)
    max_dist = None     # max number of dists to try
    nplot_dist = 5      # set to 0 to disable plotting
    nplot_log_dist = nplot_dist  # set to 0 to disable log-density plotting
    normalize_vol_ewma = False       # True: divide returns by EWMA vol before fitting
    ewma_lambda = 0.94              # RiskMetrics decay parameter (lambda)
    ewma_warmup = 100               # drop the first N normalized obs to allow warm-up
    # always plot these distributions, even if not in the top nplot_dist by AIC
    plot_dist_always = ['norm']

    return_type = "log"  # "log", "simple", or "gross"

    print("prices file:", prices_file)
    df = pd.read_csv(
        prices_file,
        parse_dates=["Date"],
        index_col="Date",
        usecols=None if max_stocks is None else range(max_stocks + 1)
    )

    if return_type == "log":
        df_ret = scale_ret * np.log(df / df.shift(1))
    elif return_type == "simple":
        df_ret = scale_ret * (df / df.shift(1) - 1)
    elif return_type == "gross":
        df_ret = scale_ret * (df / df.shift(1))
    else:
        raise ValueError("return_type must be one of: log, simple, gross")

    print("return type:", return_type)
    print("normalize_vol_ewma:", normalize_vol_ewma)
    if normalize_vol_ewma:
        print("ewma_lambda:", ewma_lambda)
        print("ewma_warmup:", ewma_warmup)

    catalog = build_dist_catalog(include_core=INCLUDE_CORE,
                                 include_extra=INCLUDE_EXTRA,
                                 max_dist=max_dist)

    # Exclude positive-support families unless we are fitting gross returns
    if return_type != "gross":
        catalog = {k: v for k, v in catalog.items() if k not in POSITIVE_SUPPORT_DISTS}
    elif normalize_vol_ewma:
        catalog = {k: v for k, v in catalog.items() if k not in POSITIVE_SUPPORT_DISTS}

    for col, series in df_ret.items():
        series_clean = series.dropna()

        if normalize_vol_ewma:
            if not (0.0 < ewma_lambda < 1.0):
                raise ValueError('ewma_lambda must be in (0, 1) when normalize_vol_ewma is True')
            alpha = 1.0 - ewma_lambda
            rets_for_vol = series_clean - 1.0 if return_type == "gross" else series_clean
            ewma_var = rets_for_vol.pow(2).ewm(alpha=alpha, adjust=False).mean()
            cond_vol = np.sqrt(ewma_var).replace(0.0, np.nan).shift(1)  # lag by one period
            series_fit = (rets_for_vol / cond_vol).replace([np.inf, -np.inf], np.nan)
            if ewma_warmup > 0:
                series_fit = series_fit.iloc[ewma_warmup:]
        else:
            series_fit = series_clean

        series_fit = series_fit.dropna()
        x = series_fit.to_numpy()

        if x.size == 0:
            print(f"\n{col:>8s}: no observations after preprocessing; skipping.")
            continue

        print("\n" + " ".join("%8s" % y for y in
              ("symbol", "#obs", "median", "mean", "sd", "skew", "kurt", "min", "max")))
        print("%8s" % col, "%8d" % len(x), " ".join("%8.4f" % y for y in
              (np.median(x), np.mean(x), np.std(x),
               stats.skew(x), stats.kurtosis(x), np.min(x), np.max(x))))

        results = fit_many(x, catalog)  # keep full cols for plotting
        print("\n" + results.drop(columns="n").to_string(index=False))

        valid_fits = int((results['aic'].notna()).sum())

        if nplot_dist and nplot_dist > 0 and valid_fits > 0:
            plot_top_densities(
                x=x,
                results_df=results,  # already sorted by AIC asc
                catalog=catalog,
                title=f"{col}: top {min(nplot_dist, valid_fits)} fits by AIC",
                nplot_dist=min(nplot_dist, valid_fits),
                plot_dist_always=plot_dist_always
            )

        if nplot_log_dist and nplot_log_dist > 0 and valid_fits > 0:
            plot_top_densities(
                x=x,
                results_df=results,
                catalog=catalog,
                title=f"{col}: top {min(nplot_log_dist, valid_fits)} fits by AIC (log density)",
                nplot_dist=min(nplot_log_dist, valid_fits),
                plot_dist_always=plot_dist_always,
                log_scale=True
            )

    total_elapsed = time.perf_counter() - t_script_start
    print(f"\nTotal runtime: {total_elapsed:.2f} s")
