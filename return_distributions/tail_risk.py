"""Lower-return-tail VaR and ES, in positive-loss convention."""
import warnings
import numpy as np
from scipy import integrate, special, stats
from .projection import PointMass, ProjectedPower


def portfolio_risk(distribution, confidence=.99):
    """Return VaR=-Q(1-confidence), ES=-E[R | R<=Q]. No zero clipping.

    Normal/t use analytic formulas; elliptical power uses a radial partial
    moment integral; other projected families use a standardized tail integral.
    Numerical failures raise ValueError, distinct from mathematically infinite ES.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('Risk confidence must lie strictly between 0 and 1')
    if isinstance(distribution, PointMass):
        return dict(var=0., es=0., es_status='finite', es_method='point-mass')
    from .joint_slash import ProjectedSlash
    from .joint_generalized_t import ProjectedGeneralizedT
    from .joint_finite_mixture import ProjectedFiniteMixture
    if isinstance(distribution,(ProjectedSlash, ProjectedGeneralizedT, ProjectedFiniteMixture)):
        return distribution.risk(confidence)
    q = 1-confidence
    quantile = float(distribution.ppf(q))
    if not np.isfinite(quantile): raise ValueError('Nonfinite risk quantile')
    name = getattr(getattr(distribution, 'dist', None), 'name', '')
    if name == 'norm':
        es = -distribution.mean()+distribution.std()*stats.norm.pdf(stats.norm.ppf(q))/q
        method = 'analytic-normal'
    elif name == 't':
        df = distribution.args[0] if distribution.args else distribution.kwds['df']
        if df <= 1:
            return dict(var=-quantile, es=np.inf, es_status='infinite (df <= 1)', es_method='analytic-student-t')
        loc = distribution.kwds.get('loc', distribution.args[1] if len(distribution.args) > 1 else 0.)
        scale = distribution.kwds.get('scale', distribution.args[2] if len(distribution.args) > 2 else 1.)
        z = stats.t.ppf(q, df)
        es = -loc+scale*(df+z*z)/(df-1)*stats.t.pdf(z, df)/q
        method = 'analytic-student-t'
    elif name in {'johnsonsu', 'crystalball'}:
        from .scipy_extra import extra_es
        es = extra_es(distribution, q, quantile)
        method = name+'-partial-moment'
        if name == 'crystalball' and np.isposinf(es):
            values = dict(zip(['beta', 'm', 'loc', 'scale'], distribution.args))
            values.update(distribution.kwds)
            if values['m'] <= 2:
                return dict(var=-quantile, es=np.inf, es_status='infinite (m <= 2)', es_method=method)
    elif name == 'generalized_t':
        from .generalized_t import lower_first_moment
        values = dict(zip(['power', 'q', 'skewness', 'loc', 'scale'], distribution.args))
        values.update(distribution.kwds)
        p, shape_q, skew = values['power'], values['q'], values['skewness']
        if p*shape_q <= 1:
            return dict(var=-quantile, es=np.inf, es_status='infinite (power*q <= 1)', es_method='generalized-t-partial-moment')
        loc, scale = values.get('loc', 0.), values.get('scale', 1.)
        partial = lower_first_moment((quantile-loc)/scale, p, shape_q, skew)
        es = -loc-scale*partial/q
        method = 'generalized-t-partial-moment'
    elif name == 'nts':
        from .nts import invert
        keys=['alpha','lam','b','loc','scale']
        values=dict(zip(keys,distribution.args)); values.update(distribution.kwds)
        loc=values.get('loc',0.); scale=values.get('scale',1.)
        y=(quantile-loc)/scale
        partial=float(invert([y],values['alpha'],values['lam'],values['b'],shortfall=True)[0])
        es=-quantile+scale*partial/q
        method='nts-shifted-fourier-partial-moment'
    elif isinstance(distribution, ProjectedPower):
        d, p = distribution.dimension, distribution.power
        z = abs((quantile-distribution.location)/distribution.scale)
        logconstant = (special.gammaln(d/2)-.5*np.log(np.pi)-special.gammaln((d-1)/2)
                       +special.gammaln((d+1)/p)-special.gammaln(d/p))
        def integrand(theta):
            with np.errstate(over='ignore'):
                cutoff = (z/np.cos(theta))**p
            return np.exp(logconstant)*np.sin(theta)**(d-2)*np.cos(theta)*special.gammaincc((d+1)/p, cutoff)
        moment = checked_quad(integrand, 0, np.pi/2, q)
        es = -distribution.location+distribution.scale*moment/q
        method = 'radial-partial-moment-quadrature'
    else:
        # E[(Q-R)+] = integral_0^infinity t*f(Q-t)dt.
        # Rescale integration to avoid missing narrow daily-return densities.
        if name == 'gh_skew_t':
            df = distribution.args[0] if distribution.args else distribution.kwds['df']
            b = distribution.args[1] if len(distribution.args)>1 else distribution.kwds['b']
            if (b < 0 and df <= 2) or (b == 0 and df <= 1):
                return dict(var=-quantile,es=np.inf,es_status='infinite (polynomial loss tail)',es_method='gh-skew-t-tail')
            scale = float(distribution.kwds.get('scale',distribution.args[3] if len(distribution.args)>3 else 1.))
        elif name in {'azzalini_skew_t', 'nct'}:
            df = distribution.args[0] if distribution.args else distribution.kwds['df']
            if df <= 1:
                return dict(var=-quantile, es=np.inf, es_status='infinite (df <= 1)', es_method='skew-t-tail')
            scale = float(distribution.kwds.get('scale', distribution.args[3] if len(distribution.args) > 3 else 1.))
        else:
            scale = float(distribution.std())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Need finite positive scale for numerical ES')
        integral = checked_quad(lambda t: t*scale*distribution.pdf(quantile-scale*t), 0, np.inf, q)
        es = -quantile+scale*integral/q
        method = 'tail-density-quadrature'
    if not np.isfinite(es): raise ValueError('Numerical ES is nonfinite')
    return dict(var=-quantile, es=float(es), es_status='finite', es_method=method)


def checked_quad(function, lower, upper, tail_probability):
    tolerance = max(1e-13, tail_probability*1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('error', integrate.IntegrationWarning)
        try:
            value, error = integrate.quad(function, lower, upper, epsabs=tolerance, epsrel=1e-8, limit=250)
        except integrate.IntegrationWarning as exc:
            raise ValueError(f'ES quadrature failed: {exc}') from exc
    if not np.isfinite(value) or error > max(tolerance*10, abs(value)*1e-6):
        raise ValueError('ES quadrature did not meet error tolerance')
    return value
