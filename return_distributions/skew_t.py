"""Fernandez-Steel two-piece t and Azzalini-Capitanio skew-t.

Both use df, skewness, loc, scale. FS right scale is exp(skew), left is exp(-skew);
Azzalini skew is alpha. Zero skew gives the ordinary Student-t.
"""
import numpy as np
from scipy import integrate, special, stats


def central_stats(df, raw):
    m1, m2, m3, m4 = raw
    variance = m2-m1*m1
    with np.errstate(all='ignore'):
        skew = (m3-3*m1*m2+2*m1**3)/variance**1.5
        kurt = (m4-4*m1*m3+6*m1*m1*m2-3*m1**4)/variance**2-3
    return (np.where(df>1,m1,np.nan), np.where(df>2,variance,np.where(df>1,np.inf,np.nan)),
            np.where(df>3,skew,np.nan), np.where(df>4,kurt,np.nan))


class FSSkewT(stats.rv_continuous):
    def _argcheck(self, df, skew):
        return (df>0)&np.isfinite(df)&np.isfinite(skew)&(np.abs(skew)<100)

    def _fitstart(self, data, args=None):
        return (8., 0., float(np.median(data)), float(np.std(data)))

    def _logpdf(self, x, df, skew):
        z = x*np.exp(np.where(x<0,skew,-skew))
        return np.log(2)-np.logaddexp(skew,-skew)+stats.t.logpdf(z,df)

    def _pdf(self, x, df, skew): return np.exp(self._logpdf(x,df,skew))

    def _cdf(self, x, df, skew):
        left = special.expit(-2*skew)
        return np.where(x<0,2*left*stats.t.cdf(x*np.exp(skew),df),
                        1-2*(1-left)*stats.t.sf(x*np.exp(-skew),df))

    def _sf(self, x, df, skew): return self._cdf(-x,df,-skew)

    def _ppf(self, q, df, skew):
        left, right = special.expit(-2*skew), special.expit(2*skew)
        # Evaluate branches separately; the other branch can have invalid probabilities.
        q,df,skew,left,right = np.broadcast_arrays(q,df,skew,left,right)
        result = np.empty_like(q)
        mask = q < left
        result[mask] = np.exp(-skew[mask])*stats.t.ppf(q[mask]/(2*left[mask]),df[mask])
        result[~mask] = -np.exp(skew[~mask])*stats.t.ppf((1-q[~mask])/(2*right[~mask]),df[~mask])
        return result

    def _rvs(self, df, skew, size=None, random_state=None):
        positive = random_state.uniform(size=size)<special.expit(2*skew)
        magnitude = np.abs(random_state.standard_t(df,size=size))
        return np.where(positive,np.exp(skew)*magnitude,-np.exp(-skew)*magnitude)

    def _stats(self, df, skew):
        raw=[]
        with np.errstate(all='ignore'):
            for r in range(1,5):
                absolute=df**(r/2)*np.exp(special.gammaln((r+1)/2)-.5*np.log(np.pi))/special.poch((df-r)/2,r/2)
                factor=(np.exp((r+1)*skew)+(-1)**r*np.exp(-(r+1)*skew))/(np.exp(skew)+np.exp(-skew))
                raw.append(np.where(df>r,absolute*factor,np.nan))
        return central_stats(df,raw)


class AzzaliniSkewT(stats.rv_continuous):
    def _argcheck(self, df, skew): return (df>0)&np.isfinite(df)&np.isfinite(skew)

    def _fitstart(self, data, args=None): return (8.,0.,float(np.median(data)),float(np.std(data)))

    def _logpdf(self, x, df, skew):
        argument=skew*np.sqrt(df+1)*(x/np.hypot(x,np.sqrt(df)))
        return np.log(2)+stats.t.logpdf(x,df)+stats.t.logcdf(argument,df+1)

    def _pdf(self, x, df, skew): return np.exp(self._logpdf(x,df,skew))

    @staticmethod
    def _cdf_scalar(x, df, skew):
        if skew == 0: return stats.t.cdf(x,df)
        if x > 0: return 1-AzzaliniSkewT._cdf_scalar(-x,df,-skew)
        if df > 100:
            def integrand(t):
                arg=skew*np.sqrt(df+1)*(t/np.hypot(t,np.sqrt(df)))
                return 2*stats.t.pdf(t,df)*special.stdtr(df+1,arg)
            value,error=integrate.quad(integrand,-np.inf,x,epsabs=1e-11,epsrel=1e-9,limit=150)
            if error>1e-8: raise ValueError('Azzalini CDF quadrature failed tolerance')
            return np.clip(value,0.,1.)
        # t= sqrt(df)*tan(theta); finite-interval quadrature avoids infinite tails.
        upper=np.arctan2(np.sqrt(df),-x)
        logc=special.gammaln((df+1)/2)-.5*np.log(np.pi)-special.gammaln(df/2)
        def smooth(y):
            sinc=np.sin(y)/y if y else 1.
            return 2*np.exp(logc+(df-1)*np.log(sinc))*special.stdtr(df+1,-skew*np.sqrt(df+1)*np.cos(y))
        if df < 1:
            value,error=integrate.quad(smooth,0,upper,weight='alg',wvar=(df-1,0),epsabs=1e-11,epsrel=1e-9,limit=150)
        else:
            def density(y):
                return 2*np.exp(logc+(df-1)*np.log(np.sin(y)))*special.stdtr(df+1,-skew*np.sqrt(df+1)*np.cos(y))
            value,error=integrate.quad(density,0,upper,epsabs=1e-11,epsrel=1e-9,limit=150)
        if error>1e-8: raise ValueError('Azzalini CDF quadrature failed tolerance')
        return np.clip(value,0.,1.)

    def _cdf(self, x, df, skew):
        return np.vectorize(self._cdf_scalar,otypes=[float])(x,df,skew)

    def _sf(self, x, df, skew): return self._cdf(-x,df,-skew)

    def _rvs(self, df, skew, size=None, random_state=None):
        delta=skew/np.hypot(1,skew)
        sn=delta*np.abs(random_state.normal(size=size))+np.sqrt(1-delta**2)*random_state.normal(size=size)
        return sn/np.sqrt(random_state.chisquare(df,size=size)/df)

    def _stats(self, df, skew):
        delta=skew/np.hypot(1,skew)
        c=np.sqrt(2/np.pi)
        sn=[c*delta,np.ones_like(delta),c*delta*(3-delta**2),3*np.ones_like(delta)]
        raw=[]
        with np.errstate(all='ignore'):
            for r in range(1,5):
                factor=(df/2)**(r/2)/special.poch((df-r)/2,r/2)
                raw.append(np.where(df>r,sn[r-1]*factor,np.nan))
        return central_stats(df,raw)


fs_skew_t=FSSkewT(name='fs_skew_t',shapes='df, skewness')
azzalini_skew_t=AzzaliniSkewT(name='azzalini_skew_t',shapes='df, skewness')
CUSTOM_DISTS={'fernandez-steel':fs_skew_t,'fs-skew-t':fs_skew_t,'fs_skew_t':fs_skew_t,
              'azzalini':azzalini_skew_t,'azzalini-skew-t':azzalini_skew_t,'azzalini_skew_t':azzalini_skew_t}
