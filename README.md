# ReturnDistributions
Fit many probability distributions from SciPy to asset returns and rank them. Run with `python xscipy_dist_returns.py`. To fit returns normalized by trailing exponentially weighted volatility set `normalize_vol_ewma = True`. Some distributions that generally fit returns well are the Johnson SU, Normal Inverse Gaussian, and Student's t. For VXX (which tracks VIX futures) and to a lesser extent SPY, distributions that allow for skew fit better. The canonical normal distribution fits the worst, because it is thin-tailed.
![Alt text](/spy_log_returns.png)
![Alt text](/vxx_log_returns.png)
```
prices file: spy_tlt_vxx.csv
return type: log
normalize_vol_ewma: False

  symbol     #obs   median     mean       sd     skew     kurt      min      max
     SPY     8226   0.0007   0.0004   0.0118  -0.2472  11.3842  -0.1159   0.1356

              name  k     loglik         aic         bic     ks   ks_p     skew      kurt     df      a       b       p      q   beta  kappa    loc  scale  fit_sec
         johnsonsu  4 26058.7336 -52109.4672 -52081.4070 0.0125 0.1491  -0.7175   20.6240    NaN 0.1115  1.0780     NaN    NaN    NaN    NaN 0.0016 0.0079   0.0659
      norminvgauss  4 26057.9726 -52107.9453 -52079.8851 0.0098 0.4051  -0.5320    7.5154    NaN 0.4231 -0.0486     NaN    NaN    NaN    NaN 0.0013 0.0075   5.5003
     genhyperbolic  5 26057.9731 -52105.9461 -52070.8709 0.0098 0.4078  -0.5300    7.4874    NaN 0.4235 -0.0485 -0.4948    NaN    NaN    NaN 0.0013 0.0075   5.6886
         jf_skew_t  4 26040.9151 -52073.8302 -52045.7700 0.0147 0.0565 -27.5217 1913.0402    NaN    NaN     NaN  1.2207 1.3441    NaN    NaN 0.0015 0.0068   0.1184
                 t  3 26030.7392 -52055.4783 -52034.4332 0.0159 0.0312      NaN       NaN 2.7299    NaN     NaN     NaN    NaN    NaN    NaN 0.0008 0.0069   0.4044
           gennorm  3 26020.3859 -52034.7719 -52013.7267 0.0163 0.0255   0.0000    4.1215    NaN    NaN     NaN     NaN    NaN 0.8924    NaN 0.0007 0.0065   0.0312
laplace_asymmetric  3 26006.7949 -52007.5898 -51986.5447 0.0192 0.0045  -0.1669    3.0186    NaN    NaN     NaN     NaN    NaN    NaN 1.0402 0.0010 0.0078   0.0163
           laplace  2 26001.1858 -51998.3717 -51984.3415 0.0167 0.0206   0.0000    3.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0007 0.0078   0.0104
         hypsecant  2 25887.4066 -51770.8132 -51756.7830 0.0312 0.0000   0.0000    2.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0007 0.0067   0.0215
          logistic  2 25731.7071 -51459.4141 -51445.3840 0.0468 0.0000   0.0000    1.2000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0007 0.0057   0.0141
              norm  2 24872.3587 -49740.7173 -49726.6872 0.0941 0.0000   0.0000    0.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0004 0.0118   0.0076

  symbol     #obs   median     mean       sd     skew     kurt      min      max
     TLT     5833   0.0005   0.0002   0.0091  -0.0197   3.3705  -0.0690   0.0725

              name  k     loglik         aic         bic     ks   ks_p    skew      kurt     df      a       b       p      q   beta  kappa    loc  scale  fit_sec
         johnsonsu  4 19342.8063 -38677.6126 -38650.9274 0.0127 0.3039 -0.2014    2.2886    NaN 0.1496  1.7589     NaN    NaN    NaN    NaN 0.0015 0.0134   0.0602
     genhyperbolic  5 19342.8191 -38675.6381 -38642.2817 0.0146 0.1648 -0.2497 4023.8535    NaN 0.1481 -0.1481 -2.9873    NaN    NaN    NaN 0.0008 0.0180   5.7937
                 t  3 19340.5051 -38675.0102 -38654.9963 0.0166 0.0805  0.0000    2.9928 6.0048    NaN     NaN     NaN    NaN    NaN    NaN 0.0002 0.0074   0.3458
      norminvgauss  4 19337.6525 -38667.3049 -38640.6198 0.0124 0.3245 -0.1760    1.8173    NaN 1.6941 -0.1292     NaN    NaN    NaN    NaN 0.0010 0.0117   3.0330
          logistic  2 19327.5153 -38651.0306 -38637.6880 0.0162 0.0911  0.0000    1.2000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0002 0.0049   0.0039
         hypsecant  2 19325.1759 -38646.3518 -38633.0092 0.0170 0.0669  0.0000    2.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0003 0.0058   0.0062
           gennorm  3 19306.3957 -38606.7914 -38586.7775 0.0170 0.0678  0.0000    1.2174    NaN    NaN     NaN     NaN    NaN 1.3348    NaN 0.0004 0.0094   0.0252
laplace_asymmetric  3 19251.5413 -38497.0826 -38477.0687 0.0385 0.0000 -0.4129    3.1140    NaN    NaN     NaN     NaN    NaN    NaN 1.1038 0.0015 0.0067   0.0145
         jf_skew_t  4 19244.9464 -38481.8928 -38455.2076 0.0207 0.0131 -8.6504 1411.1766    NaN    NaN     NaN  1.2972 1.3371    NaN    NaN 0.0005 0.0064   0.1020
           laplace  2 19235.5430 -38467.0860 -38453.7435 0.0395 0.0000  0.0000    3.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0005 0.0068   0.0076
              norm  2 19143.7511 -38283.5023 -38270.1597 0.0350 0.0000  0.0000    0.0000    NaN    NaN     NaN     NaN    NaN    NaN    NaN 0.0002 0.0091   0.0083

  symbol     #obs   median     mean       sd     skew     kurt      min      max
     VXX     1933  -0.0065  -0.0021   0.0455   1.3785   7.3475  -0.2312   0.3308

              name  k    loglik        aic        bic     ks   ks_p    skew      kurt     df       a      b       p      q   beta  kappa     loc  scale  fit_sec
      norminvgauss  4 3509.3395 -7010.6790 -6988.4116 0.0125 0.9178  1.4112    7.9889    NaN  0.6011 0.2121     NaN    NaN    NaN    NaN -0.0140 0.0318   1.0471
         johnsonsu  4 3508.7676 -7009.5353 -6987.2680 0.0148 0.7850  1.7252   17.1161    NaN -0.3880 1.1692     NaN    NaN    NaN    NaN -0.0183 0.0333   0.0205
     genhyperbolic  5 3509.7390 -7009.4780 -6981.6438 0.0123 0.9284  1.6311   10.3419    NaN  0.5828 0.2428 -0.7948    NaN    NaN    NaN -0.0140 0.0365   1.4609
laplace_asymmetric  3 3489.8411 -6973.6822 -6956.9817 0.0214 0.3368  0.7945    3.4260    NaN     NaN    NaN     NaN    NaN    NaN 0.8202 -0.0139 0.0297   0.0068
                 t  3 3480.9020 -6955.8040 -6939.1035 0.0299 0.0624     NaN       NaN 2.8162     NaN    NaN     NaN    NaN    NaN    NaN -0.0069 0.0274   0.0866
           gennorm  3 3464.4637 -6922.9273 -6906.2269 0.0396 0.0046  0.0000    3.6209    NaN     NaN    NaN     NaN    NaN 0.9351    NaN -0.0067 0.0277   0.0124
           laplace  2 3462.8789 -6921.7578 -6910.6242 0.0406 0.0033  0.0000    3.0000    NaN     NaN    NaN     NaN    NaN    NaN    NaN -0.0065 0.0307   0.0024
         jf_skew_t  4 3457.6075 -6907.2149 -6884.9476 0.0426 0.0018 -5.4832 1960.7800    NaN     NaN    NaN  0.9796 0.9906    NaN    NaN -0.0040 0.0250   0.0386
         hypsecant  2 3447.9305 -6891.8610 -6880.7273 0.0400 0.0041  0.0000    2.0000    NaN     NaN    NaN     NaN    NaN    NaN    NaN -0.0060 0.0263   0.0088
          logistic  2 3413.8759 -6823.7517 -6812.6180 0.0492 0.0002  0.0000    1.2000    NaN     NaN    NaN     NaN    NaN    NaN    NaN -0.0052 0.0224   0.0038
              norm  2 3231.3179 -6458.6358 -6447.5022 0.1063 0.0000  0.0000    0.0000    NaN     NaN    NaN     NaN    NaN    NaN    NaN -0.0021 0.0455   0.0025

Total runtime: 37.14 s
```
