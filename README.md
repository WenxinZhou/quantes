# quantes (Quantile and Expected Shortfall Regression)
This package contains two main modules. The `linear` module implements convolution-smoothed quantile regression in both low and high dimensions, as well as joint linear quantile and expected shortfall (ES) regression. The `nonlinear` module contains three classes to implement joint nonlinear/nonparametric quantile and ES regression. This package is also avaiable at https://pypi.org/project/quantes/.


The `low_dim` class in the `linear` module applies a convolution smoothing approach to fit linear quantile regression models, known as *conquer*.  It also constructs normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients. The `high_dim` class fits sparse quantile regression models in high dimensions via *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) conquer methods. The IRW method is inspired by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://doi.org/10.1214/009053607000000802) for folded concave penalized estimation, exemplified by the SCAD penalty ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)) and the minimax concave penalty (MCP) ([Zhang, 2010](https://doi.org/10.1214/09-AOS729)). Computationally, each weighted l<sub>1</sub>-penalized conquer estimator is solved using the local adaptive majorize-minimization algorithm ([LAMM](https://doi.org/10.1214/17-AOS1568)). For comparison, the proximal ADMM algorithm ([pADMM](https://doi.org/10.1080/00401706.2017.1345703)) is implemented in the `admm` module. The `joint` class fits joint linear quantile and ES regression models ([Dimitriadis & Bayer, 2019](https://doi.org/10.1214/19-EJS1560); [Patton, Ziegel & Chen, 2019](https://doi.org/10.1016/j.jeconom.2018.10.008)) using either FZ loss minimization ([Fissler & Ziegel, 2016](https://doi.org/10.1214/16-AOS1439)) or two-step procedures ([Barendse, 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2937665); [Peng & Wang, 2023](https://onlinelibrary.wiley.com/doi/10.1002/sta4.619); [He, Tan & Zhou, 2023](https://doi.org/10.1093/jrsssb/qkad063)). For the second step of ES estimation, setting ``robust=TRUE`` uses the Huber loss with an adaptively chosen robustification parameter to gain robustness against heavy-tailed error/response; see [He, Tan & Zhou (2023)](https://doi.org/10.1093/jrsssb/qkad063) for more details. Moreover, a combination of the iteratively reweighted least squares (IRLS) algorithm and quadratic programming is used to compute non-crossing ES estimates. This ensures that the fitted ES does not exceed the fitted quantile at each observation.
Estimation and inference for sparse ES regression are currently under construction ([Zhang et al., 2023](https://arxiv.org/abs/2307.02695)). 


The `nonlinear` modules contains three classes, `KRR`, `LocPoly` and `FNN`, which implement three nonparametric joint quantile and expected shortfall estimators using kernel ridge regression ([Takeuchi et al., 2006](https://www.jmlr.org/papers/v7/takeuchi06a.html)), local polynomial regression ([Olma, 2021](https://arxiv.org/abs/2109.06150)), and feedforward neural network regression, respectively. For fitting nonparametric QR through the `qt()` method in both `KRR` and `FNN`, there is a `smooth` option available. When set to `TRUE`, it uses the Gaussian kernel convoluted check loss. For fitting nonparametric ES regression using (nonparametrically) generated surrogate response variables, the `es()` method in `FNN` provides two options: *squared loss* (`robust=FALSE`) and the *Huber loss* (`robust=TRUE`); the `res()` method in `KRR` fits robust ES kernel ridge regression using the Huber loss. Currently, the `es()` method in the `LocPoly` module uses only the squared loss to compute the two-step local polynomial ES regression estimator.


## Dependencies

```
python >= 3.9, numpy, matplotlib, scipy, scikit-learn, cvxopt, qpsolvers, torch
optional: osqp
```


## Examples

```
import numpy as np
import numpy.random as rgt
from scipy.stats import t
from quantes.linear import low_dim, high_dim
from quantes.admm import proximal
```
Generate data from a linear model with random covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The itercept is 4, and all the `p` regression coefficients are set as 1 in magnitude. The errors are generated from the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom), centered by subtracting the population *&tau;*-quantile of *t<sub>2</sub>*. 

When `p < n`, the module `low_dim` constains functions for fitting linear quantile regression models with uncertainty quantification. If the bandwidth `h` is unspecified, the default value *max\{0.01, \{&tau;(1- &tau;)\}^0.5 \{(p+log(n))/n\}^0.4\}* is used. The default kernel function is ``Laplacian``. Other choices are ``Gaussian``, ``Logistic``, ``Uniform`` and ``Epanechnikov``.

```
n, p = 8000, 400
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

qr = low_dim(X, Y, intercept=True).fit(tau=tau)
qr.keys()
# qr['beta'] : conquer estimate (intercept & slope coefficients)
# qr['res'] : n-vector of fitted residuals
# qr['bw'] : bandwidth
# qr['message']: convergence message 
```

At each quantile level *&tau;*, the `norm_ci` and `boot_ci` methods provide four 100* (1-alpha)% confidence intervals (CIs) for regression coefficients: (i) normal distribution calibrated CI using estimated covariance matrix, (ii) percentile bootstrap CI, (iii) pivotal bootstrap CI, and (iv) normal-based CI using bootstrap variance estimates. For multiplier/weighted bootstrap implementation with `boot_ci`, the default weight distribution is ``Exponential``. Other choices are ``Rademacher``, ``Multinomial`` (Efron's nonparametric bootstrap), ``Gaussian``, ``Uniform`` and ``Folded-normal``. The latter two require a variance adjustment; see Remark 4.7 in [Paper](https://doi.org/10.1016/j.jeconom.2021.07.010).

```
n, p = 500, 20
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

qr = low_dim(X, Y, intercept=True)
ci1 = qr.norm_ci(tau=tau)
ci2 = qr.mb_ci(tau=tau)

# ci1['normal'] : p+1 by 2 numpy array of normal CIs based on estimated asymptotic covariance matrix
# ci2['percentile'] : p+1 by 2 numpy array of bootstrap percentile CIs
# ci2['pivotal'] : p+1 by 2 numpy array of bootstrap pivotal CIs
# ci2['normal'] : p+1 by 2 numpy array of normal CIs based on bootstrap variance estimates
```

The module `high_dim` contains functions that fit high-dimensional sparse quantile regression models through the LAMM algorithm. The default bandwidth value is *max\{0.05, \{&tau;(1- &tau;)\}^0.5 \{ log(p)/n\}^0.25\}*. To choose the penalty level, the `self_tuning` method implements the simulation-based approach proposed by [Belloni & Chernozhukov (2011)](https://doi.org/10.1214/10-AOS827). 
The `l1` and `irw` methods compute *L<sub>1</sub>*- and IRW-*L<sub>1</sub>*-penalized conquer estimators, respectively. For the latter, the default concave penality is `SCAD` with constant `a=3.7` ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)). Given a sequence of penalty levels, the solution paths can be computed by `l1_path` and `irw_path`. 

```
p, n = 1028, 256
tau = 0.8
itcp, beta = 4, np.zeros(p)
beta[:15] = [1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.6]

X = rgt.normal(0, 1, size=(n,p))
Y = itcp + X@beta  + rgt.standard_t(2,size=n) - t.ppf(tau,df=2)

sqr = high_dim(X, Y, intercept=True)
lambda_max = np.max(sqr.tuning(tau))
lambda_seq = np.linspace(0.25*lambda_max, lambda_max, num=20)

## l1-penalized conquer
l1_model = sqr.l1(tau=tau, Lambda=0.75*lambda_max)

## iteratively reweighted l1-penalized conquer (default is SCAD penality)
irw_model = sqr.irw(tau=tau, Lambda=0.75*lambda_max)

## solution path of l1-penalized conquer
l1_path = sqr.l1_path(tau=tau, lambda_seq=lambda_seq)

## solution path of irw-l1-penalized conquer
irw_path = sqr.irw_path(tau=tau, lambda_seq=lambda_seq)

## model selection via bootstrap
boot_model = sqr.boot_select(tau=tau, Lambda=0.75*lambda_max, weight="Multinomial")
print('selected model via bootstrap:', boot_model['majority_vote'])
print('true model:', np.where(beta!=0)[0])
```

The module `admm` contains two classes, `proximal` and `ncvx`. The former employs the proximal ADMM algorithm to solve weighted *L<sub>1</sub>*-penalized quantile regression (without smoothing).
```
beta_true = np.insert(beta, 0, itcp)
lambda_max = np.max(high_dim(X, Y, intercept=True).tuning(tau))
lambda_seq = np.linspace(0.25*lambda_max, lambda_max, num=20)
admm = proximal(X, Y, intercept=True)

## l1-penalized QR
l1_admm = admm.l1(tau=tau, Lambda=0.5*lambda_max)
print(np.mean((l1_admm['beta'] - beta_true)**2).round(5))
print()

## iteratively reweighted l1-penalized QR (default is SCAD penality)
irw_admm = admm.irw(tau=tau, Lambda=0.75*lambda_max)
print(np.mean((irw_admm['beta'] - beta_true)**2).round(5))
print()

## solution path of l1-penalized QR
l1_admm_path = admm.l1_path(tau=tau, lambda_seq=lambda_seq)
print(np.mean((l1_admm_path['beta_seq'] - beta_true.reshape(-1,1))**2, axis=0).round(5))
print()

## solution path of irw-l1-penalized QR
irw_admm_path = admm.irw_path(tau=tau, lambda_seq=lambda_seq)
print(np.mean((irw_admm_path['beta_seq'] - beta_true.reshape(-1,1))**2, axis=0).round(5))
print()
```

The `joint` class in `quantes.linear` contains functions that fit joint (linear) quantile and expected shortfall models. The `fz` method computes joint quantile and ES regression estimates based on FZ loss minimization ([Fissler & Ziegel, 2016](https://doi.org/10.1214/16-AOS1439)). The `twostep` method implements two-stage procesures to compute quantile and ES regression estimates, with the ES part depending on a user-specified `loss`. Options are ``L2``, ``TrunL2``, ``FZ`` and ``Huber``. The `nc` method computes non-crossing counterparts of the ES estimates when `loss` = `L2` or `Huber`.

```
import numpy as np
import pandas as pd
import numpy.random as rgt
from quantes.linear import joint

p, n = 10, 5000
tau = 0.1
beta  = np.ones(p)
gamma = np.r_[0.5*np.ones(2), np.zeros(p-2)]

X = rgt.uniform(0, 2, size=(n,p))
Y = 2 + X @ beta + (X @ gamma) * rgt.normal(0, 1, n)

lm = joint(X, Y)
## two-step least squares
m1 = lm.twostep(tau=tau, loss='L2')

## two-step truncated least squares
m2 = lm.twostep(tau=tau, loss='TrunL2')

## two-step FZ loss minimization
m3 = lm.twostep(tau=tau, loss='FZ', G2_type=1)

## two-step adaptive Huber 
m4 = lm.twostep(tau=tau, loss='Huber')

## non-crossing two-step least squares
m5 = lm.nc(tau=tau, loss='L2')

## non-crossing two-step adaHuber
m6 = lm.nc(tau=tau, loss='Huber')

## joint quantes regression via FZ loss minimization (G1=0)
m7 = lm.fz(tau=tau, G1=False, G2_type=1, refit=False)

## joint quantes regression via FZ loss minimization (G1(x)=x)
m8 = lm.fz(tau=tau, G1=True, G2_type=1, refit=False)

out = pd.DataFrame(np.c_[(m1['coef_e'], m2['coef_e'], m3['coef_e'], m4['coef_e'], 
                          m5['coef_e'], m6['coef_e'], m7['coef_e'], m8['coef_e'])], 
                   columns=['L2', 'TLS', 'FZ', 'AH', 'NC-L2', 'NC-AH', 'Joint0', 'Joint1'])
out
```

The `KRR` class in `quantes.nonlinear` contains functions to compute quantile and expected shortfall kernel ridge regression (KRR) estimators, respectively. Recast as a quadratic program ([Takeuchi et al., 2006](https://www.jmlr.org/papers/v7/takeuchi06a.html)), the `qt()` method computes quantile KRR using QP solvers from `qpsolvers` ([Caron et al., 2024](https://pypi.org/project/qpsolvers/)). Setting `smooth=TRUE`, a convolution-smoothed quantile KRR is computed by the BFGS algorithm using `scipy.optimize.minimize`. By plugging in estimated quantiles, the two-step ES KRR estimator and its robust counterpart (using the *Huber loss*) are computed by the `es()` method with `robust=FALSE` and `robust=TRUE`, respectively. The methods `ls()`, `qt()` and `es()` from the `FNN` class in `quantes.nonlinear` compute nonparametric mean, quantile and expected shortfall regression estimators using feedforward neural networks, respectively. 

```
import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from quantes.nonlinear import KRR, LocPoly

mean_fn = lambda x: np.cos(2*np.pi*(x[:,0])) \
                    + (1 + np.exp(-x[:,1]-x[:,2]))**(-1) + (1 + x[:,3] \
                    + x[:,4])**(-3) + (x[:,5] + np.exp(x[:,6]*x[:,7]))**(-1)
std_fn = lambda x: np.sin(np.pi*(x[:,0] + x[:,1])*0.5) \
                   + np.log(1 + (x[:,2]*x[:,3]*x[:,4])**2) \
                   + x[:,7]*(1 + np.exp(-x[:,5]-x[:,6]))**(-1)
n, p = 2048, 8
X = rgt.uniform(0, 1, (n, p))
Y = mean_fn(X) + std_fn(X)*rgt.normal(0, 1, n)
tau = 0.2
qt = norm.ppf(tau)
es = norm.expect(lambda x : (x if x < qt else 0))/tau

X_test = rgt.uniform(0, 1, (1024, p))
qt_test = mean_fn(X_test) + std_fn(X_test)*qt
es_test = mean_fn(X_test) + std_fn(X_test)*es

# kernel ridge regression
kr = KRR(X, Y, kernel='polynomial', 
         kernel_params={'degree': 3, 'gamma': 1, 'coef0': 1})
kr.qt(tau=tau, alpha=.5, solver='cvxopt')
qt_pred = kr.qt_predict(X_test)
print('Mean squared prediction error of quantile KRR:', np.mean((qt_test - qt_pred)**2))

kr.ES(tau=tau, alpha=2, x=X_test)
es_pred = kr.pred_e
print('Mean squared prediction error of ES KRR:', np.mean((es_test - es_pred)**2))

# local polynomial regression
lp = LocPoly(X, Y) # default kernel is Gaussian
qt_fit = lp.qt_predict(x0=X, tau=tau, bw=0.5, degree=1)
qt_pred = lp.qt_predict(x0=X_test, tau=tau, bw=0.5, degree=1)
print('Mean squared prediction error of local linear ES:', np.mean((qt_test - qt_pred)**2))

es_pred = lp.es(x0=X_test, tau=tau, bw=1, degree=1, fit_q=qt_fit)
print('Mean squared prediction error of local linear ES:', np.mean((es_test - es_pred)**2))
```


## References

Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. *J. Bus. Econ. Statist.* **39**(1) 338–357. [Paper](https://doi.org/10.1080/07350015.2019.1660177)

He, X., Tan, K. M. and Zhou, W.-X. (2023). Robust estimation and inference for expected shortfall regression with many regressors. *J. R. Stat. Soc. B.* **85**(4) 1223-1246. [Paper](https://doi.org/10.1093/jrsssb/qkad063)

He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2023). Smoothed quantile regression with large-scale inference. *J. Econom.* **232**(2) 367-388. [Paper](https://doi.org/10.1016/j.jeconom.2021.07.010)

Koenker, R. (2005). *Quantile Regression*. Cambridge University Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Pan, X., Sun, Q. and Zhou, W.-X. (2021). Iteratively reweighted *l<sub>1</sub>*-penalized robust regression. *Electron. J. Stat.* **15**(1) 3287-3348. [Paper](https://doi.org/10.1214/21-EJS1862)

Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. *J. R. Stat. Soc. B.*  **84**(1) 205-233. [Paper](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12485)

## License 

This package is released under the GPL-3.0 license.