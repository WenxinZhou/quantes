import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from scipy.optimize import minimize
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


from quantes.solvers import bbgd, lamm
from quantes.utils import (mad, find_root, boot_weight, concave_weight,
                           smooth_check, conquer_weight, AHuber_fn, AHuber_grad,
                           sparse_supp, sparse_proj,
                           G2)
from quantes.admm import proximal


###############################################################################
################## Convolution Smoothed Quantile Regression ###################
###############################################################################
class low_dim():
    '''
        Convolution Smoothed Quantile Regression (conquer)

    Methods:
        bw() : bandwidth selection.
        als() : asymmetric least squares or Huber regression.
        fit() : fit conquer via the GD-BB algorithm.
        bfgs_fit() : fit conquer via the BFGS algorithm.
        bw_path() : solution path of conquer at a sequence of bandwidths.
        norm_ci() : normal calibrated confidence intervals.
        mb() : multiplier bootstrap estimates.
        mb_ci() : multiplier bootstrap confidence intervals.
        Huber() : Huber regression via BFGS.
        adaHuber() : adaptive Huber regression.

    Attributes:
        kernels : built-in smoothing kernels.
        weights : built-in random weight distributions.
        params : internal statistical and optimization parameters.
    '''
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    weights = ["Exponential", "Multinomial", "Rademacher",
               "Gaussian", "Uniform", "Folded-normal"]
    params = {'init_lr': 1, 'max_lr': 50, 'max_iter': 1e3, 'tol': 1e-6,
              'warm_start': True, 'nboot': 200, 'min_bandwidth': 1e-4}

    def __init__(self, X, Y, intercept=True, options=dict()):
        '''
        Args:
            X: n by p matrix of covariates; each row is an observation vector.
            Y: an ndarray of response variables.
            intercept: logical flag for adding an intercept to the model.
            options: a dictionary of internal statistical and optimization parameters.
                max_iter: maximum numder of iterations in the GD-BB algorithm; 
                          default is 500.
                max_lr: maximum step size/learning rate. 
                        If set to False, there is no contraint on the maximum step size.
                tol: the iteration will stop when max{|g_j|: j = 1, ..., p} <= tol
                     where g_j is the j-th component of the (smoothed) gradient; 
                     default is 1e-4.
                warm_start: logical flag for using a robust expectile 
                            regression estimate as an initial value.
                nboot: number of bootstrap samples for inference.
                min_bandwidth: minimum bandwidth value; default is 1e-4.
        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept

        if intercept:
            self.X = np.c_[np.ones(self.n), X[:]]
            self.X1 = np.c_[np.ones(self.n), 
                            (X[:] - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X[:], X[:]/self.sdX

        self.params.update(options)


    def bandwidth(self, tau):
        n, p = self.X.shape
        h0 = min((p + np.log(n))/n, 0.5)**0.4
        return max(self.params['min_bandwidth'], 
                   h0 * (tau-tau**2)**0.5)


    def als(self, tau=0.5, robust=5,
            standardize=True, adjust=True, scale=False,
            solver='BB-GD', options=None):
        '''
            Asymmetric Least Squares/Huber Regression
        '''
        y = self.Y
        X = self.X1 if standardize else self.X
        n, p = X.shape  
        beta = np.zeros(p)
        if self.itcp: beta[0] = np.quantile(y, tau)

        if scale == True:
            asym = lambda x : np.where(x < 0, (1-tau) * x, tau * x)
            fn0 = lambda b: AHuber_fn(y - X@b, tau, 0)
            gd0 = lambda b: X.T@(-AHuber_grad(y - X@b, tau, 0)) / n
            if solver == 'BB-GD':
                model = bbgd(self.params)
                model.minimize(fn0, gd0, beta)
            elif solver == 'BFGS':
                model = minimize(fn0, beta, method='L-BFGS-B',
                                 jac=gd0, tol=self.params['tol'], 
                                 options=options)
            beta = model.x
            robust = robust * mad(asym(y - X@beta))

        fn = lambda b: AHuber_fn(y - X@b, tau, robust)
        gd = lambda b: X.T@(-AHuber_grad(y - X@b, tau, robust)) / n
        if solver == 'BB-GD':
            model = bbgd(self.params)
            model.minimize(fn, gd, beta)
        elif solver == 'BFGS':
            model = minimize(fn, beta, method='L-BFGS-B',
                             jac=gd, tol=self.params['tol'], 
                             options=options)
        
        beta = model.x
        res = y - X @ beta
        if standardize and adjust:
            beta[self.itcp:] = beta[self.itcp:]/self.sdX
            if self.itcp: beta[0] -= self.mX @ beta[1:]

        return {'beta': beta,
                'res': res,
                'robust': robust,
                'model': model}


    def fit(self, tau=0.5, h=None, kernel="Laplacian",
            beta0=np.array([]), weight=np.array([]), 
            standardize=True, adjust=True,
            solver='BB-GD', options=None):
        '''
            Convolution Smoothed Quantile Regression

        Args:
            tau: quantile level between 0 and 1; default is 0.5.
            h: bandwidth; the default value is computed by self.bandwidth(tau).
            kernel: a character string representing one of the 
                    built-in smoothing kernels; default is "Laplacian".
            beta0: initial estimate; default is np.array([]).
            weight: an ndarray of observation weights; default is np.array([]).
            standardize: logical flag for x variable standardization 
                         prior to fitting the model; default is TRUE.
            adjust: logical flag for returning coefficients on the original scale.
            solver: a character string representing the optimization algorithm;
                     default is 'BB-GD'.
            options : a dictionary of solver options. Default is 
                      options={'gtol': 1e-05, 'norm': inf, 'maxiter': None,
                               'disp': False, 'return_all': False}
                      gtol : gradient norm must be less than gtol(float) 
                             before successful termination.
                      norm : order of norm (Inf is max, -Inf is min).
                      maxiter : maximum number of iterations to perform.
                      disp : set to True to print convergence messages.
                      return_all : set to True to return a list of the 
                                   best solution at each of the iterations.

        Returns:
            'beta': conquer estimate.
            'bw': bandwidth.
        '''
        bw = self.bandwidth(tau) if h is None else h 
        if kernel not in self.kernels:
            raise ValueError('Kernel "{}" is not included'.format(kernel))
        y = self.Y
        X = self.X1 if standardize else self.X
        n, p = X.shape

        if not beta0.any():
            beta0 = np.zeros(p)
            if self.itcp: 
                beta0[0] = np.quantile(y, tau)

        func = lambda b : smooth_check(y-X@b, tau, bw, kernel, weight)
        grad = lambda b : X.T@(conquer_weight((X@b-y)/bw, tau, 
                                              kernel, weight))/n
        
        if solver == 'BB-GD':
            model = bbgd(self.params)
            model.minimize(func, grad, beta0)
        elif solver == 'BFGS':
            model = minimize(func, beta0, method='L-BFGS-B',
                             jac=grad, tol=self.params['tol'], 
                             options=options)
        
        beta = model.x
        res = y - X @ beta
        if standardize and adjust:
            beta[self.itcp:] = beta[self.itcp:]/self.sdX
            if self.itcp: beta[0] -= self.mX @ beta[1:]

        return {'beta': beta,
                'res': res,
                'bw': bw,
                'model': model,
                'message': model.message}


    def bw_path(self, tau=0.5, h_seq=np.array([]), L=20, 
                kernel="Laplacian", standardize=True, 
                adjust=True, solver='BB-GD'):
        '''
            Solution Path of Conquer at a Sequence of Bandwidths

        Args:
            tau : quantile level; default is 0.5.
            h_seq : a sequence of bandwidths.
            L : number of bandwdiths; default is 20.
            kernel : a character string representing one of the built-in
                     smoothing kernels; default is "Laplacian".
            standardize : logical flag for x variable standardization.
            adjust : logical flag for returning coefficients 
                     on the original scale.

        Returns:
            'beta_seq' : a sequence of conquer estimates.
            'res_seq' : a sequence of residual vectors.
            'bw_seq' : a sequence of bandwidths in descending order.
        '''

        n, p = self.X.shape
        if not np.array(h_seq).any():
            h_seq = np.linspace(0.01, min((p + np.log(n))/n, 0.5)**0.4, num=L)

        X = self.X1 if standardize else self.X
        h_seq, L = np.sort(h_seq)[::-1], len(h_seq)
        beta_seq = np.empty(shape=(X.shape[1], L))
        res_seq = np.empty(shape=(n, L))
        model = self.fit(tau, h_seq[0], kernel, 
                         standardize=standardize, 
                         adjust=False, solver=solver)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']

        for l in range(1,L):      
            model = self.fit(tau, h_seq[l], kernel, 
                             model['beta'], model['res'],
                             standardize=standardize, 
                             adjust=False, solver=solver)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
    
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp:
                beta_seq[0,:] -= self.mX @ beta_seq[1:,]

        return {'beta_seq': beta_seq, 
                'res_seq': res_seq, 
                'bw_seq': h_seq}

        
    def norm_ci(self, tau=0.5, h=None, kernel="Laplacian", 
                solver='BB-GD', alpha=0.05, standardize=True):
        '''
            Normal Calibrated Confidence Intervals
        
        Args:
            tau : quantile level; default is 0.5.
            h : bandwidth. The default is computed by self.bandwidth(tau).
            kernel : a character string representing one of the built-in 
                     smoothing kernels; default is "Laplacian".
            solver : a character string representing the method for 
                     computing the estimate; default is 'BB-GD'.
            alpha : miscoverage level for each CI; default is 0.05.
            standardize : logical flag for x variable standardization 
                          prior to fitting the model; default is TRUE.

        Returns:
            'beta' : conquer estimate.
            'normal' : numpy array. Normal CIs based on estimated 
                          asymptotic covariance matrix.
        '''
        if h is None: h = self.bandwidth(tau)
        X = self.X
        if solver=='BFGS':
            model = self.fit(tau, h, kernel, solver='BFGS')
        else:
            model = self.fit(tau, h, kernel, 
                             standardize=standardize)
        h = model['bw']
        hess_weight = norm.pdf(model['res']/h)
        grad_weight = ( norm.cdf(-model['res']/h) - tau)**2
        hat_V = (X.T * grad_weight) @ X/self.n
        inv_J = np.linalg.inv((X.T * hess_weight) @ X/(self.n * h))
        ACov = inv_J @ hat_V @ inv_J
        rad = norm.ppf(1-0.5*alpha)*np.sqrt( np.diag(ACov) / self.n )        
        ci = np.c_[model['beta'] - rad, model['beta'] + rad]

        return {'beta': model['beta'], 'normal': ci}


    def mb(self, tau=0.5, h=None, kernel="Laplacian", 
           weight="Exponential", standardize=True, solver='BB-GD'):
        '''
            Multiplier Bootstrap Estimates

        Args:
            tau : quantile level; default is 0.5.
            h : bandwidth. The default is computed by self.bandwidth(tau).
            kernel : a character string representing one of the built-in 
                     smoothing kernels; default is "Laplacian".
            weight : a character string representing weight distribution;
                     default is "Exponential".
            standardize : logical flag for x variable standardization 
                          prior to fitting the model; default is TRUE.

        Returns:
            'mb_beta' : numpy array. 
                        1st column: conquer estimate; 
                        2nd to last: bootstrap estimates.
        '''
        h = self.bandwidth(tau) if h is None else h
        
        if weight not in self.weights:
            raise ValueError('Distribution "{}" is not included'.format(weight))

        mdl = self.fit(tau, h, kernel, standardize=standardize, 
                       adjust=False, solver=solver)
        mb_beta = np.empty([len(mdl['beta']), self.params['nboot']+1])
        mb_beta[:,0] = np.copy(mdl['beta'])

        for b in range(self.params['nboot']):
            mdl = self.fit(tau, h, kernel, beta0=mb_beta[:,0],
                           weight=boot_weight(weight)(self.n),
                           standardize=standardize, solver=solver)
            mb_beta[:,b+1] = mdl['beta']

        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX @ mb_beta[1:,0]

        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]

        return mb_beta


    def mb_ci(self, tau=0.5, h=None, kernel="Laplacian", 
              weight="Exponential", alpha=0.05, 
              standardize=True, solver='BB-GD'):
        '''
            Multiplier Bootstrap Confidence Intervals

        Arguments
        ---------
        tau : quantile level; default is 0.5.
        h : bandwidth. The default is computed by self.bandwidth(tau).
        kernel : a character string representing one of the built-in 
                 smoothing kernels; default is "Laplacian".
        weight : a character string representing weight distribution;
                 default is "Exponential".
        alpha : miscoverage level for each CI; default is 0.05.
        standardize : logical flag for x variable standardization 
                      prior to fitting the model; default is TRUE.

        Returns
        -------
        'boot_beta' : numpy array. 
                      1st column: conquer estimate; 
                      2nd to last: bootstrap estimates.
        'percentile' : numpy array. Percentile bootstrap CI.
        'pivotal' : numpy array. Pivotal bootstrap CI.
        'normal' : numpy array. Normal-based CI using bootstrap variance estimates.
        '''
        if h==None: h = self.bandwidth(tau)
        
        mb_beta = self.mb(tau, h, kernel, weight, standardize, solver)
        if weight in self.weights[:4]:
            adj = 1
        elif weight == 'Uniform':
            adj = np.sqrt(1/3)
        elif weight == 'Folded-normal':
            adj = np.sqrt(0.5*np.pi - 1)

        per = np.c_[np.quantile(mb_beta[:,1:], 0.5*alpha, axis=1),
                    np.quantile(mb_beta[:,1:], 1-0.5*alpha, axis=1)]
        piv = np.c_[(1+1/adj)*mb_beta[:,0] - per[:,1]/adj,
                    (1+1/adj)*mb_beta[:,0] - per[:,0]/adj] 

        radi = norm.ppf(1-0.5*alpha)*np.std(mb_beta[:,1:], axis=1)/adj
        clt = np.c_[mb_beta[:,0] - radi, mb_beta[:,0] + radi]

        return {'boot_beta': mb_beta, 
                'percentile': per,
                'pivotal': piv,
                'normal': clt}


    def Huber(self, c=1, beta0=np.array([]), tol=1e-6, options=None):
        '''
            Huber Regression via BFGS

        Args:
            c : robustness parameter; default is 1.
            beta0 : initial estimate; default is np.array([]).
            tol : tolerance for termination.
            options : a dictionary of solver options. Default is 
                      {'gtol': 1e-05, 'norm': inf, 'maxiter': None, 
                       'disp': False, 'return_all': False}
        '''

        y, X = self.Y, self.X

        huber_loss = lambda u : \
            np.where(abs(u)<=c, 0.5 * u**2, c * abs(u) - 0.5 * c**2)
        huber_grad = lambda u : np.where(abs(u)<=c, u, np.sign(u)*c)

        beta0 = np.zeros(X.shape[1]) if len(beta0) == 0 else beta0

        fun = lambda b : np.mean(huber_loss(y - X@b))
        grad = lambda b : X.T @ huber_grad(X@b - y)/X.shape[0]
        model = minimize(fun, beta0, method='BFGS', 
                         jac=grad, tol=tol, options=options)

        return {'beta': model.x, 'robust': c,
                'res': y - X @ model.x,
                'niter': model.nit,
                'loss_val': model.fun,
                'grad_val': model.jac,
                'message': model.message}


    def adaHuber(self, dev_prob=None, max_iter=100):
        '''
            Adaptive Huber Regression

        Args:
            dev_prob : exception probability value between; 
                       default is 1/sample_size.
            max_iter : maximum number of iterations; default is 100.
        '''
        dev_prob = 1 / self.n if dev_prob is None else dev_prob
        y, X = self.Y, self.X
        beta_hat = np.linalg.solve(X.T @ X, X.T @ y)

        rel, err, t = (X.shape[1] + np.log(1 / dev_prob)) / self.n, 1, 0
        while err > self.params['tol'] and t < max_iter:
            res = y - X @ beta_hat 
            f = lambda c: np.mean(np.minimum((res / c) ** 2, 1)) - rel
            robust = find_root(f, np.min(np.abs(res)) \
                               + self.params['tol'], np.sqrt(res @ res))
            model = self.Huber(c=robust)
            err = np.max(np.abs(model['beta'] - beta_hat))
            beta_hat = model['beta']
            t += 1

        return {'beta': beta_hat, 'niter': t, 'robust': robust, 'res': res}



###############################################################################
################### Penalized Smoothed Quantile Regression ####################
###############################################################################
class high_dim(low_dim):
    '''
        Penalized Convolution Smoothed Quantile Regression via ILAMM
        (iterative local adaptive majorize-minimization)
    '''
    weights = ['Multinomial', 'Exponential', 'Rademacher']
    penalties = ["L1", "SCAD", "MCP", "CapppedL1"]
    params = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-8, 
              'iter_warning': True, 'warm_start': False, 'max_lr': 50,
              'irw_tol': 1e-5, 'nsim': 200, 'nboot': 200, 
              'min_bandwidth': 1e-4}

    def __init__(self, X, Y, intercept=True, options={}):

        '''
        Args:
            X: n by p matrix of covariates; each row is an observation vector.
            Y: an ndarray of response variables.
            intercept: logical flag for adding an intercept to the model.
            options: a dictionary of internal statistical and optimization parameters.
                phi: initial quadratic coefficient parameter in the ILAMM algorithm; 
                     default is 0.1.
                gamma: adaptive search parameter that is larger than 1; default is 1.25.
                max_iter: maximum numder of iterations in the ILAMM algorithm; default is 1e3.
                tol: the ILAMM iteration terminates when |beta^{k+1} - beta^k|_max <= tol; 
                     default is 1e-8.
                iter_warning: logical flag for warning when the maximum number 
                              of iterations is achieved for the l1-penalized fit.
                warm_start: logical flag for using a penalized robust expectile regression 
                            estimate as an initial value.
                irw_tol: tolerance parameter for stopping iteratively reweighted L1-penalization; 
                         default is 1e-5.
                nsim: number of simulations for computing a data-driven lambda; default is 200.
                nboot: number of bootstrap samples for post-selection inference; default is 200.
        '''
        self.n, self.p = X.shape
        self.Y = Y.reshape(self.n)
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X[:]]
            self.X1 = np.c_[np.ones(self.n), (X[:] - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X[:], X[:]/self.sdX

        self.params.update(options)


    def bandwidth(self, tau):
        h0 = (np.log(self.p) / self.n) ** 0.25
        return max(self.params['min_bandwidth'], h0 * (tau-tau**2) ** 0.5)


    def tuning(self, tau=0.5, standardize=True):
        '''
            A Simulation-based Approach for Choosing the Penalty Level
        
        Refs:
            l1-Penalized quantile regression in high-dimensinoal sparse models
            by Alexandre Belloni and Victor Chernozhukov
            The Annals of Statistics 39(1): 82--130, 2011

        Args:
            tau: quantile level; default is 0.5.
            standardize: logical flag for x variable standardization
                         prior to fitting the model; default is TRUE.
        
        Returns:
            lambda_sim: an ndarray of simulated lambda values.
        '''

        X = self.X1 if standardize else self.X
        lambda_sim = \
            np.array([max(abs(X.T@(tau - (rgt.uniform(0,1,self.n) <= tau))))
                      for b in range(self.params['nsim'])])
        return lambda_sim/self.n


    def l1_als(self, tau=0.5, Lambda=np.array([]), robust=5,
               beta0=np.array([]), standardize=True, adjust=True):
        '''
            L1-Penalized Asymmetric (Robust) Least Squares Regression
        '''
        y = self.Y
        X = self.X1 if standardize else self.X
        n, p = X.shape

        if len(beta0) == 0:
            beta0 = np.zeros(p)
            if self.itcp: beta0[0] = np.quantile(y, tau)
        
        Lambda = np.array(Lambda).reshape(-1)
        if not Lambda.any():
            Lambda = np.quantile(self.tuning(tau,standardize), 0.5)
            lambda_vec = Lambda * np.ones(p)
            if self.itcp: lambda_vec[0] = 0
        elif len(Lambda) == 1:
            lambda_vec = Lambda * np.ones(p)
            if self.itcp: lambda_vec[0] = 0
        else:
            lambda_vec = Lambda
            if self.itcp: lambda_vec = np.insert(lambda_vec, 0, 0)

        ln = lambda b: np.mean(AHuber_fn(y - X@b, tau, robust))
        gd = lambda b: X.T@(-AHuber_grad(y - X@b, tau, robust))/n
        model = lamm(self.params)
        model.minimize(ln, gd, beta0, lambda_vec)
        beta = model.x
        res = y - X @ beta
        if standardize and adjust:
            beta[self.itcp:] = beta[self.itcp:]/self.sdX
            if self.itcp: beta[0] -= self.mX @ beta[1:]

        return {'beta': beta, 
                'res': res, 
                'lambda': lambda_vec,
                'model': model}


    def l1(self, tau=0.5, Lambda=np.array([]), 
           h=None, kernel="Laplacian", beta0=np.array([]),
           standardize=True, adjust=True, weight=np.array([])):
        '''
            L1-Penalized Convolution Smoothed Quantile Regression (l1-conquer)

        Args:
            tau : quantile level; default is 0.5.
            Lambda : regularization parameter. This should be either a scalar, or 
                     a vector of length equal to the column dimension of X. If unspecified, 
                     it will be computed by self.tuning().
            h : bandwidth/smoothing parameter; the default value is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels; 
                     default is "Laplacian".
            beta0 : initial estimate. If unspecified, it will be set as a vector of zeros.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
            weight : an ndarray of observation weights; default is np.array([]) (empty).
        
        Returns:
            'beta' : an ndarray of estimated coefficients.
            'res' : an ndarray of fitted residuals.
            'lambda' : regularization parameter(s).
            'bw' : bandwidth.
        '''
        y = self.Y
        X = self.X1 if standardize else self.X
        n, p = X.shape
        h = self.bandwidth(tau) if h is None else h

        if len(beta0) == 0:
            if self.params['warm_start']:
                init = self.l1_als(tau, Lambda, 
                                   standardize=standardize, adjust=False)
                beta0 = init['beta'] 
            else:
                beta0 = np.zeros(p)
                if self.itcp: beta0[0] = np.quantile(y, tau)
        elif len(beta0) != p:
            print('Dimensions {} and {} not match'.format(len(beta0), p))
        
        Lambda = np.array(Lambda).reshape(-1)
        if not Lambda.any():
            Lambda = 0.75*np.quantile(self.tuning(tau,standardize), 0.9)
            lambda_vec = Lambda * np.ones(p)
            if self.itcp: lambda_vec[0] = 0
        elif len(Lambda) == 1:
            lambda_vec = Lambda * np.ones(p)
            if self.itcp: lambda_vec[0] = 0
        else:
            lambda_vec = Lambda
            if self.itcp: lambda_vec = np.insert(lambda_vec, 0, 0)

        fn = lambda b : smooth_check(y-X@b, tau, h, kernel, weight)
        gd = lambda b : X.T@(conquer_weight((X@b-y)/h, tau, kernel, weight))/n
        model = lamm(self.params)
        model.minimize(fn, gd, beta0, lambda_vec)
        beta = model.x
        res = y - X @ beta
        if standardize and adjust:
            beta[self.itcp:] = beta[self.itcp:]/self.sdX
            if self.itcp: beta[0] -= self.mX @ beta[1:]
            
        return {'beta': beta, 
                'res': res,
                'lambda': lambda_vec, 
                'bw': h,
                'model': model}


    def irw(self, tau=0.5, Lambda=np.array([]),
            h=None, kernel="Laplacian",
            beta0=np.array([]), penalty="SCAD", a=3.7, nstep=3, 
            standardize=True, adjust=True, weight=np.array([])):
        '''
            Iteratively Reweighted L1-Penalized Conquer (irw-l1-conquer)
        
        Args:
            tau : quantile level; default is 0.5.
            Lambda : regularization parameter. This should be either a scalar, or 
                     a vector of length equal to the column dimension of X. If unspecified, 
                     it will be computed by self.tuning().
            h : bandwidth/smoothing parameter; 
                default value is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            beta0 : initial estimator. If unspecified, it will be set as zero.
            penalty : a character string representing one of the built-in concave penalties;
                      default is "SCAD".
            a : the constant (>2) in the concave penality; default is 3.7.
            nstep : number of iterations/steps of the IRW algorithm; default is 3.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
            weight : an ndarray of observation weights; default is np.array([]) (empty).

        Returns:
            'beta' : an ndarray of estimated coefficients.
            'res' : an ndarray of fitted residuals.
            'nstep' : number of reweighted penalization steps.
            'lambda' : regularization parameter(s).
            'niter' : total number of iterations.
            'nit_seq' : a sequence of numbers of iterations.
        '''
        Lambda = np.array(Lambda).reshape(-1)
        if not Lambda.any():
            Lambda = 0.75*np.quantile(self.tuning(tau,standardize), 0.9)

        h = self.bandwidth(tau) if h is None else h
        if len(beta0) == 0:
            model = self.l1(tau, Lambda, h, kernel, standardize=standardize,
                            adjust=False, weight=weight)
        else:
            model = self.l1(tau, Lambda, h, kernel, beta0, standardize, 
                            adjust=False, weight=weight)

        nit_seq = []
        beta0 = model['beta']
        nit_seq.append(model['model'].niter)
        if penalty == 'L1': nstep = 0

        lam = Lambda * np.ones(len(self.mX))
        pos = lam > 0
        rw_lam = np.zeros(len(self.mX))

        dev, step = 1, 1
        while dev > self.params['irw_tol'] and step <= nstep:
            rw_lam[pos] = lam[pos] * \
                          concave_weight(beta0[self.itcp:][pos]/lam[pos], 
                                         penalty, a)
            model = self.l1(tau, rw_lam, h, kernel, beta0,
                            standardize, adjust=False, weight=weight)
            dev = max(abs(model['beta']-beta0))
            beta0, res = model['beta'], model['res']
            nit_seq.append(model['model'].niter)
            step += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX @ beta0[1:]
        nit_seq = np.array(nit_seq)
            
        return {'beta': beta0, 'res': res, 'nstep': step, 'lambda': Lambda,
                'niter': np.sum(nit_seq), 'nit_seq': nit_seq, 'bw': h}


    def irw_als(self, tau=0.5, Lambda=np.array([]), robust=3,
                penalty="SCAD", a=3.7, nstep=3,
                standardize=True, adjust=True):
        '''
            Iteratively Reweighted L1-Penalized Asymmetric Least Squares 
            (irw-l1-als)
        '''
        Lambda = np.array(Lambda).reshape(-1)
        if not Lambda.any():
            Lambda = np.quantile(self.tuning(tau,standardize), 0.9)
        
        model = self.l1_als(tau, Lambda, robust, 
                            standardize=standardize, adjust=False)
        beta0, res = model['beta'], model['res']

        lam = Lambda * np.ones(len(self.mX))
        pos = lam > 0
        rw_lam = np.zeros(len(self.mX))        

        dev, step = 1, 1
        while dev > self.params['irw_tol'] and step <= nstep:
            rw_lam[pos] = lam[pos] * \
                          concave_weight(beta0[self.itcp:][pos]/lam[pos], 
                                         penalty, a)
            model = self.l1_als(tau, rw_lam, robust,
                                beta0, standardize, adjust=False)
            dev = max(abs(model['beta']-beta0))
            beta0, res = model['beta'], model['res']
            step += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX @ beta0[1:]
            
        return {'beta': beta0, 'res': res, 'nstep': step, 
                'lambda': Lambda, 'robust': robust}
    

    def l1_path(self, tau, 
                lambda_seq=np.array([]), nlambda=50, order="descend",
                h=None, kernel="Laplacian", 
                standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized Conquer

        Args:
            tau : quantile level (float between 0 and 1).
            lambda_seq : an ndarray of lambda values.
            nlambda : number of lambda values (int).
            order : a character string indicating the order of lambda values 
                    along which the solution path is obtained; default is 'descend'.
            h : bandwidth/smoothing parameter (float).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.

        Returns:
            'beta_seq' : a sequence of l1-conquer estimates; 
                         each column corresponds to an estiamte for a penalty value.
            'res_seq' : a sequence of residual vectors. 
            'size_seq' : a sequence of numbers of selected variables. 
            'lambda_seq' : a sequence of penalty levels in ascending/descending order.
            'niter' : total number of iterations.
            'nit_seq' : a sequence of numbers of iterations.
            'bw' : bandwidth.
        '''

        if h is None: h = self.bandwidth(tau)
        
        lambda_seq = np.array(lambda_seq).reshape(-1)
        if not lambda_seq.any():
            lam_max = max(self.tuning(tau, standardize))
            lambda_seq = np.linspace(0.1*lam_max, lam_max, num=nlambda)
 
        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        nit_seq = []
        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))
        model = self.l1(tau, lambda_seq[0], h, kernel, 
                        standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        nit_seq.append(model['model'].niter)

        for l in range(1, len(lambda_seq)):
            model = self.l1(tau, lambda_seq[l], h, kernel, beta_seq[:,l-1],
                            standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
            nit_seq.append(model['model'].niter)

        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX @ beta_seq[1:,]

        return {'beta_seq': beta_seq, 
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[self.itcp:,:] != 0, axis=0),
                'lambda_seq': lambda_seq,
                'niter': np.sum(nit_seq),
                'nit_seq': np.array(nit_seq), 
                'bw': h}


    def irw_path(self, tau, 
                 lambda_seq=np.array([]), nlambda=50, order="descend",
                 h=None, kernel="Laplacian",
                 penalty="SCAD", a=3.7, nstep=3, 
                 standardize=True, adjust=True):
        '''
            Solution Path of Iteratively Reweighted L1-Conquer

        Args: 
            tau : quantile level (float between 0 and 1).
            lambda_seq : an ndarray of lambda values (int).
            nlambda : number of lambda values.
            order : a character string indicating the order of lambda values
                    along which the solution path is obtained; default is 'descend'.
            h : bandwidth/smoothing parameter (float).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            penalty : a character string representing one of the built-in concave penalties;
                      default is "SCAD".
            a : the constant (>2) in the concave penality; default is 3.7.
            nstep : number of iterations/steps of the IRW algorithm; default is 3.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.

        Returns:
            'beta_seq' : a sequence of irw-l1-conquer estimates;
                         each column corresponds to an estiamte for a penalty value.
            'res_seq' : a sequence of residual vectors. 
            'size_seq' : a sequence of numbers of selected variables. 
            'lambda_seq' : a sequence of penalty values in ascending/descending order.
            'nit_seq' : a sequence of numbers of iterations.
            'bw' : bandwidth.
        '''

        if h is None: h = self.bandwidth(tau)

        lambda_seq = np.array(lambda_seq).reshape(-1)
        if not lambda_seq.any():
            lam_max = max(self.tuning(tau, standardize))
            lambda_seq = np.linspace(0.5*lam_max, lam_max, num=nlambda)
        
        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        beta_seq = np.zeros(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))
        nit_seq = []
        model = self.irw(tau, lambda_seq[0], h, kernel,
                         penalty=penalty, a=a, nstep=nstep,
                         standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        nit_seq.append(model['niter'])

        for l in range(1, len(lambda_seq)):
            model = self.irw(tau, lambda_seq[l], h, kernel,
                             beta_seq[:,l-1], penalty, a, nstep, 
                             standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
            nit_seq.append(model['niter'])

        if standardize and adjust:
            beta_seq[self.itcp:,] /= self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX @ beta_seq[1:,]
    
        return {'beta_seq': beta_seq, 
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[self.itcp:,:] != 0, axis=0),
                'lambda_seq': lambda_seq,
                'nit_seq': np.array(nit_seq),
                'bw': h}


    def bic(self, tau=0.5, 
            lambda_seq=np.array([]), nlambda=100, order='descend',
            h=None, kernel="Laplacian", max_size=False, Cn=None,
            penalty="SCAD", a=3.7, nstep=3, 
            standardize=True, adjust=True):
        '''
            Model Selection via Bayesian Information Criterion
        
        Refs:
            Model selection via Bayesian information criterion 
            for quantile regression models (2014)
            by Eun Ryung Lee, Hohsuk Noh and Byeong U. Park
            Journal of the American Statistical Association 109(505): 216--229.

        Args:
            max_size : an upper bound on the selected model size; default is FALSE (no restriction).
            Cn : a positive constant in the modified BIC; default is log(log n).

        Returns:
            'bic_beta' : estimated coefficient vector for the BIC-selected model.
            'bic_res' : residual vector for the BIC-selected model.
            'bic_size' : size of the BIC-selected model.
            'bic_lambda' : penalty value that corresponds to the BIC-selected model.
            'beta_seq' : a sequence of penalized conquer estimates; 
                         each column corresponds to an estiamte for a penalty value.
            'size_seq' : a vector of estimated model sizes corresponding to lambda_seq.
            'lambda_seq' : a vector of penalty values.
            'bic' : a vector of BIC values corresponding to lambda_seq.
            'bw' : bandwidth.
        '''  
        lambda_seq = np.array(lambda_seq).reshape(-1)
        if not lambda_seq.any():
            lam_max = max(self.tuning(tau=tau, standardize=standardize))
            lambda_seq = np.linspace(0.1 * lam_max, lam_max, num=nlambda)
        else:
            nlambda = len(lambda_seq)

        if Cn is None: Cn = max(2, np.log(np.log(self.n)))

        if penalty not in self.penalties: 
            raise ValueError('Penalty "{}" is not included'.format(penalty))

        check_sum = lambda x : np.sum(np.where(x >= 0, tau * x, (tau - 1) * x))

        if penalty == "L1":
            model_all = self.l1_path(tau, lambda_seq, nlambda, order,
                                     h, kernel, standardize, adjust)
        else:
            model_all = self.irw_path(tau, lambda_seq, nlambda, order,
                                      h, kernel, penalty, a, nstep, 
                                      standardize, adjust)

        BIC = np.array([np.log(check_sum(model_all['res_seq'][:,l])) 
                        for l in range(nlambda)])
        BIC += model_all['size_seq'] * np.log(self.p) * Cn / self.n
        if not max_size:
            bic_select = np.argmin(BIC)
        else:
            bic_select = np.where(BIC==min(BIC[model_all['size_seq'] 
                                               <= max_size]))[0][0]

        return {'bic_beta': model_all['beta_seq'][:,bic_select],
                'bic_res':  model_all['res_seq'][:,bic_select],
                'bic_size': model_all['size_seq'][bic_select],
                'bic_lambda': model_all['lambda_seq'][bic_select],
                'beta_seq': model_all['beta_seq'],
                'size_seq': model_all['size_seq'],
                'lambda_seq': model_all['lambda_seq'],
                'bic': BIC,
                'bw': model_all['bw']}


    def boot_select(self, tau=0.5, Lambda=None, 
                    h=None, kernel="Laplacian",
                    weight="Multinomial", penalty="SCAD", a=3.7, nstep=3,
                    standardize=True, parallel=False, ncore=None):
        '''
            Model Selection via Bootstrap 

        Args:   
            tau : quantile level; default is 0.5.
            Lambda : regularization parameter (float);
                     if unspecified, it will be computed by self.tuning().
            h : smoothing parameter/bandwidth. The default is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            weight : a character string representing the random weight distribution;
                     default is "Multinomial".
            penalty : a character string representing one of the built-in concave penalties;
                      default is "SCAD".
            a : the constant (>2) in the concave penality; default is 3.7.
            nstep : number of iterations/steps of the IRW algorithm; default is 3.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            parallel : logical flag to implement bootstrap using parallel computing;
                       default is FALSE.
            ncore : number of cores used for parallel computing.
        
        Returns:
            'boot_beta' : numpy array. 
                          1st column: penalized conquer estimate; 
                          2nd to last: bootstrap estimates.
            'majority_vote' : selected model by majority vote.
            'intersection' : selected model by intersecting.
        '''

        if Lambda is None: 
            Lambda = 0.75*np.quantile(self.tuning(tau, standardize), 0.9)
        if h is None: h = self.bandwidth(tau) 
        if weight not in self.weights[:3]:
            raise ValueError('Distribution "{}" is not included'.format(weight))
            
        model = self.irw(tau, Lambda, h, kernel, 
                         penalty=penalty, a=a, nstep=nstep,
                         standardize=standardize, adjust=False)
        mb_beta = np.zeros(shape=(self.p+self.itcp, self.params['nboot']+1))
        mb_beta[:,0] = model['beta']
        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX @ mb_beta[1:,0]

        if parallel:
            import multiprocessing
            max_ncore = multiprocessing.cpu_count()
            if ncore is None: ncore = max_ncore
            if ncore > max_ncore: 
                raise ValueError("Number of cores exceeds the limit")

        def bootstrap(b):
            boot_fit = self.irw(tau, Lambda, h, kernel,
                                beta0=model['beta'],
                                penalty=penalty, a=a, nstep=nstep,
                                standardize=standardize,
                                weight=boot_weight(weight)(self.n))
            return boot_fit['beta']

        if not parallel:
            for b in range(self.params['nboot']): 
                mb_beta[:,b+1] = bootstrap(b)
        else:
            from joblib import Parallel, delayed
            boot_results \
                = Parallel(n_jobs=ncore)(delayed(bootstrap)(b)
                                         for b in range(self.params['nboot']))
            mb_beta[:,1:] = np.array(boot_results).T
        
        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        
        ## Method 1: Majority vote among all bootstrap models
        selection_rate = np.mean(mb_beta[self.itcp:,1:]!=0, axis=1)
        model1 = np.where(selection_rate>0.5)[0]
        
        ## Method 2: Intersection of all bootstrap models
        model2 = np.arange(self.p)
        for b in range(len(mb_beta[0,1:])):
            boot_model = np.where(mb_beta[self.itcp:,b+1] != 0)[0]
            model2 = np.intersect1d(model2, boot_model)

        return {'boot_beta': mb_beta,
                'majority_vote': model1,
                'intersection': model2}


    def boot_inference(self, tau=0.5, Lambda=None, 
                       h=None, kernel="Laplacian",
                       weight="Multinomial", alpha=0.05, 
                       penalty="SCAD", a=3.7, nstep=3,
                       standardize=True, parallel=False, ncore=None):
        '''
            Post-Selection-Inference via Bootstrap

        Returns:
            'boot_beta' : numpy array. 
                          1st column: penalized conquer estimate; 
                          2nd to last: bootstrap estimates.
            'majority_vote' : selected model by majority vote.
            'intersection' : selected model by intersecting.
            'percentile' : numpy array. Percentile bootstrap CI.
            'pivotal' : numpy array. Pivotal bootstrap CI.
            'normal' : numpy array. Normal-based CI using bootstrap variance estimates.
        '''

        mb_model = self.boot_select(tau, Lambda, h, kernel, weight,
                                    penalty, a, nstep, standardize,
                                    parallel, ncore)
        
        per = np.empty([self.p + self.itcp, 2])
        piv = np.empty([self.p + self.itcp, 2])
        clt = np.empty([self.p + self.itcp, 2])

        # post-selection bootstrap inference
        X_select = self.X[:, mb_model['majority_vote']+self.itcp]
        fit = low_dim(X_select, self.Y, self.itcp)\
              .mb_ci(tau, kernel=kernel, weight=weight,
                     alpha=alpha, standardize=standardize)

        per[mb_model['majority_vote']+self.itcp,:] \
            = fit['percentile'][self.itcp:,:]
        piv[mb_model['majority_vote']+self.itcp,:] \
            = fit['pivotal'][self.itcp:,:]
        clt[mb_model['majority_vote']+self.itcp,:] \
            = fit['normal'][self.itcp:,:]

        if self.itcp: 
            per[0,:] = fit['percentile'][0,:]
            piv[0,:] = fit['pivotal'][0,:]
            clt[0,:] = fit['normal'][0,:]

        return {'boot_beta': mb_model['boot_beta'],
                'percentile': per,
                'pivotal': piv,
                'normal': clt,
                'majority_vote': mb_model['majority_vote'],
                'intersection': mb_model['intersection']}


    def l0(self, tau=0.5, h=None, kernel='Laplacian', 
           sparsity=5, exp_size=5, beta0=np.array([]),
           standardize=True, adjust=True,
           tol=1e-5, max_iter=1e3):
        '''
            L0-Penalized Conquer via Two-Step Iterative Hard-Thresholding

        Refs:
            On iterative hard thresholding methods for high-dimensional M-estimation
            by Prateek Jain, Ambuj Tewari and Purushottam Kar
            Advances in Neural Information Processing Systems 27, 2014

        Args:
            tau : quantile level between 0 and 1 (float); default is 0.5.
            h : smoothing/bandwidth parameter (float).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            sparsity : sparsity level (int, >=1); default is 5.
            exp_size : expansion size (int, >=1); default is 5.
            beta0 : initial estimate.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.

        Returns:
            'beta' : an ndarray of estimated coefficients.
            'select' : indices of non-zero estimated coefficients (intercept excluded).
            'bw' : bandwidth.
            'niter' : number of IHT iterations.
        '''

        X, Y, itcp = self.X, self.Y, self.itcp

        if h is None: 
            h0 = min((sparsity + np.log(self.n))/self.n, 0.5) ** 0.4
            h = max(0.01, h0 * (tau-tau**2) ** 0.5)
        if len(beta0) == 0: beta0 = np.zeros(X.shape[1])
        
        t, dev = 0, 1
        while t < max_iter and dev > tol:
            grad0 = X.T@(conquer_weight((X@beta0 - Y)/h, tau, kernel)/self.n)
            supp0 = sparse_supp(grad0[itcp:], exp_size) + (beta0[itcp:] != 0)
            beta1 = np.zeros(X.shape[1])
            out0 = low_dim(X[:,itcp:][:,supp0], Y, intercept=itcp) \
                   .fit(tau=tau, h=h, standardize=standardize, adjust=adjust)
            beta1[itcp:][supp0] = out0['beta'][itcp:]
            if itcp: beta1[0] = out0['beta'][0]
            beta1[itcp:] = sparse_proj(beta1[itcp:], sparsity)
            supp1 = beta1[itcp:] != 0
            out1 = low_dim(X[:,itcp:][:,supp1], Y, intercept=itcp) \
                   .fit(tau=tau, h=h, standardize=standardize, adjust=adjust)
            beta1[itcp:][supp1] = out1['beta'][itcp:]
            if itcp: beta1[0] = out1['beta'][0]
            dev = max(abs(beta1 - beta0))
            beta0 = beta1[:]
            t += 1

        return {'beta': beta0, 
                'select': np.where(beta0[itcp:] != 0)[0],
                'bw': h,
                'niter': t}


    def l0_path(self, tau, h=None, kernel='Laplacian', 
                sparsity_seq=np.array([]), order='ascend',
                sparsity_max=20, exp_size=5, 
                standardize=True, adjust=True,
                tol=1e-5, max_iter=1e3):
        '''
            Solution Path of L0-Penalized Conquer

        Args:
            tau : quantile level between 0 and 1 (float).
            h : smoothing/bandwidth parameter (float).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            sparsity_seq : a sequence of sparsity levels (int, >=1).
            order : a character string indicating the order of sparsity levels 
                    along which the solution path is obtained; default is 'ascend'.
            sparsity_max : maximum sparsity level (int, >=1).
            exp_size : expansion size (int, >=1); default is 5.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
            tol : tolerance for convergence.
            max_iter : maximum number of iterations.

        Returns:
            'beta_seq' : a sequence of l0-conquer estimates;
                         each column corresponds to an estiamte for a sparsity level.
            'size_seq' : a sequence of numbers of selected variables.
            'bw_seq' : a sequence of bandwidths.
            'nit_seq' : a sequence of numbers of iterations.
        '''
        if len(sparsity_seq) == 0:
            sparsity_seq = np.array(range(1, sparsity_max+1))
            
        if order=='ascend':
            sparsity_seq = np.sort(sparsity_seq)
        elif order=='descend':
            sparsity_seq = np.sort(sparsity_seq)[::-1]
        L = len(sparsity_seq)

        if h is None: 
            h0 = np.minimum((sparsity_seq + np.log(self.n))/self.n, 0.5)
            h = np.maximum(0.01, h0 ** 0.4 * (tau-tau**2) ** 0.5)
        else:
            h = h * np.ones(L)

        beta_seq, nit_seq = np.zeros((self.X.shape[1], L+1)), []
        for k in range(L):
            model = self.l0(tau, h[k], kernel,
                            sparsity_seq[k], exp_size, beta_seq[:,k-1],
                            standardize, adjust, tol, max_iter)
            beta_seq[:,k] = model['beta']
            nit_seq.append(model['niter'])

        return {'beta_seq': beta_seq[:,:L],  
                'size_seq': np.sum(beta_seq[self.itcp:,:L] != 0, axis=0),
                'bw_seq': h,
                'nit_seq': np.array(nit_seq)}



##############################################################################
##### Use Cross-Validation to Choose the Regularization Parameter Lambda #####
##############################################################################
class cv_lambda:
    '''
        Cross-Validated Penalized Quantile Regression 
    '''
    penalties = ["L1", "SCAD", "MCP"]
    params = {'phi': 0.1, 'gamma': 1.25, 'warm_start': True, 
              'max_iter': 1e3, 'max_lr': 50,
              'tol': 1e-6, 'irw_tol': 1e-5, 'nsim': 200}
    methods = ['conquer', 'admm']


    def __init__(self, X, Y, intercept=True, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.itcp = intercept
        self.params.update(options)


    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds


    def fit(self, tau=0.5, h=None, kernel="Laplacian", 
            lambda_seq=np.array([]), nlambda=40, order='descend',
            nfolds=5, penalty="SCAD", a=3.7, nstep=3,
            method='conquer', standardize=True, adjust=True,
            sigma=0.01, eta=None, smoothed_criterion=False):

        if method not in self.methods: 
            raise ValueError("Method must be either conquer or admm")
        if penalty not in self.penalties: 
            raise ValueError("Penalty must be either L1, SCAD or MCP")

        init = high_dim(self.X, self.Y, self.itcp, self.params)
        if h is None: h = init.bandwidth(tau)
        itcp = self.itcp

        if not lambda_seq.any():
            lam_max = max(init.tuning(tau, standardize))
            lambda_seq = np.linspace(0.25*lam_max, lam_max, num=nlambda)

        else:
            nlambda = len(lambda_seq)
        
        # empirical check loss
        check = lambda x : np.mean(np.where(x >= 0, tau * x, (tau - 1)*x))   
        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train = self.X[np.setdiff1d(idx,folds[v]),:] 
            Y_train = self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]

            if method == 'conquer':
                train = high_dim(X_train, Y_train, itcp, self.params)
            elif method == 'admm':
                train = proximal(X_train, Y_train, itcp)

            if penalty == "L1":
                if method == 'conquer':
                    model = train.l1_path(tau, lambda_seq, nlambda, order,
                                          h, kernel, standardize, adjust)
                elif method == 'admm':
                    model = train.l1_path(tau, lambda_seq, nlambda, order, 
                                          sigma=sigma, eta=eta)
            else:
                if method == 'conquer':
                    model = train.irw_path(tau, lambda_seq, nlambda, order,
                                           h, kernel, penalty, a, nstep, 
                                           standardize, adjust)
                elif method == 'admm':
                    model = train.irw_path(tau, lambda_seq, nlambda, order,
                                           sigma=sigma, eta=eta,
                                           penalty=penalty, a=a, nstep=nstep)
            
            if not smoothed_criterion:
                val_err[v,:] \
                    = np.array([check(Y_val - model['beta_seq'][0,l]*itcp 
                                      - X_val@(model['beta_seq'][itcp:,l]))
                                      for l in range(nlambda)])
            else:
                val_err[v,:] \
                    = np.array([smooth_check(Y_val - model['beta_seq'][0,l]*itcp
                                             - X_val@(model['beta_seq'][itcp:,l]),
                                             tau, h, kernel)
                                             for l in range(nlambda)])

        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][np.argmin(cv_err)]

        if penalty == "L1":
            if method == 'conquer':
                cv_model = init.l1(tau, lambda_min, h, kernel,
                                   standardize=standardize, adjust=adjust)
            elif method == 'admm':
                init = proximal(self.X, self.Y, itcp)
                cv_model = init.l1(tau, lambda_min, sigma=sigma, eta=eta)
        else:
            if method == 'conquer':
                cv_model = init.irw(tau, lambda_min, h, kernel,
                                    penalty=penalty, a=a, nstep=nstep,
                                    standardize=standardize, adjust=adjust)
            elif method == 'admm':
                init = proximal(self.X, self.Y, itcp)
                cv_model = init.irw(tau, lambda_min, sigma=sigma, eta=eta,
                                    penalty=penalty, a=a, nstep=nstep)

        return {'cv_beta': cv_model['beta'],
                'cv_res': cv_model['res'],
                'lambda_min': lambda_min,
                'lambda_seq': model['lambda_seq'],
                'min_cv_err': cv_min,
                'cv_err': cv_err}



##############################################################################
###### Use Validation Set to Choose the Regularization Parameter Lambda ######
##############################################################################
class validate_lambda(cv_lambda):
    '''
        Train Penalized Conquer on a Validation Set
    '''
    penalties = ["L1", "SCAD", "MCP"]
    
    def __init__(self, X_train, Y_train, X_val, Y_val, 
                 intercept=True, options={}):
        self.n, self.p = X_train.shape
        self.X_train, self.Y_train = X_train, Y_train.reshape(self.n)
        self.X_val, self.Y_val = X_val, Y_val.reshape(len(Y_val))
        self.itcp = intercept
        self.params.update(options)


    def train(self, tau=0.5, h=None, kernel="Laplacian", 
              lambda_seq=np.array([]), nlambda=20, order='descend', 
              penalty="SCAD", a=3.7, nstep=3, standardize=True):
        '''
        Args:
            tau : quantile level between 0 and 1 (float).
            h : smoothing/bandwidth parameter (float).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            lambda_seq : an ndarray of lambda values (int).
            nlambda : number of lambda values.
            order : a character string indicating the order of lambda values
                    along which the solution path is obtained; default is 'descend'.
            penalty : a character string representing one of the built-in concave penalties;
                      default is "SCAD".
            a : the constant (>2) in the concave penality; default is 3.7.
            nstep : number of iterations/steps of the IRW algorithm; default is 3.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
        
        Returns:
            'val_beta' : an ndarray of regression estimates. 
            'val_res' : an ndarray of fitted residuals.
            'model_size' : a sequence of selected model sizes.
            'lambda_min' : the value of lambda that gives minimum validation error.
            'lambda_seq' : a sequence of lambdas in descending order. 
            'val_min' : minimum validation error.
            'val_seq' : a sequence of validation errors.
        '''
  
        train = high_dim(self.X_train, self.Y_train,
                         intercept=self.itcp, options=self.params)
        if not lambda_seq.any():
            lam_max = max(train.tuning(tau, standardize))
            lambda_seq = np.linspace(0.25*lam_max, lam_max, num=nlambda)
        else:
            nlambda = len(lambda_seq)
        
        h = train.bandwidth(tau) if h is None else h

        if penalty not in self.penalties:
            raise ValueError("Penalty must be either L1, SCAD or MCP")
        elif penalty == "L1":
            model = train.l1_path(tau, lambda_seq, nlambda, order,
                                  h, kernel, standardize)
        else:
            model = train.irw_path(tau, lambda_seq, nlambda, order,
                                   h, kernel, penalty, a, nstep, standardize)
        
        # empirical check loss
        loss = lambda x : np.mean(np.where(x >= 0, tau * x, (tau - 1)*x))   
        val_err = \
                np.array([loss(self.Y_val - model['beta_seq'][0,l]*self.itcp 
                               - self.X_val@(model['beta_seq'][self.itcp:,l]))
                               for l in range(nlambda)])
        val_min = min(val_err)
        l_min = np.where(val_err==val_min)[0][0]

        return {'val_beta': model['beta_seq'][:,l_min],
                'val_res': model['res_seq'][:,l_min],
                'val_size': model['size_seq'][l_min],
                'lambda_min': model['lambda_seq'][l_min],
                'lambda_seq': model['lambda_seq'],
                'min_val_err': val_min, 'val_err': val_err}
    


##############################################################################
######################### Linear QuantES Regression ##########################
##############################################################################
class joint(low_dim):
    '''
    Joint Linear Quantile and Expected Shortfall Regression    
    
    Methods:
        fz(): Joint quantile and expected shortfall regression
        twostep(): Two-step procedure for joint regression
        boot_es(): Bootstrap for expected shortfall regression
        nc_fit(): Non-crossing joint quantile and expected shortfall regression    
    '''

    def __init__(self, X, Y, intercept=True, options=dict()):
        super().__init__(X, Y, intercept, options)


    def _fz_loss(self, x, tau, G1=False, G2_type=1):
        '''
            Fissler and Ziegel's Joint Loss Function

        Args:
            G1 : logical flag for the specification function G1 in FZ's loss;
                 G1(x)=0 if G1=False, and G1(x)=x and G1=True.
            G2_type : an integer (from 1 to 5) that indicates the type.
                        of the specification function G2 in FZ's loss.
        
        Returns:
            FZ loss function value.
        '''

        X = self.X
        if G2_type in {1, 2, 3}:
            Ymax = np.max(self.Y)
            Y = self.Y - Ymax
        else:
            Y = self.Y
        dim = X.shape[1]
        Yq = X @ x[:dim]
        Ye = X @ x[dim : 2*dim]
        f0, f1, _ = G2(G2_type)
        loss = f1(Ye) * (Ye - Yq - (Y - Yq) * (Y<= Yq) / tau) - f0(Ye)
        if G1:
            return np.mean((tau - (Y<=Yq)) * (Y-Yq) + loss)
        else:
            return np.mean(loss)


    def fz(self, tau=0.5, G1=False, G2_type=1,
           standardize=True, refit=True, tol=None, 
           options={'maxiter': None, 'maxfev': None, 'disp': False, 
                    'return_all': False, 'initial_simplex': None, 
                    'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False}):
        '''
            Joint Quantile & Expected Shortfall Regression via FZ Loss Minimization

        Refs:
            Higher Order Elicitability and Osband's Principle
            by Tobias Fissler and Johanna F. Ziegel
            Ann. Statist. 44(4): 1680-1707, 2016

            A Joint Quantile and Expected Shortfall Regression Framework
            by Timo Dimitriadis and Sebastian Bayer 
            Electron. J. Stat. 13(1): 1823-1871, 2019
        
        Args:
            tau : quantile level; default is 0.5.
            G1 : logical flag for the specification function G1 in FZ's loss; 
                 G1(x)=0 if G1=False, and G1(x)=x and G1=True.
            G2_type : an integer (from 1 to 5) that indicates the type of the specification function G2 in FZ's loss.
            standardize : logical flag for x variable standardization prior to fitting the quantile model; 
                          default is TRUE.
            refit : logical flag for refitting joint regression if the optimization is terminated early;
                    default is TRUE.
            tol : tolerance for termination.
            options : a dictionary of solver options; 
                      see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
        
        Returns:
            'coef_q' : quantile regression coefficient estimate.
            'coef_e' : expected shortfall regression coefficient estimate.
            'nit' : total number of iterations. 
            'nfev' : total number of function evaluations.
            'message' : a message that describes the cause of the termination.
        '''

        dim = self.X.shape[1]
        Ymax = np.max(self.Y)
        ### warm start with QR + truncated least squares
        qrfit = low_dim(self.X[:, self.itcp:], self.Y, intercept=True)\
                .fit(tau=tau, standardize=standardize)
        coef_q = qrfit['beta']
        tail = self.Y <= self.X @ coef_q
        tail_X = self.X[tail,:] 
        tail_Y = self.Y[tail]
        coef_e = np.linalg.solve(tail_X.T @ tail_X, tail_X.T @ tail_Y)
        if G2_type in {1, 2, 3}:
            coef_q[0] -= Ymax
            coef_e[0] -= Ymax
        x0 = np.r_[(coef_q, coef_e)]

        ### joint quantile and ES fit
        fun  = lambda x : self._fz_loss(x, tau, G1, G2_type)
        esfit = minimize(fun, x0, method='Nelder-Mead', 
                         tol=tol, options=options)
        nit, nfev = esfit['nit'], esfit['nfev']

        ### refit if convergence criterion is not met
        while refit and not esfit['success']:
            esfit = minimize(fun, esfit['x'], method='Nelder-Mead',
                             tol=tol, options=options)
            nit += esfit['nit']   
            nfev += esfit['nfev'] 

        coef_q, coef_e = esfit['x'][:dim], esfit['x'][dim : 2*dim]
        if G2_type in {1, 2, 3}:
            coef_q[0] += Ymax
            coef_e[0] += Ymax

        return {'coef_q': coef_q, 'coef_e': coef_e,
                'nit': nit,   # total number of iterations
                'nfev': nfev, # total number of function evaluations
                'success': esfit['success'],
                'message': esfit['message']}


    def twostep(self, tau=0.5, h=None, kernel='Laplacian', 
                loss='L2', robust=None, G2_type=1,
                standardize=True, tol=None, options=None,
                ci=False, level=0.95):
        '''
            Two-Step Procedure for Joint QuantES Regression

        Refs:
            Higher Order Elicitability and Osband's Principle
            by Tobias Fissler and Johanna F. Ziegel
            Ann. Statist. 44(4): 1680-1707, 2016

            Effciently Weighted Estimation of Tail and Interquantile Expectations
            by Sander Barendse 
            SSRN Preprint, 2020
        
            Robust Estimation and Inference 
            for Expected Shortfall Regression with Many Regressors
            by Xuming He, Kean Ming Tan and Wen-Xin Zhou
            J. R. Stat. Soc. B. 85(4): 1223-1246, 2023

            Inference for Joint Quantile and Expected Shortfall Regression
            by Xiang Peng and Huixia Judy Wang
            Stat 12(1) e619, 2023

        Args:
            tau : quantile level; default is 0.5.
            h : bandwidth; the default value is computed by self.bandwidth(tau).
            kernel : a character string representing one of the built-in smoothing kernels;
                     default is "Laplacian".
            loss : the loss function used in stage two. There are three options.
                1. 'L2': squared/L2 loss;
                2. 'Huber': Huber loss;
                3. 'FZ': Fissler and Ziegel's joint loss.
            robust : robustification parameter in the Huber loss;
                     if robust=None, it will be automatically determined in a data-driven way;
            G2_type : an integer (from 1 to 5) that indicates the type of the specification function G2 in FZ's loss.
            standardize : logical flag for x variable standardization prior to fitting the quantile model;
                          default is TRUE.
            tol : tolerance for termination.
            options : a dictionary of solver options;
                      see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html.
            ci : logical flag for computing normal-based confidence intervals.
            level : confidence level between 0 and 1.

        Returns:
            'coef_q' : quantile regression coefficient estimate.
            'res_q' : a vector of fitted quantile regression residuals.
            'coef_e' : expected shortfall regression coefficient estimate.
            'robust' : robustification parameter in the Huber loss.
            'ci' : coordinates-wise (100*level)% confidence intervals.
        '''
  
        if loss in {'L2', 'Huber', 'TrunL2', 'TrunHuber'}:
            qrfit = self.fit(tau=tau, h=h, kernel=kernel, 
                             standardize=standardize)
            nres_q = np.minimum(qrfit['res'], 0)
        
        if loss == 'L2':
            adj = np.linalg.solve(self.X.T@self.X, self.X.T@nres_q / tau)
            coef_e = qrfit['beta'] + adj
            robust = None
        elif loss == 'Huber':
            Z = nres_q + tau*(self.Y - qrfit['res'])
            X0 = self.X[:, self.itcp:]
            esr = low_dim(tau*X0, Z, intercept=self.itcp)
            if robust == None:
                esfit = esr.adaHuber()
                coef_e = esfit['beta']
                robust = esfit['robust']
                if self.itcp: coef_e[0] /= tau
            elif robust > 0:
                esfit = esr.als(robust=robust, 
                                    standardize=standardize, scale=False)
                coef_e = esfit['beta']
                robust = esfit['robust']
                if self.itcp: coef_e[0] /= tau
            else:
                raise ValueError("Robustification parameter must be positive")
        elif loss == 'FZ':
            if G2_type in np.arange(1,4):
                Ymax = np.max(self.Y)
                Y = self.Y - Ymax
            else:
                Y = self.Y
            qr = low_dim(self.X[:, self.itcp:], Y, intercept=True)
            qrfit = qr.fit(tau=tau, h=h, kernel=kernel, 
                           standardize=standardize)
            adj = np.minimum(qrfit['res'], 0)/tau + Y - qrfit['res']
            f0, f1, f2 = G2(G2_type)

            fun  = lambda z : np.mean(f1(self.X @ z) * (self.X @ z - adj) \
                                      - f0(self.X @ z))
            grad = lambda z : self.X.T@(f2(self.X@z)*(self.X@z - adj))/self.n
            esfit = minimize(fun, qrfit['beta'], method='BFGS', 
                             jac=grad, tol=tol, options=options)
            coef_e = esfit['x']
            robust = None
            if G2_type in np.arange(1,4):
                coef_e[0] += Ymax
                qrfit['beta'][0] += Ymax
        elif loss == 'TrunL2':
            tail = self.Y <= self.X @ qrfit['beta']
            tail_X = self.X[tail,:] 
            tail_Y = self.Y[tail]
            coef_e = np.linalg.solve(tail_X.T @ tail_X, 
                                     tail_X.T @ tail_Y)
            robust = None
        elif loss == 'TrunHuber':
            tail = self.Y <= self.X @ qrfit['beta']
            esfit = low_dim(self.X[tail, self.itcp:], self.Y[tail],
                            intercept=self.itcp).adaHuber()
            coef_e = esfit['beta']
            robust = esfit['robust']

        if loss in {'L2', 'Huber'} and ci:
            res_e = nres_q + tau * self.X@(qrfit['beta'] - coef_e)
            n, p = self.X[:,self.itcp:].shape
            X0 = np.c_[np.ones(n,), self.X[:,self.itcp:] - self.mX]
            if loss == 'L2':
                weight = res_e ** 2
            else:
                weight = np.minimum(res_e ** 2, robust ** 2)
    
            inv_sig = np.linalg.inv(X0.T @ X0 / n)   
            acov = inv_sig @ ((X0.T * weight) @ X0 / n) @ inv_sig
            radius = norm.ppf(1/2 + level/2) * np.sqrt(np.diag(acov)/n) / tau
            ci = np.c_[coef_e - radius, coef_e + radius]

        return {'coef_q': qrfit['beta'], 
                'res_q': qrfit['res'], 
                'coef_e': coef_e,
                'loss': loss, 
                'robust': robust,
                'ci': ci, 
                'level': level}


    def boot_es(self, tau=0.5, h=None, kernel='Laplacian', 
                loss='L2', robust=None, standardize=True, 
                B=200, level=0.95):

        fit = self.twostep(tau, h, kernel, loss,
                           robust, standardize=standardize)
        boot_coef = np.empty((self.X.shape[1], B))
        for b in range(B):
            idx = rgt.choice(np.arange(self.n), size=self.n)
            boot = joint(self.X[idx,self.itcp:], self.Y[idx], 
                         intercept=self.itcp)
            if loss == 'L2':
                bfit = boot.twostep(tau, h, kernel, loss='L2',
                                    standardize=standardize)
            else:
                bfit = boot.twostep(tau, h, kernel, loss,
                                    robust=fit['robust'],
                                    standardize=standardize)
            boot_coef[:,b] = bfit['coef_e']
        
        left  = np.quantile(boot_coef, 1/2-level/2, axis=1)
        right = np.quantile(boot_coef, 1/2+level/2, axis=1)
        piv = np.c_[2*fit['coef_e'] - right, 2*fit['coef_e'] - left]
        per = np.c_[left, right]

        return {'coef_q': fit['coef_q'],
                'coef_e': fit['coef_e'],
                'boot_coef_e': boot_coef,
                'loss': loss, 
                'robust': fit['robust'], 
                'pivotal': piv, 
                'percentile': per, 
                'level': level}


    def nc(self, tau=0.5, h=None, kernel='Laplacian', 
           loss='L2', robust=None, standardize=True, 
           ci=False, level=0.95):
        '''
            Non-Crossing Joint Quantile & Expected Shortfall Regression

        Refs:
            Robust Estimation and Inference  
            for Expected Shortfall Regression with Many Regressors
            by Xuming He, Kean Ming Tan and Wen-Xin Zhou
            J. R. Stat. Soc. B. 85(4): 1223-1246, 2023
        '''

        qrfit = self.fit(tau=tau, h=h, kernel=kernel, 
                         standardize=standardize)
        nres_q = np.minimum(qrfit['res'], 0)
        fitted_q = self.Y - qrfit['res']
        Z = nres_q/tau + fitted_q
 
        P = matrix(self.X.T @ self.X / self.n)
        q = matrix(-self.X.T @ Z / self.n)
        G = matrix(self.X)
        hh = matrix(fitted_q)
        l, c = 0, robust 
        
        if loss == 'L2':
            esfit = solvers.qp(P, q, G, hh, 
                               initvals={'x': matrix(qrfit['beta'])})
            coef_e = np.array(esfit['x']).reshape(self.X.shape[1],)
        else:
            rel = (self.X.shape[1] + np.log(self.n)) / self.n
            esfit = self.twostep(tau, h, kernel, loss, 
                                 robust, standardize=standardize)
            coef_e = esfit['coef_e']
            res  = np.abs(Z - self.X @ coef_e)
            c = robust
            
            if robust == None:
                fun = lambda t : np.mean(np.minimum((res/t)**2, 1)) - rel
                c = find_root(fun,
                              np.min(res)+self.params['tol'], 
                              np.sqrt(res @ res))

            sol_diff = 1
            while l < self.params['max_iter'] \
                and sol_diff > self.params['tol']:
                wt = np.where(res > c, res/c, 1)
                P = matrix( (self.X.T / wt ) @ self.X / self.n)
                q = matrix( -self.X.T @ (Z / wt) / self.n)
                esfit = solvers.qp(P, q, G, hh, 
                                   initvals={'x': matrix(coef_e)})
                tmp = np.array(esfit['x']).reshape(self.X.shape[1],)
                sol_diff = np.max(np.abs(tmp - coef_e))
                res = np.abs(Z - self.X @ tmp)
                if robust == None:
                    fun = lambda t : np.mean(np.minimum((res/t)**2, 1)) - rel
                    c = find_root(fun,
                                  np.min(res)+self.params['tol'], 
                                  np.sqrt(res @ res))
                coef_e = tmp
                l += 1
            c *= tau

        if ci:
            res_e = nres_q + tau * (fitted_q - self.X @ coef_e)
            X0 = np.c_[np.ones(self.n,), self.X[:,self.itcp:] - self.mX]
            if loss == 'L2': weight = res_e ** 2
            else: weight = np.minimum(res_e ** 2, c ** 2)
    
            inv_sig = np.linalg.inv(X0.T @ X0 / self.n)   
            acov = inv_sig @ ((X0.T * weight) @ X0 / self.n) @ inv_sig
            radius = norm.ppf((1+level)/2)*np.sqrt(np.diag(acov)/self.n) / tau
            ci = np.c_[coef_e - radius, coef_e + radius]

        return {'coef_q': qrfit['beta'], 
                'res_q': qrfit['res'], 
                'coef_e': coef_e, 
                'nit': l,
                'loss': loss, 
                'robust': c,
                'ci': ci, 
                'level': level}
##############################################################################
##############################################################################
##############################################################################