import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.sparse import csc_matrix

from sklearn.metrics import pairwise_kernels as PK
from sklearn.kernel_ridge import KernelRidge as KR
from sklearn.preprocessing import PolynomialFeatures as PF

from qpsolvers import Problem, solve_problem
from cvxopt import solvers
solvers.options['show_progress'] = False

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from enum import Enum

from quantes.utils import (mad, QuantLoss, JointLoss, HuberLoss, 
                           HuberGrad, weighted_quantile,
                           make_train_step_fn, make_val_step_fn, mini_batch)


###############################################################################
########################## Kernel Ridge Regression ############################
###############################################################################
class KRR:
    '''
    Kernel Ridge Regression
    
    Methods:
        __init__(): Initialize the KRR object
        ls(): Fit kernel ridge mean (least squares) regression
        qt(): Fit (smoothed) quantile kernel ridge regression
        qt_seq(): Fit a sequence of quantile kernel ridge regressions
        es(): Fit expected shortfall kernel ridge regression
        res(): Fit robust expected shortfall kernel ridge regression
        ES(): Fit expected shortfall kernel ridge regression via sklearn.KernelRidge
        ls_predict(): Compute predicted mean at test data
        qt_predict(): Compute predicted quantile at test data
        es_predict(): Compute predicted expected shortfall at test data
        bahadur(): Compute Bahadur representation of the expected shortfall estimator
        qt_loss(): Check or smoothed check loss
        qt_grad(): Compute the (sub)gradient of the (smoothed) check loss
        bw(): Compute the bandwidth (smoothing parameter)
        genK(): Generate the kernel matrix for test data

    Attributes:
        params (dict): a dictionary of kernel parameters;
            gamma (float), default is 1;
            coef0 (float), default is 1;
            degree (int), default is 3.
            rbf : exp(-gamma*||x-y||_2^2)
            polynomial : (gamma*<x,y> + coef0)^degree
            laplacian : exp(-gamma*||x-y||_1)
    '''
    params = {'gamma': 1, 'coef0': 1, 'degree': 3}


    def __init__(self, X, Y, normalization=None, 
                 kernel='rbf', kernel_params=dict(),
                 smooth_method='convolution', 
                 min_bandwidth=1e-6, n_jobs=None):
        ''' 
            Initialize the KRR object

        Args:
            X (ndarray): n by p matrix of covariates;
                         n is the sample size, p is the number of covariates.
            Y (ndarray): response/target variable.
            normalization (str): method for normalizing covariates;
                                 should be one of [None, 'MinMax', 'Z-score'].
            kernel (str): type of kernel function; 
                          choose one from ['rbf', 'polynomial', 'laplacian'].
            kernel_params (dict): a dictionary of user-specified kernel parameters; 
                                  default is in the class attribute.
            smooth_method (str): method for smoothing the check loss;
                                 choose one from ['convolution', 'moreau'].
            min_bandwidth (float): minimum value of the bandwidth; default is 1e-4.
            n_jobs (int): the number of parallel jobs to run; default is None.

        Attributes:
            n (int) : number of observations
            Y (ndarray) : target variable
            nm (str) : method for normalizing covariates
            kernel (str) : type of kernel function
            params (dict) : a dictionary of kernel parameters
            X0 (ndarray) : normalized covariates
            xmin (ndarray) : minimum values of the original covariates
            xmax (ndarray) : maximum values of the original covariates
            xm (ndarray) : mean values of the original covariates
            xsd (ndarray) : standard deviations of the original covariates
            K (ndarray) : n by n kernel matrix
        '''

        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization
        self.kernel = kernel
        self.params.update(kernel_params)

        if normalization is None:
            self.X0 = X[:]
        elif normalization == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X[:] - self.xmin)/(self.xmax - self.xmin)
        elif normalization == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X[:] - self.xm)/self.xsd
        
        self.min_bandwidth = min_bandwidth
        self.n_jobs = n_jobs
        self.smooth_method = smooth_method
        self.fit_q = None

        # compute the kernel matrix
        self.K = PK(self.X0, metric=kernel, filter_params=True,
                    n_jobs=self.n_jobs, **self.params)


    def genK(self, x):
        ''' Generate the kernel matrix for test data '''
        if np.ndim(x) == 1:
            x = x.reshape(1, -1)
        if self.nm == 'MinMax':
            x = (x - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            x = (x - self.xm)/self.xsd
        
        # return an n * m matrix, m is test data size
        return PK(self.X0, x, metric=self.kernel, 
                  filter_params=True, n_jobs=self.n_jobs, 
                  **self.params)


    def qt_loss(self, x, h=0):
        '''
            Check or smoothed check loss
        '''
        tau = self.tau
        if h == 0:
            out = np.where(x > 0, tau * x, (tau - 1) * x)
        elif self.smooth_method == 'convolution':
            out = (tau - norm.cdf(-x/h)) * x \
                  + (h/2) * np.sqrt(2/np.pi) * np.exp(-(x/h) ** 2 /2)
        elif self.smooth_method == 'moreau':
            out = np.where(x > tau*h, tau*x - tau**2 * h/2, 
                           np.where(x < (tau - 1)*h, 
                                    (tau - 1)*x - (tau - 1)**2 * h/2, 
                                    x**2/(2*h)))
        return np.sum(out)


    def qt_grad(self, x, h=0):
        '''
            Gradient/subgradient of the (smoothed) check loss
        '''
        if h == 0:
            return np.where(x >= 0, self.tau, self.tau - 1)
        elif self.smooth_method == 'convolution':
            return self.tau - norm.cdf(-x / h)
        elif self.smooth_method == 'moreau':
            return np.where(x > self.tau * h, self.tau, 
                            np.where(x < (self.tau - 1) * h, 
                                     self.tau - 1, x/h))


    def bw(self, exponent=1/3, alpha=1, intercept=True):
        '''
            Compute the bandwidth (smoothing parameter)

        Args: 
            exponent (float): the exponent in the formula; default is 1/3.
        '''
        self.ls(alpha=alpha, intercept=intercept)
        krr_res = self.Y - self.ls_predict(self.X0)
        return max(self.min_bandwidth, 
                   np.std(krr_res)/self.n ** exponent)


    def ls(self, alpha=1, intercept=True, x=None):
        '''
            Fit kernel ridge mean (least squares) regression (closed-form solution)
        '''
        self.alpha_m = alpha
        n = self.n
        if intercept:
            y_mean = np.mean(self.Y)
            K_mean = self.K.mean(axis=0)
            self.coef_m = np.linalg.solve(self.K - K_mean + alpha*np.eye(n),
                                          self.Y - y_mean)
            self.bias_m = y_mean - K_mean @ self.coef_m
        else:
            self.coef_m = np.linalg.solve(self.K + alpha*np.eye(n), self.Y)
            self.bias_m = 0

        self.fit_m = self.K @ self.coef_m + self.bias_m
        self.res_m = self.Y - self.fit_m

        if x is not None:
            self.pred_m = self.ls_predict(x)


    def qt(self, tau=0.5, alpha=1, init=None, intercept=True, 
           smooth=False, h=0., method='L-BFGS-B', solver='osqp',
           tol=1e-8, options=None, x=None):
        '''
            Fit quantile kernel ridge regression

        Args:
            tau (float): quantile level between 0 and 1; default is 0.5.
            alpha (float): regularization parameter; default is 1.
            init (ndarray): initial values for optimization; default is None.
            intercept (bool): whether to include intercept term; 
                              default is True.
            smooth (bool): a logical flag for using smoothed check loss; 
                           default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            method (str): type of solver if smoothing (h>0) is used;
                          choose one from ['BFGS', 'L-BFGS-B'].
            solver (str): type of QP solver if check loss is used; 
                          default is 'osqp' (https://osqp.org/docs/).
            tol (float): tolerance for termination; default is 1e-7.
            options (dict): a dictionary of solver options; default is None.
        
        Attributes:
            sol_q (OptimizeResult): solution of the optimization problem
            bias_q (float): intercept term
            coef_q (ndarray): quantile KRR coefficients
            fit_q (ndarray): fitted quantiles (in-sample)
        '''
        self.alpha_q, self.tau, self.itcp_q = alpha, tau, intercept
        if smooth and h == 0: 
            h = np.minimum(self.bw(alpha=alpha, intercept=intercept), 1e-1)
        n, self.h = self.n, h

        # compute smoothed quantile KRR estimator with bandwidth h
        if self.h > 0: 
            if intercept:
                x0 = init if init is not None else np.zeros(n + 1)
                x0[0] = np.quantile(self.Y, tau)
                res = lambda x: self.Y - x[0] - self.K @ x[1:]
                func = lambda x: self.qt_loss(res(x), h) + \
                                 (alpha/2) * np.dot(x[1:], self.K @ x[1:])
                grad = lambda x: np.insert(-self.K @ self.qt_grad(res(x),h) 
                                           + alpha*self.K @ x[1:], 
                                           0, np.sum(-self.qt_grad(res(x),h)))
                self.sol_q = minimize(func, x0, method=method, 
                                      jac=grad, tol=tol, options=options)
                self.bias_q = self.sol_q.x[0]
                self.coef_q = self.sol_q.x[1:]
                self.fit_q = self.bias_q + self.K @ self.coef_q
            else:
                x0 = init if init is not None else np.zeros(n)
                res = lambda x: self.Y - self.K @ x
                func = lambda x: self.qt_loss(res(x), h) \
                                 + (alpha/2) * np.dot(x, self.K @ x)
                grad = lambda x: -self.K @ self.qt_grad(res(x), h) \
                                 + alpha * self.K @ x
                self.sol_q = minimize(func, x0=x0, method=method, 
                                      jac=grad, tol=tol, options=options)
                self.coef_q = self.sol_q.x
                self.bias_q = 0
                self.fit_q = self.K @ self.coef_q
        else: # fit quantile KRR by solving a quadratic program
            self.itcp_q = True
            C = 1 / alpha
            lb = C * (tau - 1)
            ub = C * tau
            prob = Problem(P=csc_matrix(self.K), q=-self.Y, G=None, h=None, 
                           A=csc_matrix(np.ones(n)), b=np.array([0.]), 
                           lb=lb * np.ones(n), ub=ub * np.ones(n))
            self.sol_q = solve_problem(prob, solver=solver)
            self.coef_q = self.sol_q.x
            self.fit_q = self.K @ self.coef_q
            self.bias_q = np.quantile(self.Y - self.fit_q, tau)
            self.fit_q += self.bias_q

        if x is not None:
            self.pred_q = self.qt_predict(x)


    def es(self, tau=0.5, alpha=1, intercept=True, fit_q=None, 
           alpha_q=1, smooth=False, h=0., method='L-BFGS-B', solver='osqp',
           tol_q=1e-7, options=None, x=None):
        '''
            Fit expected shortfall kernel ridge regression (closed-form solution)
        '''
        if fit_q is None:
            self.qt(tau, alpha_q, None, intercept, smooth, h, 
                    method, solver, tol_q, options)
            fit_q = self.fit_q
        elif len(fit_q) != self.n:
            raise ValueError("Length of fit_q should be equal to \
                              the number of observations.")
        
        self.alpha_e, self.tau, self.itcp_e = alpha, tau, intercept
        n = self.n
        Z = np.minimum(self.Y - fit_q, 0)/tau + fit_q
        if intercept:
            z_mean = np.mean(Z)
            K_mean = self.K.mean(axis=0)
            self.coef_e = np.linalg.solve(self.K - K_mean + alpha*np.eye(n),
                                          Z - z_mean)
            self.bias_e = z_mean - K_mean @ self.coef_e
        else: 
            self.coef_e = np.linalg.solve(self.K + alpha*np.eye(n), Z)
            self.bias_e = 0
        
        self.fit_e = self.K @ self.coef_e + self.bias_e
        self.res_e = Z - self.fit_e

        if x is not None:
            self.pred_e = self.es_predict(x)


    def res(self, tau=0.5, alpha=1, init=None, intercept=True, c=None,
            fit_q=None,  alpha_q=1,  smooth=False, h=0., 
            method='L-BFGS-B', solver='osqp', 
            tol_q=1e-7, tol_e=1e-7, options=None, x=None):
        """ 
            Fit robust expected shortfall kernel ridge regression
        
        Args:
            tau (float): quantile level between 0 and 1; default is 0.5.
            alpha (float): regularization parameter; default is 1.
            init (ndarray): initial values for optimization; default is None.
            intercept (bool): whether to include intercept term; 
                              default is True.
            c (float): positive tuning parameter for the Huber estimator; 
                       if not specified, it will be automatically chosen.
            fit_q (ndarray): fitted quantiles from the first step; 
                             default is None.
            alpha_q (float): regularization parameter for the first step;
            smooth (bool): a logical flag for using smoothed check loss; 
                           default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            
            method (str): type of solver if smoothing (h>0) is used;
                          choose one from ['BFGS', 'L-BFGS-B'].
            solver (str): type of QP solver if check loss is used; 
                          default is 'osqp' (https://osqp.org/docs/).
            tol_q (float): tolerance for termination in qt-KRR; 
                           default is 1e-7.
            tol_e (float): tolerance for termination in es-KRR; 
                            default is 1e-7.
            options (dict): a dictionary of solver options; default is None.
    
        Attributes:
            sol_e (OptimizeResult): solution of the optimization problem
            coef_e (ndarray): expected shortfall KRR coefficients
            bias_e (float): intercept term
            fit_e (ndarray): fitted expected shortfalls (in-sample)
            res_e (ndarray): fitted expected shortfall residuals
        """
        if fit_q is None:
            self.qt(tau, alpha_q, None, intercept, smooth, h, 
                    method, solver, tol_q, options)
            fit_q = self.fit_q
        elif len(fit_q) != self.n:
            raise ValueError("Length of fit_q should be equal to \
                              the number of observations.")
        
        self.alpha_e, self.tau, self.itcp_e = alpha, tau, intercept
        n = self.n
        
        nres_q = np.minimum(self.Y - fit_q, 0)
        if c is None:
            c = np.std(nres_q) * (n/np.log(n))**(1/3) / tau
        self.c = c
        Z = nres_q/tau + fit_q # surrogate response
        if intercept:
            x0 = init if init is not None else np.zeros(n + 1)
            x0[0] = np.mean(Z)
            res = lambda x: Z - x[0] - self.K @ x[1:]
            func = lambda x: HuberLoss(res(x), c) + \
                             (alpha/2) * np.dot(x[1:], self.K @ x[1:])
            grad = lambda x: np.insert(-self.K @ HuberGrad(res(x), c)
                                       + alpha * self.K @ x[1:],
                                       0, -np.sum(HuberGrad(res(x), c)))
            self.sol_e = minimize(func, x0, method=method, 
                                  jac=grad, tol=tol_e, options=options)
            self.bias_e = self.sol_e.x[0]
            self.coef_e = self.sol_e.x[1:]
        else:
            x0 = init if init is not None else np.zeros(n)
            res = lambda x: Z - self.K @ x
            func = lambda x: HuberLoss(res(x), c) \
                             + (alpha/2) * np.dot(x, self.K @ x)
            grad = lambda x: -self.K @ HuberGrad(res(x), c)  \
                             + alpha * self.K @ x
            self.sol_e = minimize(func, x0=x0, method=method, 
                                  jac=grad, tol=tol_e, options=options)
            self.coef_e = self.sol_e.x
            self.bias_e = 0
        self.fit_e = self.bias_e + self.K @ self.coef_e
        self.res_e = Z - self.fit_e

        if x is not None:
            self.pred_e = self.es_predict(x) 


    def qt_seq(self, tau=0.5, alpha_seq=np.array([0.1]), order='ascend', 
               intercept=True, smooth=False, h=0., method='L-BFGS-B', 
               solver='osqp', tol=1e-8, options=None, x=None):
        '''
            Fit a sequence of quantile kernel ridge regressions
        '''
        if order == 'ascend':
            alphas = np.sort(alpha_seq)
        else:
            alphas = np.sort(alpha_seq)[::-1]
        args = [intercept, smooth, h, method, solver, tol, options]

        x0 = None
        bias, coef, fit = [], [], [] 
        for alpha in alphas:
            self.qt(tau, alpha, x0, *args)
            bias.append(self.bias_q)
            coef.append(self.coef_q)
            fit.append(self.fit_q)
            x0 = self.coef_q
            if self.itcp_q:
                x0 = np.insert(x0, 0, self.bias_q)

        self.alpha_q = alphas
        self.bias_q = np.array(bias)
        self.coef_q = np.array(coef).T
        self.fit_q = np.array(fit).T

        if x is not None:
            self.pred_q = self.qt_predict(x)

    
    def es_seq(self, tau=0.5, alpha_seq=np.array([0.1]), 
               order='ascend', intercept=True, x=None):
        '''
            Fit a sequence of expected shortfall kernel ridge regressions
        '''
        self.tau = tau
        if order == 'ascend':
            alphas = np.sort(alpha_seq)
        else:
            alphas = np.sort(alpha_seq)[::-1]
        
        if self.fit_q is None: 
            raise ValueError("Fit quantile KRR first.")
        
        bias, coef, fit = [], [], [] 
        for alpha in alphas:
            self.es(tau, alpha, intercept, self.fit_q)
            bias.append(self.bias_e)
            coef.append(self.coef_e)
            fit.append(self.fit_e)

        self.alpha_e = alphas
        self.bias_e = np.array(bias)
        self.coef_e = np.array(coef).T
        self.fit_e = np.array(fit).T

        if x is not None:
            self.pred_e = self.es_predict(x)


    def ls_predict(self, x):
        '''
            Compute predicted mean at new input x

        Args:
            x (ndarray): new input.
        '''
        return self.genK(x).T @ self.coef_m + self.bias_m
    

    def qt_predict(self, x): 
        '''
            Compute predicted quantile at new input x
        
        Args:
            x (ndarray): new input.
        '''
        return self.genK(x).T @ self.coef_q + self.bias_q


    def es_predict(self, x): 
        '''
            Compute predicted expected shortfall at new input x
        
        Args:
            x (ndarray): new input.
        '''
        return self.genK(x).T @ self.coef_e + self.bias_e


    def bahadur(self, x):
        '''
            Compute Bahadur representation of the expected shortfall estimator
        '''        
        A = self.K + self.alpha_e * np.eye(self.n)
        return np.linalg.solve(A, self.genK(x)) \
               * self.res_e.reshape(-1,1)


    def ES(self, tau=.5, alpha=1, kernel='rbf', other_params=None, x=None):
        '''
            Expected shortfall kernel ridge regression (via sklearn.KernelRidge)
        '''
        self.tau, self.alpha_e = tau, alpha
        if x is not None:
            x = np.array(x)
            if np.ndim(x) <= 1:
                x = x.reshape(-1, 1)
            if self.nm == 'MinMax':
                x = (x - self.xmin)/(self.xmax - self.xmin)
            elif self.nm == 'Z-score':
                x = (x - self.xm)/self.xsd

        self.model_e = KR(alpha, kernel=kernel, 
                          gamma=self.params['gamma'], 
                          coef0=self.params['coef0'],
                          degree=self.params['degree'], 
                          kernel_params=other_params)
        if self.fit_q is None:
            raise ValueError("Fit quantile KRR first.")
        elif self.fit_q.ndim == 1:
            Y0 = np.minimum(self.Y - self.fit_q, 0)/tau + self.fit_q
            self.model_e.fit(self.X0, Y0)
            self.fit_e = self.model_e.predict(self.X0)
            self.res_e = Y0 - self.fit_e
            if x is not None:
                self.pred_e = self.model_e.predict(x)
        elif self.fit_q.ndim == 2:
            Y0 = self.Y.reshape(-1, 1)
            Y0 = np.minimum((Y0 - self.fit_q),
                             np.zeros(self.fit_q.shape))/tau + self.fit_q
            self.fit_e = np.empty(Y0.shape)
            self.res_e = np.empty(Y0.shape)
            if x is not None:
                self.pred_e = np.empty((x.shape[0], Y0.shape[1]))
            for i in range(self.fit_q.shape[1]):
                self.model_e.fit(self.X0, Y0[:,i])
                self.fit_e[:,i] = self.model_e.predict(self.X0)
                self.res_e[:,i] = Y0[:,i] - self.fit_e[:,i]
                if x is not None:
                    self.pred_e[:,i] = self.model_e.predict(x)



    def ES_seq(self, tau=.5, alpha_seq=np.array([.1]), 
               kernel='rbf', other_params=None, x=None):
        '''
            Fit a sequence of expected shortfall kernel ridge regressions
        '''
        self.tau, self.alpha_e = tau, alpha_seq
        if x is not None:
            if np.ndim(x) == 1:
                x = x.reshape(-1, 1)
            if self.nm == 'MinMax':
                x = (x - self.xmin)/(self.xmax - self.xmin)
            elif self.nm == 'Z-score':
                x = (x - self.xm)/self.xsd

        if self.fit_q is None: 
            raise ValueError("Fit quantile KRR first.")
        Y0 = np.minimum(self.Y - self.fit_q, 0)/tau + self.fit_q
        args = {'kernel': kernel, 
                'gamma': self.params['gamma'],
                'coef0': self.params['coef0'], 
                'degree': self.params['degree'],
                'kernel_params': other_params}
        self.fit_e = np.empty((Y0.shape[0], len(alpha_seq)))
        self.res_e = np.empty((Y0.shape[0], len(alpha_seq)))
        if x is not None:
            self.pred_e = np.empty((x.shape[0], len(alpha_seq)))
        for i, alpha in enumerate(alpha_seq):
            self.model_e = KR(alpha, **args)
            self.model_e.fit(self.X0, Y0)
            self.fit_e[:,i] = self.model_e.predict(self.X0)
            self.res_e[:,i] = Y0 - self.fit_e[:,i]
            if x is not None:
                self.pred_e[:,i] = self.model_e.predict(x)



##############################################################################
######################### Local Polynomial Regression ########################
##############################################################################
class LocPoly:
    '''
    Local Polynomial Regression

    Methods:
        __init__(): Initialize the LocPoly object
        get_features(): Generate polynomial features
        get_weights(): Compute the weights for local polynomial regression
        qt_loss(): Check or smoothed check loss
        qt_grad(): Compute the (sub)gradient of the (smoothed) check loss
        qt(): Local polynomial quantile regression
        qt_predict(): Compute predicted quantile at test data
        ls(): Local polynomial least squares regression
        ls_predict(): Compute predicted mean at test data
        es(): Local polynomial expected shortfall regression

    Attributes:
        n (int): number of observations
        Y (ndarray): target variable
        nm (str): method for normalizing covariates
        kernel_fn (function): kernel function
        X0 (ndarray): normalized covariates
        xmin (ndarray): minimum values of the original covariates
        xmax (ndarray): maximum values of the original covariates
        xm (ndarray): mean values of the original covariates
        xsd (ndarray): standard deviations of the original covariates
        smooth_method (str): method for smoothing the check loss
    '''
    def __init__(self, X, Y, 
                 normalization=None, 
                 kernel=norm.pdf, 
                 smooth_method='convolution'):
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization
        if callable(kernel):
            self.kernel_fn = kernel
        else:
            raise ValueError('Invalid kernel function')

        if normalization is None:
            self.X0 = X[:]
        elif normalization == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X[:] - self.xmin)/(self.xmax - self.xmin)
        elif normalization == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X[:] - self.xm)/self.xsd
        
        self.smooth_method = smooth_method


    def get_features(self, x, degree=1):
        '''
            Generate polynomial features
        '''
        poly = PF(degree, interaction_only=False, include_bias=True)
        return poly.fit_transform(self.X0 - x)


    def get_weights(self, x, bw=.5):
        '''
            Compute the weights for local polynomial regression
        '''
        x = np.array(x).reshape(-1)
        bw = np.array(bw).reshape(-1)
        W = self.kernel_fn((self.X0 - x) / bw).prod(axis=1) 
        if not np.any(W):
            raise ValueError('Bandwidth too small')
        else:
            return W


    def qt_loss(self, x, tau=.5, h=0., w=None):
        '''
            Check or smoothed check loss
        '''
        if h == 0:
            out = np.where(x > 0, tau * x, (tau - 1) * x)
        elif self.smooth_method == 'convolution':
            out = (tau - norm.cdf(-x/h)) * x \
                  + (h/2) * np.sqrt(2/np.pi) * np.exp(-(x/h) ** 2 /2)
        elif self.smooth_method == 'moreau':
            out = np.where(x > tau*h, tau*x - tau**2 * h/2, 
                           np.where(x < (tau - 1)*h, 
                                    (tau - 1)*x - (tau - 1)**2 * h/2, 
                                    x**2/(2*h)))
        if w is not None:
            return np.sum(out * w)
        else:
            return np.sum(out)


    def qt_grad(self, x, tau=.5, h=0.):
        '''
            Gradient/subgradient of the (smoothed) check loss
        '''
        if h == 0:
            return np.where(x >= 0, tau, tau - 1)
        elif self.smooth_method == 'convolution':
            return tau - norm.cdf(-x / h)
        elif self.smooth_method == 'moreau':
            return np.where(x > tau * h, tau, 
                            np.where(x < (tau - 1) * h, 
                                     tau - 1, x/h))


    def qt(self, x, tau=0.5, bw=0.5, h=0., degree=1,
           method='L-BFGS-B', tol=1e-7, options=None):
        '''
            Local polynomial quantile regression

        Args:
            x: 1D array, the query point
            tau: float, quantile level
            bw: float, bandwidth
            h: float, smoothing parameter
            degree: int, degree of the polynomial
            method: str, optimization method
            tol: float, tolerance
            options: dict, optimization options
        '''
        y = self.Y
        Z = self.get_features(x, degree)
        W = self.get_weights(x, bw)
        fn = lambda b: self.qt_loss(y - Z@b, tau, h, W)
        gd = lambda b: -Z.T @ (W * self.qt_grad(y - Z@b, tau, h))
        x0 = np.zeros(Z.shape[1])
        x0[0] = weighted_quantile(y, [tau], W)[0]
        sol = minimize(fn, x0, jac=gd, 
                       method=method, tol=tol, options=options)
        self.bias_q = sol.x[0]
        self.coef_q = sol.x[1:]
        self.sol_q = sol

    
    def qt_predict(self, x0, tau=.5, bw=0.1, h=0., degree=1,
                   method='L-BFGS-B', tol=1e-7, options=None):
        '''
            Local polynomial quantile regression

        Args:
            x0: 1D or 2D array, the query point(s)
            tau: float, quantile level
            bw: float, bandwidth
            h: float, smoothing parameter
            degree: int, degree of the polynomial
            method: str, optimization method
            tol: float, tolerance
            options: dict, optimization options
        '''
        if np.ndim(x0) == 1:
            self.qt(x0, tau, bw, h, degree, method, tol, options)
            return self.bias_q
        elif np.ndim(x0) == 2:
            fit_q = []
            for x in x0:
                self.qt(x, tau, bw, h, degree, method, tol, options)
                fit_q.append(self.bias_q)
            return np.array(fit_q)


    def ls(self, x0, bw=.5, degree=1):
        '''
            Local polynomial least squares regression

        Args:
            x0: 1D array, the query point
            bw: float, bandwidth
            degree: int, degree of the polynomial
        '''
        Z = self.get_features(x0, degree)
        ZW = Z.T * self.get_weights(x0, bw)
        class sol: x = np.linalg.solve(ZW @ Z,  ZW @ self.Y)
        self.bias_m = sol.x[0]
        self.coef_m = sol.x[1:]
        self.sol_m = sol


    def ls_predict(self, x0, bw=.5, degree=1):
        '''
            Local polynomial least squares regression

        Args:
            x0: 1D or 2D array, the query point(s)
            bw: float, bandwidth
            degree: int, degree of the polynomial
        '''
        if np.ndim(x0) == 1:
            self.ls(x0, bw, degree)
            return self.bias_m
        elif np.ndim(x0) == 2:
            fit_m = []
            for x in x0:
                self.ls(x, bw, degree)
                fit_m.append(self.bias_m)
            return np.array(fit_m)
        
    
    def es(self, x0, tau=0.5, bw=.5, degree=1, fit_q=None,
           method='L-BFGS-B', tol=1e-7, options=None):
        '''
            Local polynomial expected shortfall regression

        Args: 
            x0: 1D array, the query point
            tau: float, quantile level
            bw: float, bandwidth
            degree: int, degree of the polynomial
            fit_q: 1D array, fitted quantiles on the training data
            method: str, optimization method
            tol: float, tolerance
            options: dict, optimization options
        '''
        if fit_q is None:
            fit_q=self.qt_predict(self.X0, tau, bw, degree=degree, 
                                  method=method, tol=tol, options=options)
        Y0 = np.minimum(self.Y - fit_q, 0)/tau + fit_q
        if np.ndim(x0) == 1:
            Z = self.get_features(x0, degree)
            ZW = Z.T * self.get_weights(x0, bw)
            estimator = np.linalg.solve(ZW @ Z,  ZW @ Y0)
            return estimator[0]
        elif np.ndim(x0) == 2:
            fit_e = []
            for x in x0:
                Z = self.get_features(x, degree)
                ZW = Z.T * self.get_weights(x, bw)
                estimator = np.linalg.solve(ZW @ Z,  ZW @ Y0)
                fit_e.append(estimator[0])
            return np.array(fit_e)
        


###############################################################################
######################### Neural Network Regression ###########################
###############################################################################
class FNN:
    '''
    Feedforward Neural Network Regression
    
    Methods:
        __init__(): Initialize the FNN object
        plot_losses(): Plot the training and validation losses
        qt(): Fit (smoothed) quantile neural network regression
        es(): Fit (robust) expected shortfall neural network regression
        ls() : Fit least squares neural network regression
        joint(): Fit joint quantile/ES neural network regression
        predict(): Compute predicted  outcomes at test data
        trainer(): Train the neural network model
    '''
    optimizers = ["sgd", "adam"]
    activations = ["sigmoid", "tanh", "relu", "leakyrelu"]
    params = {'batch_size' : 64, 'val_pct' : .25, 'step_size': 10,
              'activation' : 'relu', 'depth': 4, 'width': 256,  
              'optimizer' : 'adam', 'lr': 1e-3, 
              'lr_decay': 1., 'n_epochs' : 200,
              'dropout_rate': .0, 'Lambda': .0, 'weight_decay': .0, 
              'momentum': 0.9, 'nesterov': True,
              'stop_early': False, 'early_stopping_step': 0, 
              'early_stopping_best_val': 1e8, 'early_stopping_patience': 10}


    def __init__(self, X, Y, normalization=None):
        '''
        Args:
            X (ndarry): n by p matrix of covariates; 
                        n is the sample size, p is the number of covariates.
            Y (ndarry): response/target variable.
            normalization (str): method for normalizing covariates;
                                 should be one of [None, 'MinMax', 'Z-score'].

        Attributes:
            Y (ndarray): response variable.
            X0 (ndarray): normalized covariates.
            n (int): sample size.
            nm (str): method for normalizing covariates.
        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization

        if self.nm is None:
            self.X0 = X
        elif self.nm == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X - self.xm)/self.xsd


    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.style.use('fivethirtyeight')
        plt.plot(self.results['losses'],
                 label='Training Loss', color='C0', linewidth=2)
        if self.params['val_pct'] > 0:
            plt.plot(self.results['val_losses'],
                     label='Validation Loss', color='C1', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig


    def joint(self, tau=0.5, options=dict(), G1=False, G2_type=1,
              plot=False, device='cpu'):
        '''
            Fit joint quantile/expected shortfall neural network regression

        Args:
            tau (float): quantile level between 0 and 1; default is 0.5.
            options (dictionary): see qt() or es() for details.
            G1 : logical flag for the specification function G1 in FZ's loss;
                 G1(x)=0 if G1=False, and G1(x)=x and G1=True.
            G2_type : an integer (from 1 to 5) that indicates the type.
                        of the specification function G2 in FZ's loss.
            plot (boolean) : whether to plot loss values at iterations.
            device (string): device to run the model on; default is 'cpu'.
        '''
        self.params.update(options)
        self.device = device
        self.tau = tau
        self.results = self.trainer(self.X0, self.Y, 
                                    JointLoss(tau, G1, G2_type), 2, 
                                    device, name='_joint')
        if plot: self.fig = self.plot_losses()
        self.model = self.results['model']
        self.fit = self.results['fit']


    def qt(self, tau=0.5, smooth=False, h=0., options=dict(),
           plot=False, device='cpu', min_bandwidth=1e-4):
        '''
            Fit (smoothed) quantile neural network regression

        Args: 
            tau (float): quantile level between 0 and 1; default is 0.5.
            smooth (boolean): a logical flag for using smoothed check loss; default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            options (dictionary): a dictionary of neural network and optimization parameters.
                batch_size (int): the number of training examples used in one iteration; 
                                  default is 64.
                val_pct (float): the proportion of the training data to use for validation;
                                   default is 0.25.
                step_size (int): the number of epochs of learning rate decay; default is 10.
                activation (string): activation function; default is the ReLU function.
                depth (int): the number of hidden layers; default is 4.
                width (int): the number of neurons for each layer; default is 256.
                optimizer (string): the optimization algorithm; default is the Adam optimizer.
                lr (float): , learning rate of SGD or Adam optimization; default is 1e-3.
                lr_decay (float): multiplicative factor by which the learning rate will be reduced;
                                  default is 0.95.
                n_epochs (int): the number of training epochs; default is 200.
                dropout_rate : proportion of the dropout; default is 0.
                Lambda (float): L_1-regularization parameter; default is 0.
                weight_decay (float): weight decay of L2 penalty; default is 0.
                momentum (float): momentum accerleration rate for SGD algorithm; 
                                  default is 0.9.
                nesterov (boolean): whether to use Nesterov gradient descent algorithm;
                                    default is TRUE.
            plot (boolean) : whether to plot loss values at iterations.
            device (string): device to run the model on; default is 'cpu'.
            min_bandwidth (float): minimum value of the bandwidth; default is 1e-4.
        '''
        self.params.update(options)
        self.device = device
        self.tau = tau
        if smooth and h == 0:
            h = max(min_bandwidth, 
                    (tau - tau**2)**0.5 / self.n ** (1/3))
        self.h = h if smooth else 0.   # bandwidth for smoothing
        self.results = self.trainer(self.X0, self.Y, QuantLoss(tau, h), 1,
                                    device, QuantLoss(tau, 0), name='_qt')
        if plot: self.fig = self.plot_losses()
        self.model = self.results['model']
        self.fit = self.results['fit']


    def es(self, tau=0.5, robust=False, c=None, 
           fit_q=None, smooth=False, h=0., 
           options=dict(), plot=False, device='cpu'):
        '''
            Fit (robust) expected shortfall neural network regression
        '''
        self.params.update(options)
        self.device = device
        self.tau = tau
        if fit_q is None:
            self.qt(tau, smooth, h, options, False, device)
            fit_q = self.fit
        elif len(fit_q) != self.n:
            raise ValueError("Length of fit_q should equal the number of observations.")
        Z = np.minimum(self.Y - fit_q, 0)/tau
        if robust == True and c is None:
            c = np.std(Z) * (self.n / np.log(self.n))**(1/3)
        self.c = c
        loss_fn = nn.MSELoss(reduction='mean') if not robust \
                    else nn.HuberLoss(reduction='mean', delta=c)
        self.results = self.trainer(self.X0, Z+fit_q, loss_fn, 1, 
                                    device, '_es')
        if plot: self.fig = self.plot_losses()
        self.model = self.results['model']
        self.fit = self.results['fit']


    def ls(self, options=dict(), robust=False, c=None, s=1., 
           plot=False, device='cpu'):
        ''' 
            Fit least squares neural network regression 
            or its robust version with Huber loss
        '''
        self.params.update(options)
        self.device = device
        if robust == True and c is None:
            ls_res = self.Y - self.results['fit']
            scale = s * np.std(ls_res) + (1 - s) * mad(ls_res)
            c = scale * (self.n / np.log(self.n))**(1/3)
        self.c = c
        loss_fn = nn.MSELoss(reduction='mean') if not robust \
                    else nn.HuberLoss(reduction='mean', delta=c)
        self.results = self.trainer(self.X0, self.Y, loss_fn, 1,
                                    device, name='_ls')
        if plot: self.fig = self.plot_losses()
        self.model = self.results['model']
        self.fit = self.results['fit']


    def predict(self, X):
        ''' Compute predicted outcomes at new input X '''
        if self.nm == 'MinMax':
            X = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            X = (X - self.xm)/self.xsd
        Xnew = torch.as_tensor(X, dtype=torch.float).to(self.device)
        return self.model.predict(Xnew)


    def trainer(self, x, y, loss_fn, output_dim=1,
                device='cpu', val_fn=None, name=''):
        '''
            Train an MLP model with given loss function
        '''
        name = 'checkpoint' + name + '.pth'
        input_dim = x.shape[1]
        x_tensor = torch.as_tensor(x).float()
        y_tensor = torch.as_tensor(y).float()
        dataset = TensorDataset(x_tensor, y_tensor)
        n_total = len(dataset)
        n_val = int(self.params['val_pct'] * n_total)
        train_data, val_data = random_split(dataset, [n_total - n_val, n_val])

        train_loader = DataLoader(train_data, 
                                  batch_size=self.params['batch_size'], 
                                  shuffle=True, 
                                  drop_last=True)
        if self.params['val_pct'] > 0:
            val_loader = DataLoader(val_data, 
                                    batch_size=self.params['batch_size'], 
                                    shuffle=False)
        
        # initialize the model
        model = MLP(input_dim, output_dim, options=self.params).to(device)
        
        # choose the optimizer
        if self.params['optimizer'] == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.params['lr'],
                                  weight_decay=self.params['weight_decay'],
                                  nesterov=self.params['nesterov'],
                                  momentum=self.params['momentum'])
        elif self.params['optimizer'] == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.params['lr'],
                                   weight_decay=self.params['weight_decay'])
        else:
            raise Exception(self.params['optimizer'] 
                            + "is currently not available")

        if val_fn is None: val_fn = loss_fn
        train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
        val_step_fn = make_val_step_fn(model, val_fn)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=self.params['step_size'],
                                              gamma=self.params['lr_decay'])

        losses, val_losses = [], []
        for epoch in range(self.params['n_epochs']):
            
            loss = mini_batch(device, train_loader, train_step_fn)
            losses.append(loss)

            if self.params['val_pct'] > 0:
                with torch.no_grad():
                    val_loss = mini_batch(device, val_loader, val_step_fn)
                    val_losses.append(val_loss)
                    loss_t = val_losses[-1]
                 
                # if loss worsens, increase early stopping step
                if loss_t >= self.params['early_stopping_best_val']:
                    self.params['early_stopping_step'] += 1
                else:
                    # if loss improves, save the model
                    torch.save({'epoch': epoch+1,
                                'model_state_dict': model.state_dict(),
                                'best_val_loss': loss_t,
                                'val_losses': val_losses,
                                'losses': losses}, name)
                    self.params['early_stopping_best_val'] = loss_t

                    # reset early stopping step
                    self.params['early_stopping_step'] = 0

                # learning rate decay
                scheduler.step()
                
                self.params['stop_early'] \
                    = self.params['early_stopping_step'] \
                        >= self.params['early_stopping_patience']
                
            if self.params['stop_early']: break

        if self.params['val_pct'] > 0:
            checkpoint = torch.load(name, weights_only=False)
            model1 = MLP(input_dim, output_dim, options=self.params).to(device)
            model1.load_state_dict(checkpoint['model_state_dict'])
        else:
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'losses': losses}, name)
            model1 = model
        
        # reset early stopping parameters
        self.params['stop_early'] = False
        self.params['early_stopping_best_val'] = 1e8
        self.params['early_stopping_step'] = 0

        return {'model': model1,
                'fit': model1.predict(x_tensor.to(device)),
                'losses': losses,
                'val_losses': val_losses,
                'total_epochs': epoch+1}


class Activation(Enum):
    ''' Activation functions '''
    relu = nn.ReLU()
    tanh = nn.Tanh()
    sigmoid = nn.Sigmoid()
    leakyrelu = nn.LeakyReLU()


class MLP(nn.Module):
    ''' Generate a multi-layer perceptron '''
    def __init__(self, input_size, output_size, options):
        super(MLP, self).__init__()

        activation = Activation[options.get('activation', 'relu')].value
        dropout = options.get('dropout_rate', 0)
        if type(options['width']) == int:
            layers = [input_size] + [options['width']] * options['depth']
        elif type(options['width']) == list \
            and len(options['width']) == options['depth']:
            layers = [input_size] + options['width']
        elif type(options['width']) == list \
            and len(options['width']) != options['depth']:
            options['depth'] = len(options['width'])
            layers = [input_size] + options['width']

        nn_structure = []
        for i in range(len(layers) - 1):
            nn_structure.extend([
                nn.Linear(layers[i], layers[i + 1]),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                activation
            ])
        nn_structure.append(nn.Linear(layers[-1], output_size))
        self.fc_in = nn.Sequential(*nn_structure)
        self.output_size = output_size

    def forward(self, x):
        return self.fc_in(x)

    def predict(self, X):
        with torch.no_grad():
            self.eval()
            if self.output_size == 1:
                yhat = self.forward(X)[:, 0]
            else:
                yhat = self.forward(X)
        return yhat.cpu().numpy()