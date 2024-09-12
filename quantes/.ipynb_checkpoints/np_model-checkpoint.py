import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
# https://qpsolvers.github.io/qpsolvers/supported-solvers.html#supported-solvers
from qpsolvers import solve_qp



class quantKRR:
    '''
    Quantile Kernel Ridge Regression
    '''    

    def __init__(self, X, Y, normalization=None):
        '''
        Arguments
        ---------
        X : ndarray, shape (n, p); each row is an observation vector
           
        Y : ndarray, shape (n,)

        normalization: string, method for normalizing covariates;
                       should be one of [None, 'MinMax', 'Z-score'] 
        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization
        
        if normalization is None:
            self.X0 = X
        if normalization == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X - self.xmin)/(self.xmax - self.xmin)
        if normalization == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X - self.xm)/self.xsd


    def ker_func(self, u, v, sigma=1, gamma=1, r=1, degree=3):
        '''
        Compute kernel function
        '''
        if self.kernel == 'RBF':
            tmp = u - v
            return np.exp(-np.dot(tmp,tmp)/(2 * sigma ** 2))
        if self.kernel == 'polynomial':
            return (gamma * np.dot(u, v) + r) ** degree


    def ker_mat(self, sigma=1, gamma=1, r=1, degree=3):
        '''
        Compute kernel matrix
        '''
        params = [sigma, gamma, r, degree]
        K = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                K[i, j] = self.ker_func(self.X0[i,:], self.X0[j,:], *params)
                K[j, i] = K[i, j]
        return K


    def _loss(self, x, h=0):
        '''
        Check or smoothed check loss
        '''
        if h == 0:
            return np.where(x>0, self.tau*x, (self.tau-1)*x)
        else:
            return (self.tau - norm.cdf(-x/h)) * x \
                   + 0.5 * h * np.sqrt(2 / np.pi) * np.exp(-(x/h) ** 2 / 2)
    
    
    def _sg(self, x, h=0):
        '''
        Compute the derivative of smoothed check loss
        '''
        if h == 0:
            return np.where(x>=0, self.tau, self.tau-1)
        else:
            return self.tau - norm.cdf(-x/h)
        
        
    def bw(self):
        '''
        Compute the bandwidth (smoothing parameter)
        '''
        return max(0.01, (self.tau-self.tau**2)**0.5 * self.n**-0.2)


    def fit(self, tau=0.5, alpha=0.01, 
            init=None, intercept=True,
            kernel='RBF', sigma=1, gamma=1, r=1, degree=3,
            h=0.1, method='L-BFGS-B', solver='clarabel',
            tol=1e-6, options=None):
        '''
        Fit (smoothed) quantile kernel ridge regression

            \min_f (1/n)\sum_{i=1}^n \rho_tau(y_i - f(x_i)) 
                   + (\alpha/2) \| f \|_H^2

        Arguments
        ---------
        tau : float, quantile level between 0 and 1
           
        alpha : float, positive ridge penalty level

        intercept : logical flag for fitting an intercept (bias) term

        kernel : string, choice of kernel;
                 should be either 'RBF' or 'polynomial' (temporary);
                 RBF: (u, v) \mapsto \exp(-\| u - v \|_2^2/(2 * sigma^2))
                 polynomial: (u, v) \mapsto (gamma * u'v + r)^degree

        sigma : float, positive value in the RBF kernel
                
        gamma : float, positive value in the polynomial kernel

        r : float, positive value in the polynomial kernel

        degree : int, positive integer in the polynomial kernel;

        h : float, positive bandwidth (smoothing) parameter

        method : string, type of solver if smoothing (h>0) is used;
                 should be one of ['BFGS', 'L-BFGS-B'], among others 
                 (https://docs.scipy.org/doc/scipy/reference/
                          generated/scipy.optimize.minimize.html)

        solver : string, type of QP solver if check loss is used;
                 should be one of [qpsolvers.available_solvers]
                 (https://pypi.org/project/qpsolvers/)

        tol : float, optional;
              tolerance for termination

        options : dict, optional;
                  a dictionary of solver options
        '''
        self.kernel, self.tau, self.h = kernel, tau, h
        self.params = [sigma, gamma, r, degree]
        self.itcp = intercept
        n = self.n
        self.K = self.ker_mat(*self.params)

        if self.h > 0 : # compute smoothed KRR-Q estimator with bandwidth h
                        # use gradient methods from 'scipy.optimize.minimize'
                        # method = 'L-BFGS-B' or method = 'BFGS'    
            if intercept:
                if init is not None:
                    x0 = init
                else:
                    x0 = np.zeros(n + 1)
                    x0[0] = np.quantile(self.Y, tau)
                res = lambda x: self.Y - x[0] - self.K@x[1:]
        
                func = lambda x: np.mean(self._loss(res(x),h))
                grad = lambda x: np.insert(-self.K@self._sg(res(x),h)/n
                                           + alpha*self.K@x[1:]/n,
                                           0, np.mean(-self._sg(res(x),h)))
                self.solution = minimize(func, x0, method=method, 
                                         jac=grad, tol=tol, options=options)
                self.beta = self.solution.x
                self.fitY = self.beta[0] + self.K@self.beta[1:]
            
            else:
                if init is not None:
                    x0 = init
                else:
                    x0 = np.zeros(n)
                res = lambda x: self.Y - self.K@x
                func = lambda x: np.mean(self._loss(res(x),h))
                grad = lambda x: -self.K@self._sg(res(x),h)/n + alpha*self.K@x/n
                self.solution = minimize(func, x0=x0, method=method, 
                                         jac=grad, tol=tol, options=options)
                self.beta = self.solution.x
                self.fitY = self.K@self.beta
                
        else: # compute KRR-Q estimator by solving a quadratic program
              # need to install https://pypi.org/project/qpsolvers/
              # check solvers installed: 
              # import qpsolvers
              # qpsolvers.available_solvers
            C = 1/(n*alpha)
            lb = C*(tau-1)
            ub = C*tau
            x = solve_qp(P=self.K, q=-self.Y, 
                         G=None, h=None, 
                         A=np.ones(n), b=np.array([0.]), 
                         lb=lb*np.ones(n), ub=ub*np.ones(n), 
                         solver=solver)
            self.fitY = self.K@x
            b = np.quantile(self.Y - self.fitY, tau)            
            self.beta = np.insert(x, 0, b)
            self.fitY += b



    def fit_seq(self, tau=0.5, alphaseq=np.array([0.1]), intercept=True,
                kernel='RBF', sigma=1, gamma=1, r=1, degree=3,
                h=0.1, method='L-BFGS-B', solver='clarabel',
                tol=1e-6, options=None):

        alphaseq = np.sort(alphaseq)[::-1]
        args = [intercept, kernel, sigma, gamma, r, degree,
                h, method, solver, tol, options]

        x0 = None
        x = []
        fitY = []
        for alpha in alphaseq:
            self.fit(tau, alpha, x0, *args)
            x.append(self.beta)
            fitY.append(self.fitY)
            x0 = self.beta

        self.beta = np.array(x).T
        self.fitY = np.array(fitY).T
        self.alpha = alphaseq


    
    def genK(self, x):
        if self.nm == 'MinMax':
            x = (x - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            x = (x - self.xm)/self.xsd
        return np.array([self.ker_func(self.X0[i,:], x, *self.params) 
                         for i in range(self.n)])
            

    def generate(self, x): 
        '''
        Compute predicted ES at new input x
        
        Arguments
        ---------
        x : ndarray, shape (m, p) or (p,) 
        '''
        if np.ndim(x) == 1:
            self.pred = self.itcp*self.beta[0] \
                        + self.genK(x) @ self.beta[self.itcp:]
        elif np.ndim(x) == 2:
            m = x.shape[0]
            pred = []
            for j in range(m):
                pred.append(self.itcp*self.beta[0] \
                            + self.genK(x[j]) @ self.beta[self.itcp:])
            self.pred = np.array(pred)
            