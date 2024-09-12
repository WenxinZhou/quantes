import numpy as np
import numpy.random as rgt

from quantes.utils import (prox_map, soft_thresh, concave_weight)


###############################################################################
###################    proximal ADMM for Penalized QR    ######################
###############################################################################
class proximal:
    '''
        Proximal ADMM algorithm for solving 
            weighted L1-penalized quantile regression

    Refs:
        ADMM for high-dimensional sparse penalized quantile regression
        by Yuwen Gu, Jun Fan, Lingchen Kong, Shiqian Ma and Hui Zou
        Technometrics 60(3): 319--331, 2018
    '''
    params = {'gamma': 1, 'max_iter': 5e3, 'tol': 1e-6, 'nsim': 200}


    def __init__(self, X, Y, intercept=True, options={}):
        '''
        Args:
            X : n by p matrix of covariates; each row is an observation vector.
            Y : n-dimensional vector of response variables.
            intercept : logical flag for adding an intercept to the model.
            options : a dictionary of internal optimization parameters.
                gamma : constant step length for the theta-step; default is 1.
                max_iter : maximum numder of iterations; default is 5e3.
                tol : tolerance level in the ADMM convergence criterion; default is 1e-5.
                nsim : number of simulations for computing a data-driven lambda; 
                       default is 200.
        '''

        self.n = len(Y)
        self.Y = Y.reshape(self.n)
        self.itcp = intercept
        self.X = np.c_[np.ones(self.n), X] if intercept else X
        self.params.update(options)


    def _eta(self):
        return np.linalg.svd(self.X, compute_uv=0).max()**2


    def tuning(self, tau=0.5, standardize=True):
        X = self.X1 if standardize else self.X
        lambda_sim = \
            np.array([max(abs(X.T@(tau - (rgt.uniform(0,1,self.n) <= tau))))
                      for b in range(self.params['nsim'])])
        return lambda_sim/self.n
    

    def l1(self, tau=0.5, Lambda=0.1, beta=np.array([]), 
           res=np.array([]), sigma=0.01, eta=None):
        '''
            Weighted L1-Penalized Quantile Regression
        
        Args:
            tau : quantile level (between 0 and 1); default is 0.5.
            Lambda : regularization parameter. This should be either a scalar, or
                     a vector of length equal to the column dimension of X.
            beta : initial estimator of slope coefficients;
                   if unspecified, it will be set as zero.
            res : residual vector of the initial estiamtor.
            sigma : augmentation parameter; default is 0.01.
            eta :  a positive parameter;
                   if unspecifed, it will be set as the largest eigenvalue of X'X.
        
        Returns:
            'beta' : an ndarray of estimated coefficients.
            'res' : an ndarray of fitted residuals.
            'lambda' : regularization parameter.
        '''

        n, dim = self.n, self.X.shape[1]
        if not beta.any(): 
            beta, res = np.zeros(dim), self.Y
        z, theta = res, np.ones(n)/n
        if eta is None: eta = self._eta()

        if self.itcp:
            Lambda = np.insert(Lambda * np.ones(dim-1), 0, 0)

        k, dev = 0, 1
        while dev > self.params['tol'] and k < self.params['max_iter']:
            beta_new = soft_thresh(beta+self.X.T@(theta/sigma + res - z)/eta,
                                   Lambda / sigma / eta)
            res = self.Y - self.X@(beta_new)
            z = prox_map(res + theta/sigma, tau, n * sigma)
            theta = theta - self.params['gamma'] * sigma * (z - res)
            dev = max(abs(beta_new - beta))
            beta = beta_new
            k += 1

        return {'beta': beta, 
                'res': res,
                'niter': k, 
                'theta': theta,
                'lambda': Lambda}


    def l1_path(self, tau=0.5, lambda_seq=np.array([]), nlambda=50,
                order="descend", sigma=0.1, eta=None):
        '''
            Solution Path of L1-Penalized Quantile Regression

        Args:
            tau : quantile level (between 0 and 1); default is 0.5.
            lambda_seq : an ndarray of lambda values (regularization parameters).
            nlambda : number of lambda values.
            order : a character string indicating the order of lambda values along
                    which the solution path is obtained; default is 'descend'.
            sigma : augmentation parameter; default is 0.01.
            eta :  a positive parameter;
                   if unspecifed, it will be set as the largest eigenvalue of X'X.

        Returns:
            'beta_seq' : a sequence of l1 estimates.
            'res_seq' : a sequence of residual vectors.
            'size_seq' : a sequence of numbers of selected variables.
            'lambda_seq' : a sequence of lambda values in ascending/descending order.
        '''
 
        if len(lambda_seq) == 0:
            lam_max = max(self.tuning(tau, standardize=False))
            lambda_seq = np.linspace(0.25*lam_max, lam_max, num=nlambda)
        
        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]
        nlambda = len(lambda_seq)
        
        if eta is None: eta = self._eta()

        beta_seq = np.zeros(shape=(self.X.shape[1], nlambda))
        res_seq = np.zeros(shape=(self.n, nlambda))
        niter_seq = np.ones(nlambda)
        model = self.l1(tau, lambda_seq[0], sigma=sigma, eta=eta)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        niter_seq[0] = model['niter']
        
        for l in range(1, nlambda):
            model = self.l1(tau, lambda_seq[l], beta_seq[:,l-1], 
                            res_seq[:,l-1], sigma=sigma, eta=eta)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
            niter_seq[l] = model['niter']

        return {'beta_seq': beta_seq, 
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[self.itcp:,:] != 0, axis=0),
                'lambda_seq': lambda_seq,
                'niter_seq': niter_seq}


    def irw(self, tau=0.5, Lambda=0.1, beta=np.array([]), res=np.array([]), 
            sigma=0.01, eta=None, penalty="SCAD", a=3.7, nstep=3):
        '''
            Iteratively Reweighted L1-Penalized Quantile Regression

        Arguments
        ---------
        tau : quantile level (between 0 and 1); default is 0.5.

        Lambda : regularization parameter. This should be either a scalar, or
                 a vector of length equal to the column dimension of X.

        beta : initial estimate of slope coefficients. 
               If unspecified, it will be set as zero.

        res : residual vector of the initial estiamtor.

        sigma : augmentation parameter; default is 0.01.

        eta :  a positive parameter; 
               if unspecifed, it will be set as the largest eigenvalue of X'X.

        penalty : a character string representing one of the built-in concave penalties; 
                  default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 3.

        Returns
        -------
        'beta' : an ndarray of estimated coefficients.
        
        'res' : an ndarray of fitted residuals.

        'lambda' : regularization parameter.
        '''
        if not beta.any():
            model = self.l1(tau, Lambda, sigma=sigma, eta=eta)
        else:
            res = self.Y - self.X.dot(beta)
            model = self.l1(tau, Lambda, beta, res, sigma, eta)
        beta, res = model['beta'], model['res']
        lam = np.ones(self.X.shape[1] - self.itcp) * Lambda
        pos = lam > 0
        rw_lam = np.zeros(self.X.shape[1] - self.itcp)

        if eta is None: eta = self._eta()

        err, t = 1, 1
        while err > self.params['tol'] and t <= nstep:
            rw_lam[pos] = lam[pos] \
                          * concave_weight(beta[self.itcp:][pos]/lam[pos],
                                           penalty, a)
            model = self.l1(tau, rw_lam, beta, res, sigma=sigma, eta=eta)
            err = max(abs(model['beta']-beta))
            beta, res = model['beta'], model['res']
            t += 1
            
        return {'beta': beta, 
                'res': res, 
                'nstep': t, 
                'lambda': lam}


    def irw_path(self, tau=0.5, lambda_seq=np.array([]), nlambda=50, 
                 order="descend", sigma=0.1, eta=None, 
                 penalty="SCAD", a=3.7, nstep=3):
        '''
            Solution Path of IRW-L1-Penalized Quantile Regression

        Returns:
            'beta_seq' : a sequence of irw estimates.
            'res_seq' : a sequence of residual vectors.
            'size_seq' : a sequence of numbers of selected variables.
            'lambda_seq' : a sequence of lambda values in ascending/descending order.
        '''

        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]
        nlambda = len(lambda_seq)

        if eta is None: eta = self._eta()

        beta_seq = np.zeros(shape=(self.X.shape[1], nlambda))
        res_seq = np.zeros(shape=(self.n, nlambda))
        model = self.irw(tau, lambda_seq[0], sigma=sigma, eta=eta, 
                         penalty=penalty, a=a, nstep=nstep)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']

        for l in range(1, nlambda):
            model = self.irw(tau, lambda_seq[l], beta_seq[:,l-1], 
                             res_seq[:,l-1], sigma, eta, penalty, a, nstep)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']

        return {'beta_seq': beta_seq,
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq[self.itcp:,:] != 0, axis=0),
                'lambda_seq': lambda_seq}



class ncvx:
    '''
        Nonconvex Penalized Quantile Regression via ADMM

    Refs:
        Convergence for nonconvex ADMM, with applications to CT imaging
        by Rina Foygel Barber and Emil Y. Sidky
        Journal of Machine Learning Research 25(38):1−46, 2024.
    '''
    def __init__(self, X, Y, intercept=True):
        '''
        Args:
            X : n by p matrix of covariates; each row is an observation vector.
            Y : n-dimensional vector of response variables.
            intercept : logical flag for adding an intercept to the model.
        '''
        
        self.n = len(Y)
        self.Y = Y.reshape(self.n)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
        else:
            self.X = X


    def loss(self, beta, tau=0.5, Lambda=0.1, c=1):
        return (np.maximum(self.Y - self.X.dot(beta), 0).sum()*tau \
                + np.maximum(self.X.dot(beta)-self.Y, 0).sum()*(1-tau)) / self.n \
                + Lambda * c * np.log(1 + np.abs(beta)/c).sum()


    def fit(self, tau=0.5, Lambda=0.1, c=1, 
            sig=0.0002, niter=5e3, tol=1e-5):
        '''
        Args:
            tau : quantile level; default is 0.5.
            Lambda : regularization parameter (float); default is 0.1.
            c : constant parameter in the penalty P_c(x) = c * log(1 + |x|/c); default = 1. 
                The penalty P_c(x) converges to |x| as c tends to infinity.
            sig : constant step length for the theta-step; default is 0.0002.
            niter : maximum numder of iterations; default is 5e3.
            tol : tolerance level in the ADMM convergence criterion; default is 1e-5.

        Returns:
            'beta' : penalized quantile regression estimate.
            'loss_val' : values of the penalized loss function at all iterates.
            'Lambda' : regularization parameter.
        '''

        gam = np.linalg.svd(self.X, compute_uv=0).max()**2
        loss_xt = np.zeros(np.int64(niter))
        loss_xtbar = np.zeros(np.int64(niter))
        beta_avg = np.zeros(self.X.shape[1])
        beta = np.zeros(self.X.shape[1])
        y = np.zeros(self.n)
        u = np.zeros(self.n)
        
        i, loss_diff = 0, 1e3
        while i < niter and loss_diff > tol:  
            beta = beta - self.X.T@(self.X@beta)/gam \
                   + self.X.T@y /gam \
                   - self.X.T@u/sig/gam \
                   + Lambda * beta/(c+np.abs(beta))/sig/gam
            beta = np.sign(beta) * np.maximum(np.abs(beta)-Lambda/sig/gam, 0)
            y = self.X@beta + u/sig
            y = (y + tau/self.n/sig) * (y + tau/self.n/sig < self.Y) \
                + (y-(1-tau)/self.n/sig) * (y-(1-tau)/self.n/sig > self.Y) \
                + self.Y * (y + tau/self.n/sig >= self.Y) \
                    * (y - (1-tau)/self.n/sig <= self.Y)       
            u = u + sig * (self.X@beta - self.Y)
            beta_avg = beta_avg * (i/(i+1)) + beta * (1/(i+1))
            loss_xt[i] = self.loss(beta, tau, Lambda, c)
            loss_xtbar[i] = self.loss(beta_avg, tau, Lambda, c)
            if i >= 5:
                loss_diff = abs(loss_xt[i] - np.mean(loss_xt[i-5 : i]))
            i += 1

        return {'beta': beta, 
                'beta_avg': beta_avg, 
                'loss_val': loss_xt, 
                'Lambda': Lambda, 
                'niter': i}