import numpy as np
import numpy.random as rgt


from quantes.utils import (conquer_weight, smooth_composite_check,
                           soft_thresh, concave_weight)


class high_dim:
    '''
        Penalized Composite Quantile Regression

    Refs:
        Sparse composite quantile regression in utrahigh dimensions 
        with tuning parameter calibration 
        by Yuwen Gu and Hui Zou
        IEEE Transactions on Information Theory 66(11): 7132-7154, 2020

        High-dimensional composite quantile regression: 
        optimal statistical guarantees and fast algorithms 
        by Haeseong Moon and Wen-Xin Zhou
        Electronic Journal of Statistics 17(2): 2067-2119, 2023
    '''
    params = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-6,
              'irw_tol': 1e-5, 'nsim': 200, 'min_bandwidth': 1e-4}

    def __init__(self, X, Y, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.X1 = (X - self.mX)/self.sdX
        self.params.update(options)


    def bandwidth(self, tau):
        n, p = self.X.shape
        h0 = min((p + np.log(n))/n, 0.5)**0.4
        return max(self.params['min_bandwidth'], 
                   h0 * (tau-tau**2)**0.5)
    

    def composite_weight(self, x, alpha=np.array([]), tau=np.array([]), 
                         h=None, kernel="Laplacian", w=np.array([])):
        out = conquer_weight((alpha[0] - x) / h, tau[0], kernel, w)
        for i in range(1, len(tau)):
            out = np.hstack((out, conquer_weight((alpha[i]-x)/h, 
                                                 tau[i], kernel, w)))
        return out / (len(tau) * self.n)


    def uniform_weights(self, tau=np.array([])):
        w = (rgt.uniform(0, 1, self.n) <= tau[0]) - tau[0]
        for i in range(1, len(tau)):
            w = np.hstack((w, (rgt.uniform(0, 1, self.n) <= tau[i]) - tau[i]))
        return w


    def lambda_tuning(self, XX, tau=np.array([])):
        lambda_sim = np.array([max(abs(XX.dot(self.uniform_weights(tau)))) \
                               for _ in range(self.params['nsim'])])
        return lambda_sim / (len(tau) * self.n)


    def l1(self, tau=np.array([]), K=9, Lambda=np.array([]),
           h=None, kernel="Laplacian", 
           alpha0=np.array([]), beta0=np.array([]), res=np.array([]), 
           standardize=True, adjust=True, weight=np.array([]), c=1.5):
        '''
           L1-Penalized Composite Quantile Regression via Convolution Smoothing
           (l1-composite-conquer)

        Args:
            tau : an ndarray of quantile levels (between 0 and 1);
                  default is {0.1, 0.2, ..., 0.9}.
            K : number of tau values; default = 9.
            Lambda : regularization parameter. This should be either a scalar, or
                     a vector of length equal to the column dimension of X. If unspecified,
                     it will be computed by lambda_tuning().
            h : bandwidth/smoothing parameter. The default is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels; 
                     default is "Laplacian".
            beta0 : initial estimator of slope coefficients; 
                    if unspecified, it will be set to zero.
            alpha0 : initial estimator of intercept terms in CQR regression (alpha terms);
                     if unspecified, it will be set as zero.
            res : residual vector of the initial estiamtor.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
            weight : an ndarray of observation weights; default is np.array([]) (empty).
            c : a constant for the regularization parameter; default is 1.5.

        Returns:
            'alpha': an ndarray of estimated coefficients for intercept terms.
            'beta' : an ndarray of estimated coefficients for slope coefficients.
            'res' : an ndarray of fitted residuals.
            'niter' : number of iterations.
            'lambda' : regularization parameter(s).
            'bw' : bandwidth.
        '''

        if not np.array(tau).any(): 
            tau = np.linspace(1/(K+1), K/(K+1), K)
        K = len(tau)
        X = self.X1 if standardize else self.X
        XX = np.tile(X.T, K)

        Lambda = np.array(Lambda).reshape(-1)
        if not Lambda.any():
            Lambda = c * np.quantile(self.lambda_tuning(XX, tau), 0.95)
        
        h = self.bandwidth(np.mean(tau)) if h is None else h
        if len(beta0)==0: beta0 = np.zeros(self.p)
        if len(alpha0)==0: alpha0 = np.zeros(K)

        res = self.Y - X @ beta0

        alphaX = np.zeros((K, self.n * K))
        for i in range(0, K):
            for j in range(i * self.n, (i + 1) * self.n):
                alphaX[i, j] = 1

        phi, dev, count = self.params['phi'], 1, 0
        while dev > self.params['tol'] and count < self.params['max_iter']:

            gradalpha0 = alphaX@(self.composite_weight(res, alpha0, tau,
                                                       h, kernel, w=weight))
            gradbeta0 = XX@(self.composite_weight(res, alpha0, tau,
                                                  h, kernel, w=weight))
            loss_eval0 = smooth_composite_check(res, alpha0, tau, 
                                                h, kernel, weight)
            alpha1 = alpha0 - gradalpha0 / phi
            beta1 = beta0 - gradbeta0 / phi
            beta1 = soft_thresh(beta1, Lambda / phi)
            diff_alpha = alpha1 - alpha0
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta) + diff_alpha.dot(diff_alpha)
            res = self.Y - X @ beta1
            loss_proxy = loss_eval0 + diff_beta.dot(gradbeta0) \
                         + diff_alpha.dot(gradalpha0) + 0.5 * phi * r0
            loss_eval1 = smooth_composite_check(res, alpha1, tau, 
                                                h, kernel, weight)

            while loss_proxy < loss_eval1:
                phi *= self.params['gamma']
                alpha1 = alpha0 - gradalpha0 / phi
                beta1 = beta0 - gradbeta0 / phi
                beta1 = soft_thresh(beta1, Lambda / phi)
                diff_alpha = alpha1 - alpha0
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta) + diff_alpha.dot(diff_alpha)
                res = self.Y - X @ beta1
                loss_proxy = loss_eval0 + diff_beta.dot(gradbeta0) \
                             + diff_alpha.dot(gradalpha0) + 0.5 * phi * r0
                loss_eval1 = smooth_composite_check(res, alpha1, tau, 
                                                    h, kernel, weight)

            dev = max(abs(beta1 - beta0)) + max(abs(alpha1 - alpha0))
            alpha0, beta0 = np.copy(alpha1), np.copy(beta1)
            phi = self.params['phi']
            count += 1

        if standardize and adjust:
            beta1 /= self.sdX

        return {'alpha': alpha1, 
                'beta': beta1, 
                'res': res,
                'niter': count, 
                'lambda': Lambda, 
                'bw': h}


    def irw(self, tau=np.array([]), K=9, Lambda=None, 
            h=None, kernel="Laplacian", 
            alpha0=np.array([]), beta0=np.array([]), res=np.array([]),
            penalty="SCAD", a=3.7, nstep=3, standardize=True, adjust=True, 
            weight=np.array([]), c=2):
        '''
            Iteratively Reweighted L1-Penalized Composite Conquer

        Args:
            tau : an ndarray of quantile levels (between 0 and 1);
                  default is {0.1, 0.2, ..., 0.9}.
            K : number of tau values; default = 9.
            Lambda : regularization parameter. This should be either a scalar, or
                     a vector of length equal to the column dimension of X. If unspecified,
                     it will be computed by lambda_tuning().
            h : bandwidth/smoothing parameter. The default is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels; 
                     default is "Laplacian".
            beta0 : initial estimator of slope coefficients;
                    if unspecified, it will be set to zero.
            alpha0 : initial estimator of intercept terms in CQR regression (alpha terms);
                     if unspecified, it will be set as zero.
            res : residual vector of the initial estiamtor.
            penalty : a character string representing one of the built-in concave penalties; 
                      default is "SCAD".
            a : the constant (>2) in the concave penality; default is 3.7.
            nstep : number of iterations/steps of the IRW algorithm; default is 3.
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
            weight : an ndarray of observation weights; default is np.array([]) (empty).

        Returns:
            'alpha': an ndarray of estimated coefficients for intercept terms.
            'beta' : an ndarray of estimated coefficients for slope coefficients.
            'res' : an ndarray of fitted residuals.
            'niter' : number of iterations.
            'lambda' : regularization parameter(s).
            'bw' : bandwidth.
        '''

        if not np.array(tau).any(): 
            tau = np.linspace(1/(K+1), K/(K+1), K)
        X = self.X1 if standardize else self.X

        if Lambda is None:
            Lambda = c * np.quantile(self.lambda_tuning(np.tile(X.T, K), tau), 
                                     0.95)
        h = self.bandwidth(np.mean(tau)) if h is None else h

        if len(beta0) == 0:
            model = self.l1(tau, K, Lambda, h, kernel,
                            alpha0=np.zeros(K), beta0=np.zeros(self.p),
                            standardize=standardize, adjust=False, 
                            weight=weight)
        else:
            model = self.l1(tau, K, Lambda, h, kernel=kernel,
                            alpha0=alpha0, beta0=beta0, res=res,
                            standardize=standardize, adjust=False, 
                            weight=weight)
        alpha0, beta0, res = model['alpha'], model['beta'], model['res']
        nit = []
        nit.append(model['niter'])

        if penalty == 'L1': nstep == 0
        dev, step = 1, 1
        while dev > self.params['irw_tol'] and step <= nstep:
            rw_lambda = Lambda * concave_weight(beta0 / Lambda, penalty, a)
            model = self.l1(tau, K, rw_lambda, 
                            h, kernel, alpha0, beta0, res,
                            standardize, adjust=False, weight=weight)
            dev = max(abs(model['beta'] - beta0)) \
                  + max(abs(model['alpha'] - alpha0))
            alpha0, beta0, res = model['alpha'], model['beta'], model['res']
            step += 1
            nit.append(model['niter'])

        if standardize and adjust:
            beta0 /= self.sdX
        nit_seq = np.array(nit)

        return {'alpha': alpha0, 
                'beta': beta0, 
                'res': res,
                'h': h, 
                'lambda': Lambda,
                'nstep': step, 
                'niter': np.sum(nit_seq),
                'nit_seq': nit_seq}


    def l1_path(self, tau=np.array([]), K=9, 
                lambda_seq=np.array([]), nlambda=40, order="descend", 
                h=None, kernel="Laplacian", 
                standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized Composite Conquer

        Args:
            tau : an ndarray of quantile levels (between 0 and 1);
                  default is {0.1, 0.2, ..., 0.9}.
            K : number of tau values; default = 9.
            lambda_seq : an ndarray of lambda values (regularization parameters).
            nlambda : number of lambda values.
            order : a character string indicating the order of lambda values along
                    which the solution path is obtained; default is 'descend'.
            h : bandwidth/smoothing parameter. The default is computed by self.bandwidth().
            kernel : a character string representing one of the built-in smoothing kernels; 
                     default is "Laplacian".
            standardize : logical flag for x variable standardization prior to fitting the model;
                          default is TRUE.
            adjust : logical flag for returning coefficients on the original scale.
        
        Returns:
            'alpha_seq' : a sequence of l1-composite-conquer estimates for intercept terms.
            'beta_seq' : a sequence of l1-composite-conquer estimates for slope coefficients.
            'res_seq' : a sequence of residual vectors.
            'size_seq' : a sequence of numbers of selected variables.
            'lambda_seq' : a sequence of lambda values in ascending/descending order.
            'bw' : bandwidth.
        '''

        if not np.array(tau).any(): 
            tau = np.linspace(1/(K+1), K/(K+1), K)
        K = len(tau)

        lambda_seq = np.array(lambda_seq).reshape(-1)
        if not lambda_seq.any():
            X = self.X1 if standardize else self.X
            lambda_sim = self.lambda_tuning(np.tile(X.T, len(tau)), tau)
            lambda_seq = np.linspace(np.min(lambda_sim), 2*max(lambda_sim),
                                     num=nlambda)
        
        h = self.bandwidth(np.mean(tau)) if h is None else h
        
        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        nit_seq = []
        alpha_seq = np.zeros(shape=(K, len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.p, len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))
        model = self.l1(tau, K, lambda_seq[0], h, kernel,
                        standardize=standardize, adjust=False)
        alpha_seq[:,0], beta_seq[:,0], res_seq[:,0] \
            = model['alpha'], model['beta'], model['res']
        nit_seq.append(model['niter'])

        for l in range(1, len(lambda_seq)):
            model = self.l1(tau, K, lambda_seq[l], h, kernel,
                            alpha_seq[:, l-1], beta_seq[:, l-1],
                            res_seq[:, l - 1], standardize, adjust=False)
            beta_seq[:, l], res_seq[:, l] = model['beta'], model['res']
            nit_seq.append(model['niter'])

        if standardize and adjust:
            beta_seq[:, ] /= self.sdX[:, None]

        return {'alpha_seq': alpha_seq, 
                'beta_seq': beta_seq, 
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq != 0, axis=0),
                'lambda_seq': lambda_seq,
                'nit_seq': np.array(nit_seq), 
                'bw': h}


    def irw_path(self, tau=np.array([]), K=9,
                 lambda_seq=np.array([]), nlambda=40, order="descend",
                 h=None, kernel="Laplacian", 
                 penalty="SCAD", a=3.7, nstep=3,
                 standardize=True, adjust=True):
        '''
            Solution Path of Iteratively Reweighted L1-Penalized Composite Conquer
        '''
        if not np.array(tau).any(): 
            tau = np.linspace(1/(K+1), K/(K+1), K)
        K = len(tau)
        
        lambda_seq = np.array(lambda_seq).reshape(-1)
        if not lambda_seq.any():
            if standardize: X = self.X1
            else: X = self.X
            lambda_sim = self.lambda_tuning(np.tile(X.T, len(tau)), tau)
            lambda_seq = np.linspace(0.75*max(lambda_sim), 2*max(lambda_sim),
                                     num=nlambda)
        
        h = self.bandwidth(np.mean(tau)) if h is None else h

        if order == 'ascend':
            lambda_seq = np.sort(lambda_seq)
        elif order == 'descend':
            lambda_seq = np.sort(lambda_seq)[::-1]

        if penalty == 'L1': nstep=0
        nit_seq = []
        alpha_seq = np.zeros(shape=(K, len(lambda_seq)))
        beta_seq = np.zeros(shape=(self.p, len(lambda_seq)))
        res_seq = np.zeros(shape=(self.n, len(lambda_seq)))
        model = self.irw(tau, K, lambda_seq[0], h, kernel,
                         penalty=penalty, a=a, nstep=nstep,
                         standardize=standardize, adjust=False)
        alpha_seq[:,0], beta_seq[:,0], res_seq[:,0] \
            = model['alpha'], model['beta'], model['res']
        nit_seq.append(model['niter'])

        for l in range(1, len(lambda_seq)):
            model = self.irw(tau, K, lambda_seq[l], h, kernel,
                             alpha_seq[:, l-1], 
                             beta_seq[:, l-1], 
                             res_seq[:, l - 1],
                             penalty, a, nstep,
                             standardize, adjust=False)
            beta_seq[:, l], res_seq[:, l] = model['beta'], model['res']
            nit_seq.append(model['niter'])

        if standardize and adjust:
            beta_seq[:, ] /= self.sdX[:, None]

        return {'alpha_seq': alpha_seq, 
                'beta_seq': beta_seq,
                'res_seq': res_seq,
                'size_seq': np.sum(beta_seq != 0, axis=0),
                'lambda_seq': lambda_seq,
                'nit_seq': np.array(nit_seq),
                'bw': h}