import numpy as np
from scipy.special import logsumexp

from quantes.utils import soft_thresh


class bbgd:
    '''
        Barzilai-Borwein gradient descent
    '''
    def __init__(self, params={'max_iter': 1e3, 'tol': 1e-8, 
                               'init_lr': 1, 'max_lr': 50}):
        self.init_lr = params['init_lr']
        self.max_lr = params['max_lr']
        self.max_iter = params['max_iter']
        self.lr_seq = [params['init_lr']]
        self.fun_seq = []
        self.niter = 0
        self.tol = params['tol']

    def minimize(self, func, grad, x0):
        grad0 = grad(x0)
        diff_x = -self.init_lr * grad0
        x1 = x0 + diff_x
        self.fun_seq.append(func(x1))

        while self.niter < self.max_iter and max(abs(diff_x)) > self.tol:
            grad1 = grad(x1)
            diff_grad = grad1 - grad0
            r0, r1 = diff_x.dot(diff_x), diff_grad.dot(diff_grad)
            if r1 == 0: lr = 1
            else:
                r01 = diff_grad.dot(diff_x)
                lr = min(logsumexp(abs(r01/r1)), logsumexp(abs(r0/r01)))

            if self.max_lr: lr = min(lr, self.max_lr)
            self.lr_seq.append(lr)
            grad0, diff_x = grad1, -lr*grad1
            x1 += diff_x
            self.fun_seq.append(func(x1))
            self.niter += 1

        if self.niter == self.max_iter:
            self.message = "Maximum number of iterations achieved in bbgd()"
        else:
            self.message = "Convergence achieved in bbgd()"

        self.lr_seq = np.array(self.lr_seq)
        self.fun_seq = np.array(self.fun_seq)
        self.x = x1


class lamm:
    '''
        Local adaptive majorization-minimization
    '''
    def __init__(self, params={'phi': 0.1, 'gamma': 1.25, 
                               'max_iter': 1e3, 'tol': 1e-8}):
        self.phi = params['phi']
        self.gamma = params['gamma']
        self.max_iter = params['max_iter']
        self.tol = params['tol']
        self.niter = 0

    def minimize(self, func, grad, x0, lambda_vec):
        phi, r2 = self.phi, 1
        while r2 > self.tol and self.niter < self.max_iter:
            grad0 = grad(x0)
            loss_eval0 = func(x0)
            x1 = x0 - grad0/phi
            x1 = soft_thresh(x1, lambda_vec/phi)
            diff_x = x1 - x0
            r2 = diff_x.dot(diff_x)
            loss_proxy = loss_eval0 + diff_x.dot(grad0) + 0.5*phi*r2
            loss_eval1 = func(x1)

            while loss_proxy < loss_eval1:
                phi *= self.gamma
                x1 = x0 - grad0/phi
                x1 = soft_thresh(x1, lambda_vec/phi)
                diff_x = x1 - x0
                r2 = diff_x.dot(diff_x)
                loss_proxy = loss_eval0 + diff_x.dot(grad0) + 0.5*phi*r2
                loss_eval1 = func(x1)

            x0, phi = x1, self.phi
            self.niter += 1

        if self.niter == self.max_iter:
            self.message = "Maximum number of iterations achieved in lamm()"
        else:
            self.message = "Convergence achieved in lamm()"
        self.x = x1