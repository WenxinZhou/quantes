from .config import np
from .utils import soft_thresh


class bbgd:
    '''
        Barzilai-Borwein gradient descent
    '''
    def __init__(self, params={'max_iter': 500, 'tol': 1e-8, 
                               'init_lr': 1, 'max_lr': 25}):
        self.init_lr = params['init_lr']
        self.max_lr = params['max_lr']
        self.max_iter = params['max_iter']
        self.lr_seq = [self.init_lr]
        self.fn_seq = []
        self.niter = 0
        self.tol = params['tol']

    def minimize(self, f, grad_f, x0):
        x, grad_old = x0, grad_f(x0)
        x_update = -self.init_lr * grad_old
        x += x_update
        self.niter += 1
        self.fn_seq.append(f(x))
        
        while self.niter < self.max_iter:
            grad = grad_f(x)
            if np.linalg.norm(grad) < self.tol:
                break
            grad_update = grad - grad_old
            r01 = x_update.dot(grad_update)
            lr = min(x_update.dot(x_update)/r01, 
                     r01/grad_update.dot(grad_update), 
                     self.max_lr)
            x_update, grad_old = -lr * grad, grad
            x += x_update
            self.niter += 1
            self.lr_seq.append(lr)
            self.fn_seq.append(f(x))
            if self.niter % 10 == 0 \
                and self.fn_seq[-1] > np.mean(self.fn_seq[-10:]):
                self.max_lr /= 2
      
        if self.niter == self.max_iter:
            self.message = "Maximum number of iterations achieved in bbgd()"
        else:
            self.message = "Convergence achieved in bbgd()"
        self.x = x


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

    def minimize(self, f, grad_f, x0, lambda_vec):
        phi, r2 = self.phi, 1
        while r2 > self.tol and self.niter < self.max_iter:
            grad0 = grad_f(x0)
            loss_eval0 = f(x0)
            x = x0 - grad0/phi
            x = soft_thresh(x, lambda_vec/phi)
            diff_x = x - x0
            r2 = diff_x.dot(diff_x)
            loss_proxy = loss_eval0 + diff_x.dot(grad0) + 0.5*phi*r2
            loss_eval1 = f(x)

            while loss_proxy < loss_eval1:
                phi *= self.gamma
                x = x0 - grad0/phi
                x = soft_thresh(x, lambda_vec/phi)
                diff_x = x - x0
                r2 = diff_x.dot(diff_x)
                loss_proxy = loss_eval0 + diff_x.dot(grad0) + 0.5*phi*r2
                loss_eval1 = f(x)

            x0, phi = x, self.phi
            self.niter += 1

        if self.niter == self.max_iter:
            self.message = "Maximum number of iterations achieved in lamm()"
        else:
            self.message = "Convergence achieved in lamm()"
        self.x = x