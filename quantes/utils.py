import numpy as np
import numpy.random as rgt
from scipy.stats import norm


###############################################################################
################## Utility Functions for Quantile Regression ##################
###############################################################################

def mad(x):
    ''' 
        Median absolute deviation 
    '''
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def boot_weight(weight):
    '''
        Bootstrap (random) weights generator 
    '''
    boot = {'Multinomial': lambda n :
            rgt.multinomial(n, pvals=np.ones(n)/n), 
            'Exponential': lambda n : rgt.exponential(size=n), 
            'Rademacher': lambda n : 2*rgt.binomial(1, 1/2, n), 
            'Gaussian': lambda n : rgt.normal(1, 1, n), 
            'Uniform': lambda n : rgt.uniform(0, 2, n), 
            'Folded-normal': lambda n :
            abs(rgt.normal(size=n)) * np.sqrt(.5 * np.pi)
            }
    return boot[weight]


def AHuber_fn(x, tau=.5, c=0):
    ''' 
        Asymmetric Huber loss 
    '''
    pos = x > 0
    if c == 0:
        fn = 0.5 * x**2
    else:
        fn = np.where(abs(x) <= c, 0.5 * x**2,
                      c * abs(x) - 0.5 * c**2)
    fn[pos] *= tau
    fn[~pos] *= 1 - tau
    return np.mean(fn)


def AHuber_grad(x, tau=.5, c=0):
    ''' 
        Gradient of asymmetric Huber loss 
    '''
    gd = np.where(x>0, tau*x, (1-tau)*x)
    if c>0:
        gd[x > c] = tau * c
        gd[x < -c] = (tau-1) * c
    return gd


def smooth_check(x, tau=0.5, h=0.5, kernel='Laplacian', w=np.array([])):
    ''' 
        Smoothed (weighted) check loss 
    '''
    if kernel == 'Laplacian':
        loss = lambda x: np.where(x >= 0, tau*x, (tau-1)*x) \
                         + (h/2) * np.exp(-abs(x)/h)
    elif kernel == 'Gaussian':
        loss = lambda x: (tau - norm.cdf(-x/h)) * x \
                              + (h/2) * np.sqrt(2 / np.pi) \
                                * np.exp(-(x/h) ** 2 / 2)
    elif kernel == 'Logistic':
        loss = lambda x: tau * x + h * np.log(1 + np.exp(-x/h))
    elif kernel == 'Uniform':
        loss = lambda x: (tau - .5) * x \
                         + h * (.25 * (x/h)**2 + .25) * (abs(x) < h) \
                         + .5 * abs(x) * (abs(x) >= h)
    elif kernel == 'Epanechnikov':  
        loss = lambda x: (tau - .5) * x + .5 * h * (.75 * (x/h) ** 2 \
                         - .125 * (x/h) ** 4 + .375) * (abs(x) < h) \
                         + .5 * abs(x) * (abs(x) >= h)
    if not w.any(): 
        return np.mean(loss(x))
    else:
        return np.mean(loss(x) * w)


def conquer_weight(x, tau, kernel="Laplacian", w=np.array([])):
    ''' 
        Gradient weights for the (weighted) smoothed check loss 
    '''
    if kernel=='Laplacian':
        Ker = lambda x : 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
    elif kernel=='Gaussian':
        Ker = lambda x : norm.cdf(x)
    elif kernel=='Logistic':
        Ker = lambda x : 1 / (1 + np.exp(-x))
    elif kernel=='Uniform':
        Ker = lambda x : np.where(x > 1, 1, 0) \
                         + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
    elif kernel=='Epanechnikov':
        Ker = lambda x : 0.25 * (2 + 3 * x / 5 ** 0.5 \
                         - (x / 5 ** 0.5)**3 ) * (abs(x) <= 5 ** 0.5) \
                         + (x > 5 ** 0.5)                      
    if not w.any():
        return (Ker(x) - tau) 
    else:
        return w * (Ker(x) - tau)


def find_root(f, tmin, tmax, tol=1e-5):
    while tmax - tmin > tol:
        tau = (tmin + tmax) / 2
        if f(tau) > 0:
            tmin = tau
        else: 
            tmax = tau
    return tau


def concave_weight(x, penalty="SCAD", a=None):
    if penalty == "SCAD":
        if a is None:
            a = 3.7
        tmp = 1 - (abs(x) - 1) / (a - 1)
        tmp = np.where(tmp <= 0, 0, tmp)
        return np.where(tmp > 1, 1, tmp)
    elif penalty == "MCP":
        if a is None:
            a = 3
        tmp = 1 - abs(x) / a 
        return np.where(tmp <= 0, 0, tmp)
    elif penalty == "CappedL1":
        if a is None:
            a = 3
        return abs(x) <= a / 2
    

def soft_thresh(x, c):
    tmp = abs(x) - c
    return np.sign(x)*np.where(tmp<=0, 0, tmp)


def sparse_proj(x, s):
    return np.where(abs(x) < np.sort(abs(x))[-s], 0, x)


def sparse_supp(x, s):
    y = abs(x)
    return y >= np.sort(y)[-s]


def prox_map(x, tau, alpha):
    '''
        Proximal map for the check loss
    '''
    return x - np.maximum((tau - 1)/alpha, np.minimum(x, tau/alpha))


def smooth_composite_check(x, alpha=np.array([]), tau=np.array([]), 
                           h=None, kernel='Laplacian', w=np.array([])):
    out = np.array([smooth_check(x - alpha[i], tau[i], h, kernel, w)
                    for i in range(len(tau))])
    return np.mean(out)


def composite_check_sum(x, tau, alpha):
    out = 0
    for i in range(0, len(tau)):
        out += np.sum(np.where(x - alpha[i] >= 0, 
                               tau[i] * (x - alpha[i]),
                               (tau[i] - 1) * (x - alpha[i])))
    return out / len(tau)


def weighted_quantile(data, quantiles, weights):
    '''
        Compute weighted quantile(s) of a dataset (numpy array)

    Args:
        data: 1D numpy array, data
        quantiles: 1D numpy array, quantile(s) to compute
        weights: 1D numpy array, weights
    '''
    # sort data and weights
    sorted_data = np.sort(data)
    sorted_weights = np.sort(weights)
    
    # compute the cumulative sum of the weights
    cum_weights = np.cumsum(sorted_weights)
    
    # normalize the cumulative weights
    cum_weights /= cum_weights[-1]
    
    # find the quantile indices
    qt_indices = np.searchsorted(cum_weights, quantiles)
    
    # interpolate to find the quantile values
    qt_values = []
    for qt_index, qt in zip(qt_indices, quantiles):
        if qt_index == 0:
            qt_values.append(sorted_data[0])
        elif qt_index == len(sorted_data):
            qt_values.append(sorted_data[-1])
        else:
            lower = sorted_data[qt_index - 1]
            upper = sorted_data[qt_index]
            interpolation = (qt - cum_weights[qt_index - 1]) \
                / (cum_weights[qt_index] - cum_weights[qt_index - 1])
            qt_value = lower + interpolation * (upper - lower)
            qt_values.append(qt_value)
    
    return qt_values



###############################################################################
################ Utility Functions for Quantile/ES Regression #################
###############################################################################
import torch
from torch.distributions.normal import Normal

def G2(G2_type=1):
    '''
        Specification Function G2 in Fissler and Ziegel's Joint Loss 
    '''
    if G2_type == 1:
        f0 = lambda x : -np.sqrt(-x)
        f1 = lambda x : 0.5 / np.sqrt(-x)
        f2 = lambda x : 0.25 / np.sqrt((-x)**3)
    elif G2_type == 2:
        f0 = lambda x : -np.log(-x)
        f1 = lambda x : -1 / x
        f2 = lambda x : 1 / x ** 2
    elif G2_type == 3:
        f0 = lambda x : -1 / x
        f1 = lambda x : 1 / x ** 2
        f2 = lambda x : -2 / x ** 3
    elif G2_type == 4:
        f0 = lambda x : np.log( 1 + np.exp(x))
        f1 = lambda x : np.exp(x) / (1 + np.exp(x))
        f2 = lambda x : np.exp(x) / (1 + np.exp(x)) ** 2
    elif G2_type == 5:
        f0 = lambda x : np.exp(x)
        f1 = lambda x : np.exp(x)
        f2 = lambda x : np.exp(x) 
    else:
        raise ValueError("G2_type must be an integer between 1 and 5")
    return f0, f1, f2 


def torchG2(G2_type=1):
    if G2_type == 1:
        f0 = lambda x : -torch.sqrt(-x)
        f1 = lambda x : 0.5 / torch.sqrt(-x)
    elif G2_type == 2:
        f0 = lambda x : -torch.log(-x)
        f1 = lambda x : -1 / x
    elif G2_type == 3:
        f0 = lambda x : -1 / x
        f1 = lambda x : 1 / torch.pow(x, 2)
    elif G2_type == 4:
        f0 = lambda x : torch.log(1 + torch.exp(x))
        f1 = lambda x : torch.exp(x) / (1 + torch.exp(x))
    elif G2_type == 5:
        f0 = lambda x : torch.exp(x)
        f1 = lambda x : torch.exp(x)
    else:
        raise ValueError("G2_type must be an integer between 1 and 5")
    return f0, f1


def make_train_step_fn(model, loss_fn, optimizer):
    '''
        Builds function that performs a step in the training loop
    '''
    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)
        if yhat.shape[1] == 1:
            loss = loss_fn(yhat, y.view_as(yhat))
        else:
            loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return perform_train_step_fn


def make_val_step_fn(model, loss_fn):
    def perform_val_step_fn(x, y):
        model.eval()
        yhat = model(x)
        if yhat.shape[1] == 1:
            loss = loss_fn(yhat, y.view_as(yhat))
        else:
            loss = loss_fn(yhat, y)
        return loss.item()
    return perform_val_step_fn


def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    return np.mean(mini_batch_losses)


def QuantLoss(tau=.5, h=.0):
    def loss(y_pred, y):
        z = y - y_pred
        if h == 0:
            return torch.max((tau - 1) * z, tau * z).mean()
        else:
            tmp = .5 * h * torch.sqrt(2/torch.tensor(np.pi))
            return torch.add((tau - Normal(0, 1).cdf(-z/h)) * z, 
                             tmp * torch.exp(-(z/h)**2/2)).mean()
    return loss


def JointLoss(tau=.5, G1=False, G2_type=1):
    f0, f1 = torchG2(G2_type)
    def loss(y_pred, y):
        if G2_type in {1, 2, 3}:
            ymax = y.max()
            y = y - ymax
            y_pred = y_pred - ymax
        z = y - y_pred[:,0]
        part1 = f1(y_pred[:,1]) * (y_pred[:,1] - y_pred[:,0] \
                   - torch.min(z, torch.tensor(0)) / tau) - f0(y_pred[:,1])
        if G1:
            part2 = torch.max((tau-1)*z, tau*z)
            return torch.add(part1, part2).mean()
        else: 
            return part1.mean()
    return loss


def HuberLoss(u, c=None):
    ''' Huber loss '''
    if c is None:
        out = 0.5 * u ** 2
    else:
        out = np.where(abs(u)<=c, 0.5*u**2, c*abs(u) - 0.5*c**2)
    return np.sum(out)


def HuberGrad(u, c=None):
    ''' Gradient of Huber loss '''
    if c is None:
        return u    
    else:
        return np.where(abs(u)<=c, u, c*np.sign(u))
