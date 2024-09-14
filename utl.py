import numpy as np
import numpy.random as rgt
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge as KR

from quantes.nonlinear import KRR, LocPoly


###############################################################################
############################# Multivariate Model ##############################
###############################################################################
# Local-scale model with d-dimensional input
d = 8
md_fn = lambda x: np.cos(2*np.pi*(x[:,0])) \
                    + (1 + np.exp(-x[:,1]-x[:,2]))**(-1) + (1 + x[:,3] \
                        + x[:,4])**(-3) + (x[:,5] + np.exp(x[:,6]*x[:,7]))**(-1)
sd_fn = lambda x: np.sin(np.pi*(x[:,0] + x[:,1])*0.5) \
                    + np.log(1 + (x[:,2]*x[:,3]*x[:,4])**2) \
                        + x[:,7]*(1 + np.exp(-x[:,5]-x[:,6]))**(-1)


def gen_train_data(n=2000, n_val=None, random_state=0):
    '''
        Generate training and/or validation data from a 
        location-scale model with d-dimensional input
    '''
    rgt.seed(random_state)
    if n_val is None: n_val = n//4
    X = rgt.uniform(0, 1, (n, d))
    Y = md_fn(X) + sd_fn(X)*rgt.normal(0, 1, n)
    X_val = rgt.uniform(0, 1, (n_val, d))
    Y_val = md_fn(X_val) + sd_fn(X_val)*rgt.normal(0, 1, n_val)
    return X, Y, X_val, Y_val


def gen_test_data(n_test=1000, random_state=2024):
    '''
        Generate test data from a location-scale model 
        with d-dimensional input
    '''
    rgt.seed(random_state)
    X_test = rgt.uniform(0, 1, (n_test, d))
    Y_test = md_fn(X_test) + sd_fn(X_test)*rgt.normal(0, 1, n_test)
    return X_test, Y_test


###############################################################################
############################## Univariate Model ###############################
###############################################################################
m1_fn = lambda x: x * np.sin(1.75 * np.pi * x)
s1_fn = lambda x: .5 + abs(np.sin(np.pi*x/2))

def gen_train(n=1000, n_val=None, random_state=0):
    rgt.seed(random_state)
    if n_val is None: n_val = n//4
    X = rgt.uniform(0, 2, n)
    Y = m1_fn(X) + s1_fn(X)*rgt.normal(0, 1, n)
    X_val = rgt.uniform(0, 2, n_val)
    Y_val = m1_fn(X_val) + s1_fn(X_val)*rgt.normal(0, 1, n_val)
    return X.reshape(-1, 1), Y, X_val.reshape(-1, 1), Y_val

def gen_test(n_test=1000, random_state=2024):
    rgt.seed(random_state)
    X_test = rgt.uniform(0, 1, n_test)
    Y_test = m1_fn(X_test) + s1_fn(X_test)*rgt.normal(0, 1, n_test)
    return X_test.reshape(-1, 1), Y_test

def es_fn(x, tau):
    qt = norm.ppf(tau)
    es = norm.expect(lambda x : (x if x <= qt else 0)) / tau
    return m1_fn(x) + s1_fn(x) * es


###############################################################################
########################## Kernel Ridge Regression ############################
###############################################################################
def QtES_KR(X, Y, X_val, Y_val, tau=.5, kernel='rbf', 
            grid_q = np.array([0.5, 0.75, 1, 1.25, 1.5]),
            grid_e = np.array([1, 1.5, 2, 2.5, 3, 3.5]),
            kernel_params=dict(), other_params=None, 
            plot=False, solver='cvxopt', smooth=False):
    # Step 1: Qt-KRR
    if type(grid_q) == float or type(grid_q) == int:
        alpha_q = grid_q
        model = KRR(X, Y, kernel=kernel, kernel_params=kernel_params)
        model.qt(tau, alpha_q, solver=solver, smooth=smooth)
    elif type(grid_q) == np.ndarray and len(grid_q) == 1:
        alpha_q = grid_q[0]
        model = KRR(X, Y, kernel=kernel, kernel_params=kernel_params)
        model.qt(tau, alpha_q, solver=solver, smooth=smooth)       
    else:
        step1 = Qt_KR(X, Y, X_val, Y_val, tau, kernel,
                      grid_q, kernel_params, plot, solver, smooth)
        alpha_q = step1['alpha_q']

    # Step 2: ES-KRR
    if type(grid_e) == float or type(grid_e) == int:
        alpha_e = grid_e
        model.ES(tau, alpha_e, None, kernel, other_params)
    elif type(grid_e) == np.ndarray and len(grid_e) == 1:
        alpha_e = grid_e[0]
        model.ES(tau, alpha_e, None, kernel, other_params)
    else:
        step2 = ES_KR(X, Y, X_val, Y_val, tau, kernel, 
                      alpha_q, grid_e, 
                      kernel_params, other_params, 
                      plot, solver)
        alpha_e = step2['alpha_e']
        model = step2['model']

    return {'alpha_q': alpha_q, 'alpha_e': alpha_e, 'model': model}


def Qt_KR(X, Y, X_val, Y_val, tau=.5, kernel='rbf',
          grid_q=np.array([1, 1.5, 2, 2.5, 3, 3.5]),
          kernel_params=dict(), plot=False, 
          solver='cvxopt', smooth=False):
    
    model = KRR(X, Y, kernel=kernel, kernel_params=kernel_params)
    model.qt_seq(tau, grid_q, solver=solver, smooth=smooth, x=X_val)
    val_err_q = np.array([quantile_loss(model.pred_q[:,m], Y_val, tau) 
                          for m in range(0, len(grid_q))])     
    alpha_q = grid_q[val_err_q.argmin()]

    if plot:
        print('minimum qt-krr val error:', val_err_q.min().round(4))
        plt.plot(grid_q, val_err_q)
        plt.ylabel('validation error')
        plt.xlabel('alpha')
        plt.title('qt-krr')
        plt.show()
    
    return {'alpha_q': alpha_q, 'grid_q': grid_q, 'model': model}


def ES_KR(X, Y, X_val, Y_val, tau=.5, kernel='rbf', 
          alpha_q=1, grid_e=np.array([0.1, 0.5, 1, 1.5, 2]),
          kernel_params=dict(), other_params=None, 
          plot=False, solver='cvxopt'):
    model = KRR(X, Y, kernel=kernel, kernel_params=kernel_params)
    model.qt(tau, alpha_q, solver=solver)
    pred_q = model.qt_predict(X_val)
    Y0 = np.minimum(Y_val - pred_q, 0)/tau + pred_q
    model.ES_seq(tau, grid_e, kernel, other_params, x=X_val)
    val_err_e = np.array([(((Y0 - model.pred_e[:,m]))**2).mean()
                          for m in range(0, len(grid_e))])
    alpha_e = grid_e[val_err_e.argmin()]

    if plot:
        print('minimum es-krr val error:', np.min(val_err_e).round(4))
        plt.plot(grid_e, val_err_e)
        plt.ylabel('validation error')
        plt.xlabel('alpha')
        plt.title('es-krr')
        plt.show()

    model.ES(tau, alpha_e, kernel=kernel, other_params=other_params)
    return {'alpha_q': alpha_q, 'alpha_e': alpha_e, 
            'grid_e': grid_e, 'model': model}


def CV_Qt_KR(X, Y, tau, kernel, grid_q, kernel_params=dict(),
             nfolds=5, random_state=42, solver='cvxopt', smooth=False, tol=1e-10):
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    cv_err = np.empty([len(grid_q), nfolds])
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        model = KRR(X_train, y_train, kernel=kernel, kernel_params=kernel_params)
        model.qt_seq(tau, grid_q, smooth=smooth, solver=solver, tol=tol, x=X_test)
        cv_err[:,k] = np.array([quantile_loss(model.pred_q[:,m], y_test, tau) 
                                for m in range(0, len(grid_q))])
    cv_mean_err = cv_err.mean(axis=1)  
    return {'alpha_q': grid_q[cv_mean_err.argmin()],
            'err': cv_mean_err}


def CV_ES_KR(X, Y, tau, kernel, alpha_q, grid_e, kernel_params=dict(),
             nfolds=5, random_state=42, solver='cvxopt', smooth=False, 
             other_params = None, tol=1e-10):
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    m1 = KRR(X, Y, kernel=kernel, kernel_params=kernel_params)
    m1.qt(tau, alpha_q, solver=solver, smooth=smooth, tol=tol)
    fit_q = m1.fit_q

    cv_err = np.empty([len(grid_e), nfolds])
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        m2 = KRR(X_train, y_train, kernel=kernel, kernel_params=kernel_params)
        m2.fit_q = fit_q[train_idx]
        m2.ES_seq(tau=tau, alpha_seq=grid_e, x=X_test)
        Y0 = np.minimum(y_test - fit_q[test_idx], 0)/tau + fit_q[test_idx]
        cv_err[:,k] = np.array([(((Y0 - m2.pred_e[:,m]))**2).mean()
                                for m in range(0, len(grid_e))])
    cv_mean_err = cv_err.mean(axis=1)
    alpha_e = grid_e[cv_mean_err.argmin()]
    m1.ES(tau, alpha_e, kernel=kernel, other_params=other_params)

    return {'alpha_e': alpha_e,
            'err': cv_mean_err, 'model': m1}


def CV_KR(X, Y, kernel, grid, kernel_params = dict(),
          nfolds=5, random_state=42, other_params = None):
    params = {'gamma': 1, 'coef0': 1, 'degree': 3} # default values of kernel params
    params.update(kernel_params)
    args = {'kernel': kernel, 
            'gamma': params['gamma'],
            'coef0': params['coef0'], 
            'degree': params['degree'],
            'kernel_params': other_params}

    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    cv_err = np.empty([len(grid), nfolds])
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        for i, alpha in enumerate(grid):
            kr = KR(alpha, **args)
            kr.fit(X_train, y_train)
            cv_err[i,k] = np.mean((kr.predict(X_test) - y_test)**2)

    cv_mean_err = cv_err.mean(axis=1)

    alpha = grid[cv_mean_err.argmin()]
    model = KR(alpha, **args)
    model.fit(X, Y)
    return {'alpha': alpha, 'err': cv_mean_err, 'model': model}



###############################################################################
######################## Local Polynomial Regression ##########################
###############################################################################
def Qt_LP(X, Y, X_val, Y_val, tau, kernel, grid_q, degree=1, plot=False):
    model = LocPoly(X, Y, kernel=kernel)
    val_err = []
    for h in grid_q:
        pred_q = model.qt_predict(x0=X_val, bw=h, tau=tau, degree=degree)
        val_err.append(quantile_loss(pred_q, Y_val, tau))
    val_err = np.array(val_err)
    if plot:
        print('minimum qt-llr val error:', val_err.min().round(4))
        plt.plot(grid_q, val_err)
        plt.ylabel('validation error')
        plt.xlabel('bandwidth')
        plt.title('qt-llr')
        plt.show()

    bw_q = grid_q[val_err.argmin()]
    fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree)
    return {'bw_q': bw_q,
            'fit_q': fit_q,
            'surrogate_y': np.minimum(Y - fit_q, 0)/tau + fit_q}


def ES_LP(X, Y, X_val, Y_val, tau, kernel_q, bw_q, kernel_e, grid_e, 
          degree=1, plot=False):
    model1 = LocPoly(X, Y, kernel=kernel_q)
    fit_q = model1.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree)
    Y0 = np.minimum(Y - fit_q, 0)/tau + fit_q
    pred_q = model1.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree)
    Y0_val = np.minimum(Y_val - pred_q, 0)/tau + pred_q
    
    val_err = []
    model2 = LocPoly(X, Y0, kernel=kernel_e)
    for h in grid_e:
        pred_e = model2.ls_predict(x0=X_val, bw=h, degree=degree)
        val_err.append(np.mean((Y0_val - pred_e)**2))
    val_err = np.array(val_err)

    if plot:
        print('minimum es-llr val error:', val_err.min().round(4))
        plt.plot(grid_e, val_err)
        plt.ylabel('validation error')
        plt.xlabel('bandwidth')
        plt.title('es-llr')
        plt.show()

    return {'bw_q': bw_q, 'bw_e': grid_e[val_err.argmin()], 'surrogate_y': Y0}


def QtES_LP(X, Y, X_val, Y_val, tau,
            kernel_q=norm.pdf, grid_q=np.linspace(.1, 1, 10), degree_q=1,
            kernel_e=norm.pdf, grid_e=np.linspace(.25, 1, 10), degree_e=1,
            plot=False):
    
    model = LocPoly(X, Y, kernel=kernel_q)
    if type(grid_q) == float or type(grid_q) == int:
        bw_q = grid_q
        fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)
        pred_q = model.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree_q)
    elif type(grid_q) == np.ndarray and len(grid_q) == 1:
        bw_q = grid_q[0]
        fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)
        pred_q = model.qt_predict(x0=X_val, bw=bw_q, tau=tau, degree=degree_q)
    else:
        val_err_q = []
        qt_pred = np.empty(shape=[len(Y_val), len(grid_q)])
        for m, h in enumerate(grid_q):
            qt_pred[:,m] = model.qt_predict(x0=X_val, bw=h, 
                                            tau=tau, degree=degree_q)
            val_err_q.append(quantile_loss(qt_pred[:,m], Y_val, tau))
        val_err_q = np.array(val_err_q)
        bw_q = grid_q[val_err_q.argmin()]
        pred_q = qt_pred[:,val_err_q.argmin()]
        fit_q = model.qt_predict(x0=X, bw=bw_q, tau=tau, degree=degree_q)

        if plot:
            print('minimum qt-localpoly val error:', 
                  val_err_q.min().round(4))
            plt.plot(grid_q, val_err_q)
            plt.ylabel('validation error')
            plt.xlabel('bandwidth')
            plt.title('qt-localpoly ')
            plt.show()

    Y0 = np.minimum(Y - fit_q, 0)/tau + fit_q
    Y0_val = np.minimum(Y_val - pred_q, 0)/tau + pred_q
    
    if type(grid_e) == float or type(grid_e) == int:
        bw_e = grid_e
    elif type(grid_e) == np.ndarray and len(grid_e) == 1:
        bw_e = grid_e[0]
    else:
        val_err_e = []
        model2 = LocPoly(X, Y0, kernel=kernel_e)
        for h in grid_e:
            pred_e = model2.ls_predict(x0=X_val, bw=h, degree=degree_e)
            val_err_e.append(np.mean((Y0_val - pred_e)**2))
        val_err_e = np.array(val_err_e)
        bw_e = grid_e[val_err_e.argmin()]

        if plot:
            print('minimum es-localpoly val error:', 
                  val_err_e.min().round(4))
            plt.plot(grid_q, val_err_e)
            plt.ylabel('validation error')
            plt.xlabel('bandwidth')
            plt.title('es-localpoly ')
            plt.show()

    return {'bw_q': bw_q, 'bw_e': bw_e, 'surrogate_y': Y0, 'fit_q': fit_q}


###############################################################################
############################## Helper Functions ###############################
###############################################################################
def quantile_loss(yhat, y, tau):
    '''
    Quantile Loss Function
    '''
    error = y - yhat
    return np.maximum(tau*error, (tau-1)*error).mean()


def check_loss(x, tau):
    tmp = np.where(x > 0, tau * x, (tau - 1) * x)
    return np.mean(tmp)


def mad(x):
    ''' Median absolute deviation '''
    return 1.4826 * np.median(np.abs(x - np.median(x)))