import numpy as np

# TODO make enum
supported_structs = ['Basic','Lag','SparseLag','OwnOther','SparseOO','HVARC','HVAROO','HVARELEM','Tapered','EFX']
supported_cv = ['Rolling', 'LOO']
class BigVAR:

    ''' 
    Y - T x k multivariate time series
    p - lagmax, maximal lag order for modeled series
    struct - Penalty Structure
    gran - granularity of penalty grid
    T1 - index of time series to start CV
    T2 index of time series to start Forecast evaluation
    RVAR - indicator for relaxed VAR
    h - desired forecast horizon
    cv - cross-validation procedure
    MN -Minnesota Prior Indicator
    verbose - indicator for verbose output
    IC - indicator for including AIC and BIC benchmarks
    VARX - VARX model specifications
    ONESE - 'One Standard Error' heuristic
    own_lambdas - user-supplied lambdas
    tf - transfer function
    alpha - grid of candidate alpha values (applies only to sparse VARX-L models)
    recursive - whether recursive multi-step forecasts are used (applies only to muliple horizon VAR models)
    C - vector of coefficients to shrink toward random walk
    dates - optional vector of dates corresponding to Y
    '''
    def __init__(self, Y, p, struct, gran, T1, T2, RVAR=False, h=1, cv='Rolling', MN=False, verbose=True, IC=True,
                 VARX=np.array([]), ONESE=False, own_lambdas=False, alpha=np.array([]), recursive=False, C = np.array([]), dates= None):

        if Y.shape[1] > Y.shape[0]: raise ValueError('k > T!')
        if p < 0: raise ValueError('p must be >= 0')
        if p == 0 and struct !='Basic': raise ValueError('Only Basic VARX-L supports a transfer function')
        if struct not in supported_structs:
            raise ValueError('penalty structure must be one of {}'.format(supported_structs))
        if h < 1: raise ValueError('h must be greater than 1!')
        if cv not in supported_cv: raise ValueError('Cross-Validation must be one of {}'.format(supported_cv))
        if len(gran) != 2 and not own_lambdas: raise ValueError('granularity must have two parameters')
        if gran[0] <=0 or gran[1] <= 0: raise ValueError('granularity parameters must be positive')

        if len(VARX) != 0:
            k = VARX.shape[1]
            if k > Y.shape[1]: raise ValueError('k is greater than the number of columns in Y')
        else: k = Y.shape[1]
        self.m = Y.shape[1] - k
        self.n_series = Y.shape[1] - (m if m < Y.shape[1] else 0)
        self.tf = (p == 0)
        if self.n_series == 1 and struct not in ['Basic', 'Lag', 'HVARC']:
            raise ValueError('Univariate support is only available for Lasso, Lag Group, and Componentwise HVAR')
        if len(VARX) == 0 and struct=='EFX': raise ValueError('EFX is only supported in the VARX framework')
        # TODO check for contemporaneous dependence
        structs = ['HVARC', 'HVAROO', 'HVARELEM']
        if len(VARX) != 0 and struct in structs: raise ValueError('EFZ is the only nested model supported in the VARX framework')
        if T1 > Y.shape[0] or T2 > Y.shape[0] or T2 < T1: raise ValueError('Training dates exceed series length')

        # TODO verify VARX specifications entered correctly

        if len(alpha) > 0 and any(a < 0 for a in alpha) and any(a > 1 for a in alpha): raise ValueError('alpha must be [0,1]')
        if len(C) != 0:
            if len(C) != k: raise ValueError('C must have length k')
            if not all(c == 0 or c== 1 for c in C): raise ValueError('Values of C must be either 0 or 1')
        else:
            self.C = [1]*k
        # TODO add logic for dates

        self.Y = Y
        self.p = p
        self.gran = gran
        self.T1 = T1
        self.T2 = T2
        self.RVAR = RVAR
        self.h = h
        self.cv = cv
        self.MN = MN
        self.verbose = verbose
        self.IC = IC
        self.VARX = VARX
        self.ONESE = ONESE
        self.own_lambdas = own_lambdas
        self.alpha = alpha
        self.recursive = recursive
        self.dates = dates


x = BigVAR(np.matrix([range(2), range(1,3)]), 1, 'Basic', [10,2], None, None)
