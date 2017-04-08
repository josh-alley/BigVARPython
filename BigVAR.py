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

        # TODO verify this step
        if len(VARX) != 0:
            k = VARX.shape[1]
            if k > Y.shape[1]: raise ValueError('k is greater than the number of columns in Y')
        else: k = Y.shape[1]
        self.m = Y.shape[1] - k
        self.n_series = Y.shape[1] - (self.m if self.m < Y.shape[1] else 0)
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
        self.struct = struct
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

def cross_validate(bv):

    if len(bv.alpha) == 0:
        if len(bv.VARX) > 0:
            alpha = 1/(bv.VARX/(bv.k) + 1)
        else:
            alpha = 1/(bv.k + 1)

    dual = len(bv.alpha > 1) and bv.struct in ['SparseLag', 'SparseOO']
    jj = 0
    T1 = bv.T1 if bv.cv == 'Rolling' else bv.p + 2
    T2 = bv.T2
    s = bv.VARX[['s']] if len(bv.VARX) != 0 else 0
    if bv.own_lambdas:
        gamm = bv.gran
        gran2 = len(gamm)
    ONESE = bv.ONESE

    if (bv.cv == 'Rolling'):
        T1 = T1 - np.max(bv.p, s)
        T2 = T2 - np.max(bv.p, s)
    if not bv.own_lambdas:
        gran2 = bv.gran[1]
        gran1 = bv.gran[0]

    # constructing a lag matrix in VARX setting
    if len(bv.VARX) != 0:
        VARX = True
        k1 = bv.VARX[['k']]
        s = bv.VARX[['s']]
        # TODO contemp VARX
        ## if --> VARX$contemp
        contemp = False
        s1 = 0

        m = bv.Y.shape[1] - k1 # k - k1
        Y1 = np.matrix(bv.Y[:,0:k1])
        # TODO complete VARX estimation
    else: # VAR estimation
        contemp = False
        if bv.struct == 'Lag' or bv.struct == 'SparseLag':
            jj  = groupFun(bv.p, bv.Y.shape[1])
        else:
            jj = lFunction3(bv.p, bv.Y.shape[1])
        Z1 = VARXCons(bv.Y, np.matrix(np.zeros(shape=bv.Y.shape)), bv.Y.shape[1], bv.p, 0, 0)
        trainZ = Z1[1:Z1.shape[0],]
        #trainY = np.matrix(bv.Y[bv.p:bv.Y.shape[0])





