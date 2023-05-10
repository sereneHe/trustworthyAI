from typing import List
from ncpol2sdpa import*
import numpy as np
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from gcastle.castle.common import BaseLearner, Tensor
from gcastle.castle.common.independence_tests import hsic_test

"""
variable example

Y=[1,2,3]
X=[1,2,3]
T=3
level=1

"""
def ncpop_gpr(Y,X,level=1):
    T = len(Y)

    # =========================================================================
    # Decision Variables
    G = generate_variables("G", n_vars=1)[0]
    # L = generate_variables("L", n_vars=1)[0]
    u = generate_variables("u", n_vars=T)
    f = generate_variables("f", n_vars=T, hermitian=True, commutative=False)
    n = generate_variables("n", n_vars=T, hermitian=True, commutative=False)
    p = generate_variables("p", n_vars=T, hermitian=True, commutative=False)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(p[i] for i in range(T))
    # Constraints
    K1 = (np.array(f)-np.array(u)).reshape(-1,1)*(np.array(f)-np.array(u)).reshape(1,-1)
    dist = squareform(pdist(np.array(X).reshape(-1,1), 'sqeuclidean'))
    K2 = np.exp(-0.5*dist)
    ine1 = [G*K2[i,j]-K1[i,j] for i in range(len(K1)) for j in range(len(K1.T))]
    ine2 = [K1[i,j]-G*K2[i,j] for i in range(len(K1)) for j in range(len(K1.T))]
    ine3 = [p[i] + n[i] for i in range(T)]
    ine4 = [p[i] - n[i] for i in range(T)]
    ines = ine1+ine2+ine3+ine4
    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,u,f,p,n]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines,chordal_extension=True)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise = []
        pred_Y = []
        params = sdp[G]
        for i in range(T):
            # est_noise.append(sdp[n[i]])
            pred_Y.append(sdp[f[i]])
        est_noise_ = est_noise
        return pred_Y, params, est_noise_

    else:
        print('Cannot find feasible solution.')



def ncpop_linear(Y,X,level=1):

    # T=Y.shape[0]
    T = len(Y)

    # =========================================================================
    # Decision Variables
    G = generate_variables("G", n_vars=1, hermitian=True, commutative=False)[0]
    f = generate_variables("f", n_vars=T, hermitian=True, commutative=False)
    n = generate_variables("n", n_vars=T, hermitian=True, commutative=False)
    p = generate_variables("p", n_vars=T, hermitian=True, commutative=False)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.35*sum(p[i] for i in range(T))
    # Constraints
    ine1 = [f[i] - G*X[i] - n[i] for i in range(T)]
    ine2 = [-f[i] + G*X[i] + n[i] for i in range(T)]
    ine3 = [p[i] + n[i] for i in range(T)]
    ine4 = [p[i] - n[i] for i in range(T)]
    ines = ine1+ine2+ine3+ine4

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,f,n,p]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines,chordal_extension=True)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise = []
        pred_Y = []
        params = sdp[G]
        for i in range(T):
            est_noise.append(sdp[n[i]])
            pred_Y.append(sdp[f[i]])
        est_noise_ = [est_noise]
        return pred_Y, params, est_noise_

    else:
        print('Cannot find feasible solution.')

def ncpop_hidden(Y,X,level=1):

    # T=Y.shape[0]
    T = len(Y)

    # =========================================================================
    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
    Fdash = generate_operators("Fdash", n_vars=1, hermitian=True, commutative=False)[0]
    phi=generate_operators("phi", n_vars=T, hermitian=True, commutative=False)
    f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
    p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)
    q = generate_operators("q", n_vars=T, hermitian=True, commutative=False)
    n = generate_operators("n", n_vars=T, hermitian=True, commutative=False)
    m = generate_operators("m", n_vars=T, hermitian=True, commutative=False)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(n[i]+m[i] for i in range(T))
    # Constraints
    ine1 = [f[i] - Fdash*phi[i] - q[i] for i in range(T)]
    ine2 = [-f[i] + Fdash*phi[i] + q[i] for i in range(T)]
    ine3 = [phi[i] - G*X[i] - p[i] for i in range(T)]
    ine4 = [-phi[i] + G*X[i] + p[i] for i in range(T)]
    ine5 = [p[i] + n[i] for i in range(T)]
    ine6 = [p[i] - n[i] for i in range(T)]
    ine7 = [q[i] + m[i] for i in range(T)]
    ine8 = [q[i] - m[i] for i in range(T)]
    ines = ine1+ine2+ine3+ine4+ine5+ine6+ine7+ine8

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,phi,f,n,m,p,q]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines,chordal_extension=True)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise1 = []
        est_noise2 = []
        pred_Y = []
        params1 = sdp[G]
        params2 = sdp[Fdash]
        for i in range(T):
            est_noise1.append(sdp[p[i]])
            est_noise2.append(sdp[q[i]])
            pred_Y.append(sdp[f[i]])
        params = [params1, params2]
        est_noise = [est_noise2, est_noise1]
        return pred_Y, params, est_noise
    else:
        print('Cannot find feasible solution.')


    # =========================================================================
def ncpop_hidden2(Y,X,level=1):

    # T=Y.shape[0]
    T = len(Y)
    D = 5

    # Decision Variables
    G = generate_variables("G", n_vars=D)
    Fdash = generate_variables("Fdash", n_vars=D)
    phi = generate_variables('phi', n_vars=T*D)
    p = generate_variables("p", n_vars=T*D)
    q = generate_variables("q", n_vars=T*1)
    n = generate_variables("n", n_vars=T*1)
    f = generate_variables("f", n_vars=T)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(n[i] for i in range(T))
    # Constraints
    ine1 = [f[i] - sum(Fdash[d]*phi[i*D+d] for d in range(D)) - q[i] for i in range(T)]
    ine2 = [-f[i] + sum(Fdash[d]*phi[i*D+d] for d in range(D)) + q[i] for i in range(T)]
    ine3 = [phi[i] - sum(G[d]*X[i] + p[i*D+d] for d in range(D))  for i in range(T)]
    ine4 = [phi[i] + sum(G[d]*X[i] + p[i*D+d] for d in range(D)) for i in range(T)]
    ine5 = [q[i] + n[i] for i in range(T)]
    ine6 = [q[i] - n[i] for i in range(T)]
    ines = ine1+ine2+ine3+ine4+ine5+ine6

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,Fdash,phi,p,q,f,n]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines,chordal_extension=True)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise1 = []
        est_noise2 = []
        pred_Y = []
        # params1 = sdp[G]
        # params2 = sdp[Fdash]
        for i in range(T):
            est_noise1.append(sdp[p[i]])
            est_noise2.append(sdp[q[i]])
            pred_Y.append(sdp[f[i]])
        # params = [params1, params2]
        params = 0
        est_noise = [est_noise2, est_noise1]
        return pred_Y, params, est_noise
    else:
        print('Cannot find feasible solution.')

    print(sdp[n[0]])

def ncpop_polyn(Y,X,level=1):

    # T=Y.shape[0]
    T = len(Y)

    # Decision Variables
    G = generate_variables("G", n_vars=3, hermitian=True, commutative=False)
    f = generate_variables("f", n_vars=T, hermitian=True, commutative=False)
    n = generate_variables("m", n_vars=T, hermitian=True, commutative=False)

    # Objective
    obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(n[i]**2 for i in range(T))
    # Constraints
    ine1 = [f[i] - G[0]*X[i]**3 - G[1]*X[i]**2 - G[2]*X[i] - n[i] for i in range(T)]
    ine2 = [-f[i] + G[0]*X[i]**3 + G[1]*X[i]**2 + G[2]*X[i] + n[i] for i in range(T)]
    ines = ine1+ine2

    # Solve the NCPO
    sdp = SdpRelaxation(variables = flatten([G,f,n]),verbose = 1)
    sdp.get_relaxation(level, objective=obj, inequalities=ines)
    sdp.solve(solver='mosek')
    #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
    print(sdp.primal, sdp.dual, sdp.status)

    if(sdp.status != 'infeasible'):
        print('ok.')
        est_noise = []
        pred_Y = []
        params = G
        for i in range(T):
            est_noise.append(sdp[n[i]])
            pred_Y.append(sdp[f[i]])
        est_noise_ = [est_noise]
        return pred_Y, params, est_noise_
    else:
        print('Cannot find feasible solution.')
        return



class NCPLinear(BaseLearner):
    """
    # introductions

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold

    Attributes
    ----------
    causal_matrix : array like shape of (n_features, n_features)
        Learned causal structure matrix.



    """

    def __init__(self,alpha=0.05):
        super(NCPLinear, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None, test_method=hsic_test, m_ncpop=ncpop_polyn, **kwargs):
        """Set up and run the ncpop algorithm.

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is GPR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        """

        # self.regressor = regressor

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        node_num = data.shape[1]
        self.causal_matrix = Tensor(np.zeros((node_num, node_num)),
                                    index=data.columns,
                                    columns=data.columns)
        self.real = [] #共6*5条
        self.pred = [] #共6*5条
        self.est_noise = [] #共6*5条

        for i, j in combinations(range(node_num), 2):
            x = data[:, i].reshape((-1, 1))
            y = data[:, j].reshape((-1, 1))

            flag = test_method(x, y, alpha=self.alpha)
            if flag == 1: 
                continue
            # test x-->y
            _,flag,y_pred, est_noise = self.ncpop_estimate(x, y, 
                                     test_method=test_method, m_ncpop=m_ncpop)
            self.real.append(y)
            self.pred.append(y_pred) 
            self.est_noise.append(est_noise)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            _,flag, y_pred, est_noise = self.ncpop_estimate(y, x, 
                                     test_method=test_method, m_ncpop=m_ncpop)
            self.real.append(x)
            self.pred.append(y_pred) 
            self.est_noise.append(est_noise)
            if flag:
                self.causal_matrix[j, i] = 1

        

    def ncpop_estimate(self, x, y, test_method=hsic_test, m_ncpop=ncpop_polyn):
        """Compute the fitness score of the ncpop model in the x->y direction.


        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y

        """

        # x = scale(x).reshape((-1, 1))
        # y = scale(y).reshape((-1, 1))

        X = list(x.flatten())
        Y = list(y.flatten())
            
        y_pred, params, est_noise = m_ncpop(Y,X)
        flag1 = []
        # flag1 = test_method(np.array(est_noise[0]).reshape((-1,1)), x, alpha=self.alpha)
        flag2 = test_method((y-np.array(y_pred).reshape((-1,1))), x, alpha=self.alpha)

        return flag1, flag2, y_pred, est_noise