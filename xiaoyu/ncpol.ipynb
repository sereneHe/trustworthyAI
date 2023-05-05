# coding: utf-8
## Xiaoyu He ##

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from itertools import combinations
from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test
from inputlds import*
from functions import*
from ncpol2sdpa import*
from math import sqrt

class NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    
    Examples
    --------
    >>> import numpy as np
    >>> Y=[1,2,3]
    >>> X=[1,2,3]
    >>> ncpolr = NCPOLR()
    >>> y_pred = ncpolr.estimate(X, Y)
    >>> print(y_pred)
    """
    
    def __init__(self, **kwargs):
        super(NCPOLR, self).__init__()

    def estimate(self, X, Y):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        X : array
            Variable seen as cause
        Y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        """
        
        T = len(Y)
        level = 1
    
        # Decision Variables
        # f=G*x+n以前是最小化n**2，现在就直接最小化p
        # G是系数
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        # f是y的估计值
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)
        # n是残差
        n = generate_operators("m", n_vars=T, hermitian=True, commutative=False)
        # p是n的绝对值
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.5*sum(p[i] for i in range(T))
        
        # Constraints
        ine1 = [f[i] - G*X[i] - n[i] for i in range(T)]
        ine2 = [-f[i] + G*X[i] + n[i] for i in range(T)]
        # fp和n的关系通过加新的限制条件p>n 和p>-n来实现
        ine3 = [p[i]-n[i] for i in range(T)]
        ine4 = [p[i]+n[i] for i in range(T)]
        ines = ine1+ine2+ine3+ine4

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G,f,n,p]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.')
            est_noise = []
            for i in range(T):
                est_noise.append(sdp[n[i]])
            print(est_noise)
            return est_noise
        else:
            print('Cannot find feasible solution.')
            return


class ANM_NCPOP(BaseLearner):
    """
    Nonlinear causal discovery with additive noise models

    Use Estimator based on NCPOP Regressor and independent Gaussian noise,
    For the independence test, we implemented the HSIC with a Gaussian kernel,
    where we used the gamma distribution as an approximation for the
    distribution of the HSIC under the null hypothesis of independence
    in order to calculate the p-value of the test result.
    
    References
    ----------
    Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters,
    Jonas and Schölkopf, Bernhard,
    "Nonlinear causal discovery with additive noise models", NIPS 2009

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold

    Attributes
    ----------
    causal_matrix : array like shape of (n_features, n_features)
        Learned causal structure matrix.
    
    Examples
    --------
    >>> # from castle.algorithms.ncpol._ncpol import NCPOLR,ANM_NCPOP
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> import numpy as np

    >>> rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
    >>> data = rawdata['x'][:10]
    >>> true_dag = rawdata['y'][:10]
    >>> #np.asarray(rawdata['y'][:10])
    >>> anmNCPO = ANM_NCPOP(alpha=0.05)
    >>> anmNCPO.learn(data=data)

    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name='result')
    >>> met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, alpha=0.05):
        super(ANM_NCPOP, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None,regressor=NCPOLR(),test_method=hsic_test, **kwargs):
        """Set up and run the ANM_NCPOP algorithm.
        
        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOLR.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        """

        self.regressor = regressor
        
        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        node_num = data.shape[1]
        self.causal_matrix = Tensor(np.zeros((node_num, node_num)),
                                    index=data.columns,
                                    columns=data.columns)

        for i, j in combinations(range(node_num), 2):
            x = data[:, i]
            y = data[:, j]            
            xx = x.reshape((-1, 1))
            yy = y.reshape((-1, 1))

            flag = test_method(xx, yy, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.anmNCPO_estimate(x, y, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.anmNCPO_estimate(y, x, regressor = regressor, test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def anmNCPO_estimate(self, x, y, regressor=NCPOLR(), test_method=hsic_test):
        """Compute the fitness score of the ANM_NCPOP Regression model in the x->y direction.

        Parameters
        ----------
        x: array
            Variable seen as cause
        y: array
            Variable seen as effect
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOP.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y)`
        test_method: callable, default test_method
            independence test method, if not provided, it is HSIC.
            If user defined, must accept three arguments--x, y and keyword
            argument--alpha. such as :
                `test_method(x, y, alpha=0.05)`
        Returns
        -------
        out: int, 0 or 1
            If 1, residuals n is independent of x, then accept x --> y
            If 0, residuals n is not independent of x, then reject x --> y
            
        Examples
        --------
        >>> import numpy as np
        >>> from castle.algorithms.ncpol._ncpol import ANM_NCPOP
        >>> rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
        >>> data = rawdata['x'][:20]
        >>> true_dag = rawdata['y'][:20]
        >>> data = pd.DataFrame(data)
        >>> Y=np.asarray(data[0])
        >>> X=np.asarray(data[1])
        >>> anmNCPO = ANM_NCPOP(alpha=0.05)
        """

        x = scale(x)
        y = scale(y)
        
        y_res = regressor.estimate(x, y)
        flag = test_method(np.asarray(y_res).reshape((-1, 1)), np.asarray(x).reshape((-1, 1)), alpha=self.alpha)
        print(flag)
        
        return flag

# from castle.algorithms.ncpol._ncpol import NCPOLR,ANM_NCPOP
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import numpy as np

rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
for i in range(10,2020,20):
    data = rawdata['x'][:i]
    true_dag = rawdata['y'][:i]
    anmNCPO = ANM_NCPOP(alpha=0.05)
    anmNCPO.learn(data=data)

    # plot predict_dag and true_dag
    sname = 'ncpol_result_'+str(i)
    GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name=sname)
    met = MetricsDAG(anmNCPO.causal_matrix, true_dag)
    print(met.metrics)
