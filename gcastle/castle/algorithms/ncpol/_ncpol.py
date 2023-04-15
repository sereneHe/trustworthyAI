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

    See Also
    --------
    import sys
    sys.path.append("/home/zhouqua1/NCPOP") 

    Examples
    --------
    >>> import numpy as np
    >>> Y=[1,2,3]
    >>> X=[1,2,3]
    >>> T=len(Y)
    >>> level=1
    >>> ncpolr = NCPOLR()
    >>> y_pred = ncpolr.estimate(X, Y, T, level)
    >>> #### y_pred,y_residuals
    >>> print(y_pred)
    """

    def __init__(self, **kwargs):
        None

    def estimate(self, x, y, T, level):
        """Fit Estimator based on NCPOP Regressor model and predict y.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        x : array
            Variable seen as cause
        y: array
            Variable seen as effect
        T : int, length of estimation data
            default as 3
        level : int, required relaxation level
            default as 1

        Returns
        -------
        y_predict: array
            regression predict values of y
        """

        # Decision Variables
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=True)[0]
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=True)
        n = generate_operators("m", n_vars=T, hermitian=True, commutative=True)

        # Objective
        obj = sum((y[i]-f[i])**2 for i in range(T)) + 0.5*sum(f[i]**2 for i in range(T))

        # Constraints
        ine1 = [f[i] - G*x[i] - n[i] for i in range(T)]
        ine2 = [-f[i] + G*x[i] + n[i] for i in range(T)]
        ines = ine1+ine2

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G,f,n]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.') 
            y_predict = [sdp[n[i]] for i in range(T)] 
        else:
            print('Cannot find feasible solution.')
        return y_predict


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
    >>> from castle.algorithms.ncpol._ncpol import ANM_NCPOP,NCPOLR
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> import numpy as np

    >>> data = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
    >>> X = data['x']
    >>> true_dag = data['y']

    >>> anmNCPO = ANM_NCPOP(alpha=0.05)
    >>> anmNCPO.learn(data=X)

    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name='result')

    you can also provide more parameters to use it. like the flowing:
    >>> from sklearn.gaussian_process.kernels import Matern, RBF
    >>> anmNCPO = ANM_NCPOP(alpha=0.05)
    >>> anmNCPO.learn(data=X, regressor=NCPOLR())
    >>> # plot predict_dag and true_dag
    >>> GraphDAG(anmNCPO.causal_matrix, true_dag, show=False, save_name='result')
    """

    def __init__(self, alpha=0.05):
        super(ANM_NCPOP, self).__init__()
        self.alpha = alpha

    def learn(self, data, columns=None, regressor=NCPOLR(), test_method=hsic_test, **kwargs):
        """Set up and run the ANM_NCPOP algorithm.
        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        regressor: Class
            Nonlinear regression estimator, if not provided, it is NCPOP.
            If user defined, must implement `estimate` method. such as :
                `regressor.estimate(x, y, T, level)`
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
            x = data[:, i].reshape((-1, 1))
            y = data[:, j].reshape((-1, 1))

            flag = test_method(x, y, alpha=self.alpha)
            if flag == 1:
                continue
            # test x-->y
            flag = self.anmNCPO_estimate(x, y, regressor=regressor,
                                     test_method=test_method)
            if flag:
                self.causal_matrix[i, j] = 1
            # test y-->x
            flag = self.anmNCPO_estimate(y, x, regressor=regressor,
                                     test_method=test_method)
            if flag:
                self.causal_matrix[j, i] = 1

    def anmNCPO_estimate(self, x, y, T, level, regressor=NCPOLR(), test_method=hsic_test):
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
                `regressor.estimate(x, y, T, level)`
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
        >>> from castle.algorithms.ncpol import ANM_NCPOP
        >>> rawdata = np.load('dataset/linear_gauss_6nodes_15edges.npz', allow_pickle=True)
        >>> data = rawdata['x'][:20]
        >>> true_dag = data_load['y'][:20]
        >>> data = pd.DataFrame(data)
        >>> Y=data[0] 
        >>> X=data[1]
        >>> T=len(Y)
        >>> level=1
        >>> anmNCPO = ANM_NCPOP(alpha=0.05)
        >>> print(anmNCPO.anmNCPO_estimate(Y, X, T, level))
        1
        """

        x = scale(x).reshape((-1, 1))
        y = scale(y).reshape((-1, 1))
        T=len(y)
        level=1
        y_predict = regressor.estimate(x, y, T, level)
        flag = test_method(y - y_predict, x, alpha=self.alpha)

        return flag
