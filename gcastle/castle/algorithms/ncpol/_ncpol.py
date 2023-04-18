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

    See Also
    --------
    import sys
    sys.path.append("/home/zhouqua1/NCPOP") 
    
    Examples
    --------
    >>> import numpy as np
    >>> Y=[1,2,3]
    >>> X=[1,2,3]
    >>> ncpolr = NCPOLR()
    >>> y_pred = ncpolr.estimate(X, Y)
    >>> print(y_pred)
    The problem has 7 commuting variables
    Calculating block structure...
    Estimated number of SDP variables: 35
    Generating moment matrix...
    Reduced number of SDP variables: 35 35 (done: 102.86%, ETA 00:00:-0.0)
    Processing 6/6 constraints...
    Problem
      Name                   :                 
      Objective sense        : minimize        
      Type                   : CONIC (conic optimization problem)
      Constraints            : 35              
      Affine conic cons.     : 0               
      Disjunctive cons.      : 0               
      Cones                  : 0               
      Scalar variables       : 0               
      Matrix variables       : 1               
      Integer variables      : 0               

    Optimizer started.
    Presolve started.
    Linear dependency checker started.
    Linear dependency checker terminated.
    Eliminator started.
    Freed constraints in eliminator : 0
    Eliminator terminated.
    Eliminator - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - tries                  : 1                 time                   : 0.00            
    Lin. dep.  - number                 : 0               
    Presolve terminated. Time: 0.02    
    Problem
      Name                   :                 
      Objective sense        : minimize        
      Type                   : CONIC (conic optimization problem)
      Constraints            : 35              
      Affine conic cons.     : 0               
      Disjunctive cons.      : 0               
      Cones                  : 0               
      Scalar variables       : 0               
      Matrix variables       : 1               
      Integer variables      : 0               

    Optimizer  - threads                : 4               
    Optimizer  - solved problem         : the primal      
    Optimizer  - Constraints            : 35
    Optimizer  - Cones                  : 0
    Optimizer  - Scalar variables       : 0                 conic                  : 0               
    Optimizer  - Semi-definite variables: 1                 scalarized             : 105             
    Factor     - setup time             : 0.00              dense det. time        : 0.00            
    Factor     - ML order time          : 0.00              GP order time          : 0.00            
    Factor     - nonzeros before factor : 630               after factor           : 630             
    Factor     - dense dim.             : 0                 flops                  : 2.28e+04        
    ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME  
    0   6.0e+00  1.0e+00  2.0e+00  0.00e+00   1.000000000e+00   0.000000000e+00   1.0e+00  0.02  
    1   1.1e+00  1.9e-01  5.4e-01  -6.30e-01  3.150886775e+00   4.141925062e+00   1.9e-01  0.03  
    2   1.1e-01  1.8e-02  2.4e-02  2.54e-01   8.273194125e+00   8.578944991e+00   1.8e-02  0.03  
    3   1.8e-02  2.9e-03  1.6e-03  7.16e-01   9.187200810e+00   9.236036817e+00   2.9e-03  0.03  
    4   1.5e-03  2.5e-04  3.8e-05  1.02e+00   9.318520144e+00   9.321884971e+00   2.5e-04  0.03  
    5   1.3e-04  2.2e-05  9.5e-07  1.02e+00   9.331938540e+00   9.332204721e+00   2.2e-05  0.03  
    6   4.2e-06  7.1e-07  5.4e-09  1.01e+00   9.333286910e+00   9.333295069e+00   7.1e-07  0.03  
    7   4.2e-08  7.0e-09  5.3e-12  1.00e+00   9.333332864e+00   9.333332944e+00   7.0e-09  0.03  
    8   1.0e-10  2.7e-11  6.2e-16  1.00e+00   9.333333332e+00   9.333333332e+00   1.7e-11  0.03  
    Optimizer terminated. Time: 0.03    

    4.6666666678038755 4.666666667613962 optimal
    ok.
    [0.04446952 0.08893904 0.13340856]
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
        
        y_predict = regressor.estimate(x, y)
        flag = test_method(np.asarray(y - y_predict).reshape((-1, 1)), np.asarray(x).reshape((-1, 1)), alpha=self.alpha)
        # print(flag)
        
        return flag



