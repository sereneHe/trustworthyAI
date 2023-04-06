
import numpy as np
from sklearn.preprocessing import scale
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test
from ncpol2sdpa import*


class NCPOLR(object):
    """Estimator based on NCPOP Regressor
    Parameters
    ----------
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with Ridge.
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed"
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".
    optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be minimized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'L-BGFS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    normalize_y : bool, default=False
        Whether the target values y are normalized, the mean and variance of
        the target values are set equal to 0 and 1 respectively. This is
        recommended for cases where zero-mean, unit-variance priors are used.
        Note that, in this implementation, the normalisation is reversed
        before the GP predictions are reported.
    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    See Also
    --------

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(10).reshape((-1, 1))
    >>> y = np.random.rand(10).reshape((-1, 1))
    >>> gpr = GPR(alpha=1e-10)
    >>> y_pred = gpr.estimate(x, y)
    >>> print(y_pred)
    [[0.30898833]
     [0.51335394]
     [0.378371  ]
     [0.47051942]
     [0.51290679]
     [0.29678631]
     [0.77848816]
     [0.47589755]
     [0.21743226]
     [0.35258412]]
    """

    def __init__(self, **kwargs):
      None
        
    def estimate(self, x, y):
        """Fit NCPOP regression model and predict y.
        Parameters
        ----------
        x : array
            Variable seen as cause
        y: array
            Variable seen as effect
        Returns
        -------
        y_predict: array
            regression predict values of x
        """

        ## Insert ncpop
        # Y=[1,2,3]
        # X=[1,2,3]
        T=3
        level=1

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
        else:
            print('Cannot find feasible solution.')
            
        print(sdp[n[0]])
        
        
        # tmp=SimCom(Y,T,level)
        #     if (tmp):
        #         Z[i,j] = tmp
        #         N[i,j] = n
        #         break
        # Zdf=pd.DataFrame(Z)
        # Zdf.to_csv('ncpop100.csv',index=False) #,index=False

        return y_predict
