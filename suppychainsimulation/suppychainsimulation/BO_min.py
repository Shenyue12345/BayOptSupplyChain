import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern,RBF


class BayeOpt():
    def __init__(self,
                 obj_func,
                 bounds, 
                 gpr_obj,
                 init_x, 
                 niter = 100,
                 n_restarts = 20): # replace Gaussian with Matern52 later
        """
        obj_func: function to minimize
        constraint_func: constraint function
        constraint_value: the minimum of allowable value
        bounds: the bounds of the value    np.array(d * 2)
        init_x: initial data    np.array(n * d)
        gpr: gaussian process for object function 
        gpr_constraint:  gaussian process for constraint
        niter: the iter times
        n_restarts: initial points to find minimums
        """
        # data parameters
        self.obj_func = obj_func
        self.bounds = bounds
        self.gpr_obj = gpr_obj
        self.init_x = init_x
        self.niter = niter
        self.n_restarts = n_restarts
        
        self.x = self.init_x
        self.y_obj = np.array([self.obj_func(t) for t in self.x]).reshape(-1, 1)
        # GP parameters
        self.f_best = self.y_obj.min()

        print(self.y_obj.shape, self.y_obj)
    
    def expected_improvement(self, X, xi=0.01):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.
        
        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.
        
        Returns:
            Expected improvements at points X.
        '''
        #print(X,self.gpr_obj.predict(X))
        mu_obj, sigma_obj = self.gpr_obj.predict(X, return_std=True)
        mu_sample = self.gpr_obj.predict(self.x)
        mu_obj = mu_obj.reshape(-1, 1)
        sigma_obj = sigma_obj.reshape(-1, 1)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        # compute ei
        mu_sample_opt = np.min(mu_sample)
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu_obj - xi
            Z = imp / sigma_obj
            
            ei = imp * norm.cdf(Z) + sigma_obj * norm.pdf(Z)
            ei[sigma_obj == 0.0] = 0.0
        return ei
    
    def propose_location(self):
        '''
        Proposes the next sampling point by optimizing the acquisition function.
        
        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''
        dim = self.x.shape[1]
        min_val = 1
        min_x = None
        
        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -1 * self.expected_improvement(X.reshape(-1, dim))
            # return -acquisition(X, X_sample, Y_sample, gpr)
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_restarts, dim)):
            #print("------------------------")
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')    #’Nelder-Mead’  ,'L-BFGS-B'  
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x           
                
        return min_x

    def bay_opt(self):
        # Initialize sample
        for _ in range(self.niter):
            m52_1 = ConstantKernel(1) * RBF(np.array([100]*12))
            self.gpr_obj = GaussianProcessRegressor(kernel=m52_1, alpha=10, noise = "gaussian")

            
            # Update Gaussian process with existing samples
            #print(self.x.shape,self.y_obj.shape )
            self.gpr_obj.fit(self.x, self.y_obj)
            #print(self.gpr_obj.predict(self.x))
            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = self.propose_location()
            
            
            # Obtain next noisy sample from the objective function
            Y_next1 = np.array([self.obj_func(X_next)]).reshape(-1,1)
           #print(Y_next1, Y_next1.shape, Y_next2,Y_next2.shape)
            # Add sample to previous samples
            self.x = np.vstack((self.x, X_next))
            self.y_obj = np.vstack((self.y_obj, Y_next1))
        self.f_best = np.min(self.y_obj)
        self.min_x = self.x[np.argmin(self.y_obj)]
        return self.f_best, self.min_x

