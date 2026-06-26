'''
    Convex optimizer using CVXPY.
'''

import cvxpy as cp
import numpy as np

class Optimizer:
    def __init__(self, prior_dist, normalized_objective_err_matrix, STATE_COUNT = 4, is_dist_preserve = True, ALPHA=0.001):

        self.prior_dist = prior_dist
        self.STATE_COUNT = STATE_COUNT
        self.ALPHA = ALPHA
        self.is_dist_preserve = is_dist_preserve
        self.__optimal_mechanism = -1
        self.normalized_objective_err_matrix = normalized_objective_err_matrix

    def get_mechanism(self, eps):
        self.__optimal_mechanism = self.optimize__(eps=eps)
        return self.__optimal_mechanism
    
    def optimize__(self, eps):
        X = cp.Variable((self.STATE_COUNT, self.STATE_COUNT))
        uniform_dist = np.ones(self.STATE_COUNT)/self.STATE_COUNT
        # Objective function
        # objective = cp.Minimize(np.transpose(self.prior_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
        objective = cp.Minimize(np.transpose(uniform_dist)@cp.sum(cp.multiply(X, self.normalized_objective_err_matrix), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)
       
        # Constraints
        X_T = X.T
        # Y = X_T@self.prior_dist
        Y = X_T@uniform_dist
        constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
                    X >= 0.000001] # Non-negative elements

        # Adding KL divergence constraints for each row
        if self.is_dist_preserve:
            for i in range(self.STATE_COUNT):
                constraints.append(Y[i] - self.prior_dist[i] <= self.ALPHA)
                constraints.append(self.prior_dist[i] - Y[i] <= self.ALPHA)

        # Adding LDP constraints
        for i in range(self.STATE_COUNT):
            for j in range(self.STATE_COUNT):
                if j == i:
                    continue
                constraints.append(X[i, i] - np.exp(eps)*X[j, i] <= 0)
                constraints.append(X[i, i] - X[j, i] >= 0)

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.HIGHS, verbose=False, ipm_optimality_tolerance=1e-6, primal_feasibility_tolerance=1e-7, dual_feasibility_tolerance=1e-7, highs_options={"solver": "ipm"})

        # Output the optimized matrix
        matrix = np.maximum(np.array(X.value), 0)
        row_sums = matrix.sum(axis=1, keepdims=True)

        return matrix/row_sums
