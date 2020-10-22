import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import math
from scipy.stats import ortho_group

# Algorithm
def BestOfBoth(T, V, alpha, K):
    """
    Best-of-both worlds algorithm
    input: T (Horizon), V (relative-weights parameter), alpha (step-size), K (number of ) 
    output: sequence of x_t
    """	
    # Initialize the meta-FW vectors (v_{t - 1}'s) and (v_t's)
    v_prev = np.zeros((d, K))
    v_curr = np.zeros((d, K))
    # Initialize the dual variables
    lambdas = np.zeros(K)
    # partial decision vectors x_t(k)
    X_partial_prev = np.zeros((d, K + 1))
    X_partial_curr = np.zeros((d, K + 1))
    # Store the decisions to return 
    XX = np.zeros((d, T))
    for t in range(1, T):
        # Primal update
        X_partial_curr[:, 0] = np.zeros(d)
        for k in range(K):
            new_position = v_prev[:, k] + (1/2/alpha)*(V*gradFLog(t - 1, X_partial_prev[:, k]) - lambdas[k]*gradG(t - 1, v_prev[:, k]))
            # Perform projection back to ground constraint set
            # Projection step 1
            for s in range(d):
                if new_position[s] < 0:
                    new_position[s] = 0
                elif new_position[s] > 1:
                    new_position[s] = 1
            v_curr[:, k] = new_position
            X_partial_curr[:, k + 1] = X_partial_curr[:, k] + (1/K)*v_curr[:, k]
        XX[:, t] = roundVec(X_partial_curr[:, K])
        # Dual update
        for k in range(K):
            lambdas[k] = max(0, lambdas[k] + g(t - 1, v_prev[:, k]) - B + np.dot(gradG(t - 1, v_prev[:, k]), v_curr[:, k] - v_prev[:, k]))
        # Set current values to be the previous ones
        X_partial_prev = X_partial_curr
        v_prev = v_curr
    return XX

def fLog(t, x):
    """
    This is the objective function oracle.
    input: time t and decision vector x (and function rollout history H)
    output: f_t(x_t)
    """
    Lt = H[t,:]
    L = np.dot(Q,np.dot(np.diag(Lt), Q.T))
    out = np.log(np.linalg.det(np.dot(np.diag(x), L - np.eye(d)) + np.eye(d)))
    return out

def gradFLog(t, x):
    """
    This is the objective function's gradient oracle.
    input: time t and decision vector x (and function rollout history H)
    output: grad(f_t(x_t))
    """
    Lt = H[t,:]
    L = np.dot(Q,np.dot(np.diag(Lt), Q.T))
    out=np.zeros(d)
    for i in range(d):
        postMulti = np.zeros((d,d))
        postMulti[i,i] = 1

        term2 = np.dot(L, postMulti) # (L-I)_i
        term1 = np.dot(np.diag(x), L - np.eye(d)) + np.eye(d) # diag(x)(L-I) + I
        term1_inv = np.linalg.inv(term1)
        out[i] = np.trace(np.dot(term1_inv, term2))
    return out

def g(t, x):
    """
    This is the constraint function oracle.
    input: time t and decision vector x (and function rollout history P)
    output: g_t(x_t) = <p_t, x_t> - B
    """	
    M = np.dot(np.dot(R.T, np.diag(P[t, :])), R)
    return np.dot(x.T, np.dot(M, x)) 

def gradG(t, x):
    """
    This is the constraint function's gradient oracle.
    input: time t and decision vector x
    output: grad(g_t(x))
    """
    M = np.dot(np.dot(R.T, np.diag(P[t, :])), R)
    return np.dot(M + M.T, x)

def roundVec(X):
    """
    Rounds the entries to zero/one
    """
    return X

################################ START PROBLEM ################################
# dimension of ambient space of decision variables
d = 10
# Budget for each round B_T/T
B = 4
# Horizon of online decision making (T)
T = 1000
# Rollout of objective functions f(x) = log(det(diag(x)(L-I) + I))
# L = Q*diag(H[t,:])*Q.T
# We take the t th row of H and get L from it
u = np.ones(d)
Q = ortho_group.rvs(dim = d)
R = ortho_group.rvs(dim = d)

numH = 10
H_sum = np.zeros((T,d))
P_sum = np.zeros((T,d))
p_elements = [2, 1, 5, 2, 3]
p = np.zeros(d)
for i in range(d):
    p[i] = p_elements[np.random.randint(5)]
for h in range(numH):
    H = np.random.uniform(2, 3, (T, d))
    H_sum = H_sum + H
    # Rollout of constraint function <p_t,x> - B.  
    # Note: E[p_t] = p
    # Each row corresponds to one p_t 
    # p = np.array([2, 1, 5, 2, 3])
    P = np.random.uniform(-0.7, 0.7, (T, d))
    for i in range(T):
        P[i, :] = P[i, :] + p
    P_sum += P
# p = np.array([2, 1, 5, 2, 3])
P = P_sum/numH
H = H_sum/numH

G=np.linalg.norm(P,np.inf)-1

# Input parameters
K = np.int(T**0.5)
V = np.int(T**0.5)
alpha = T

# Run the algorithm for these parameters
# This is the code for experiment 3
# We plot the average_utility and average_budget_violation in Figure 1(c) in the paper
factors = [0.1, 1/2, 2/3, 3/4, 0.85, 5/4]
f_historyAll = np.zeros((6, T))
g_historyAll = np.zeros((6, T))
counter = 0
for factor in factors:
    # Run the algorithm and obtain the data for utility and budget violation
    X_historyBOB = BestOfBoth(T, np.int(T**factor), np.int(T**(factor + 0.5)), K)

    f_historyBOB = np.zeros(T)
    f_historyBOB[0] = fLog(0, X_historyBOB[:, 0])
    for t in range(1, T):
        f_historyBOB[t] = f_historyBOB[t-1] + fLog(t, X_historyBOB[:, t])
    g_historyBOB = np.zeros(T)
    g_historyBOB[0] = g(0, X_historyBOB[:, 0]) - B
    for t in range(1, T):
        g_historyBOB[t] = g_historyBOB[t-1] + g(t, X_historyBOB[:, t]) - B
    f_historyAll[counter, :] = f_historyBOB
    g_historyAll[counter, :] = g_historyBOB
    counter += 1

average_budget_violation = np.zeros((6, T))
average_utility = np.zeros((6, T))
for counter in range(6):
    for t in range(1, T):
        average_budget_violation[counter, t] = g_historyAll[counter, t]/t
        average_utility[counter, t] = f_historyAll[counter, t]/t