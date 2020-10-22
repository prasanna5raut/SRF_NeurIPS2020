import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import math

################################ DEFINE Functions and classes ################################

def OLFW(T, eps):
    """
    Online Lagragian Franke-Wolf implementation
    input: T (Horizon), eps (confidence) 
    output: sequence of x_t
    """	
    # Number of times each oracle is called 
    K = np.int(T**0.5)
    # Step size of Online Gradient Ascent
    mu = 1.0 / K/ beta
    # delta
    delta = beta**2
    # Initial point for the OGA oracles
    v0 = np.zeros(d)
    # Initialize  K instances of OGA oracle and store in a list
    listOracle = [OnlineGA(v0) for i in range(K)]
    # Initialize the current estimate of p
    p_hat = pEstimate()
    # Initialize gamma_t's
    gammas = np.zeros(T)
    # for t in range(1,T):
    # 	gammas[t] = (2*G**2*np.log(2.0*T/eps)/t)**(0.5)
    # 	gammas[t] = 0
    # Initialize the lambda_t's
    lambdas = np.zeros(T)
    # Store the decisions to return as output
    XX=np.zeros((d,T))

    for t in range(T):
        # Online GA step
        X_temp = np.zeros((d,K+1))
        for k in range(K):
            X_temp[:,k+1] = X_temp[:,k] + (1.0/K)*listOracle[k].v
        XX[:,t] = roundVec(X_temp[:,K])
        # Update dual variables
        lambdas[t] = (1/(delta*mu))*max(0,np.dot(p_hat.vec, XX[:,t]) - B - gammas[t])
        # Feedback the data to Online GA sub-routine
        for k in range(K):
            listOracle[k].update(mu, gradFJoke(t, X_temp[:,k]) - lambdas[t]*p_hat.vec)
        # Update the current estimate after nature reveals the data
        p_hat.update(t+1, P[t,:])

    return XX


def fM(t):
    """
    This returns Mt matrix in the definition of crowdsourcing objective (f_t(x) = sum(h_i(x_i)) + x^TM_tx)
    """
    M = M_matrix[:, t*d:(t + 1)*d]
    for i in range(d):
        M[i, i] = 0
    return M

def fJoke(t, x):
    """
    This is the objective function oracle.
    input: time t and decision vector x 
    output: f_t(x_t)
    """
    M = fM(t)
    out=np.dot(x.T,np.dot(M,x))
    for dim in range(d):
        out += (1 + scaling[dim])*np.log(1 + x[dim])
    return out

def gradFJoke(t, x):
    """
    This is the objective function's gradient oracle.
    input: time t and decision vector x 
    output: grad(f_t(x_t))
    """
    out=0
    M = fM(t)
    v = np.zeros(d)
    for dim in range(d):
        v[dim] = (1 + scaling[dim])/(1 + x[dim])
    out = v + np.dot(M+M.T,x)
    return out

def g(t, x):
    """
    This is the constraint function oracle.
    input: time t and decision vector x (and function rollout history P)
    output: g_t(x_t) = <p_t, x_t> - B
    """	
    return np.dot(P[t,:], x) 

def gradG(t, x):
    """
    This is the constraint function's gradient oracle.
    input: time t and decision vector x
    output: grad(g_t(x))
    """
    return P[t, :]

def roundVec(X):
    """
    Rounds the entries to zero/one
    """
    out = np.zeros(d)
    for dim in range(d):
        if (X[dim] >= 0.5):
            out[dim] = 1
        else:
            out[dim] = 0
    return out

class pEstimate:
    """
    This class stores the current estimate of p in the field named 'vec'
    """
    def __init__(self):
        self.vec = np.zeros(d)

    def update(self, t, p_t):
        """ Updates the current estimate by taking as input the current realization. """
        self.vec = (t-1.0)/t*self.vec + p_t/t


class OnlineGA:
    """ 
    This is the class for defining Online Gradient Ascent Oracle.
    The output of the oracle is stored as field named 'v'
    """
    def __init__(self, v0):
        """ Set the current output of oracle to be v0"""
        self.v = v0

    def update(self, mu, direction):
        """ Updates the output by moving 'mu' units along 'direction'. """
        self.v = self.v + mu*direction
        # Projection step 1
        for s in range(d):
            if self.v[s] < 0:
                self.v[s] = 0
            elif self.v[s] > 1:
                self.v[s] = 1

def OSPHG(T):
    """
    Online Lagragian Franke-Wolf implementation
    input: T (Horizon), eps (confidence) 
    output: sequence of x_t
    """
    # Number of times each oracle is called 
    K = np.int(T**0.5)
    # Step size of Online Gradient Ascent
    mu = 1.0 / K/ beta
    # delta
    delta = beta**2
    # Initial point for the OGA oracles
    v0 = np.zeros(d)
    # Initialize  K instances of OGA oracle and store in a list
    listOracle = [OnlineGA(v0) for i in range(K)]
    # Initialize the lambda_t's
    lambdas = np.zeros(T)
    # Store the decisions to return as output
    XX=np.zeros((d,T))

    for t in range(T):
        # Online GA step
        X_temp = np.zeros((d,K+1))
        for k in range(K):
            X_temp[:,k+1] = X_temp[:,k] + (1.0/K)*listOracle[k].v
        
        XX[:,t] = roundVec(X_temp[:,K])
        
        # Feedback the data to Online GA sub-routine
        for k in range(K):
            listOracle[k].update(mu, gradFJoke(t, X_temp[:,k]) - lambdas[t]*P[t, :])
        
        # Update dual variables
        lambdas[t] = max(0,(1 - delta*mu**2)*lambdas[t] + mu*(np.dot(P[t, :], XX[:, t]) - B))        
        
    return XX

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
            new_position = v_prev[:, k] + (1/2/alpha)*(V*gradFJoke(t - 1, X_partial_prev[:, k]) - lambdas[k]*gradG(t - 1, v_prev[:, k]))
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

def BestOfBothOneGrad(T, V, alpha, K):
    """
    Best-of-both worlds algorithm with one gradient evaluation per step
    input: T (Horizon), V (relative-weights parameter), alpha (step-size), K (number of ) 
    output: sequence of x_t
    """
    Q = np.int(T/K)
    # Initialize the meta-FW vectors (v_{q - 1}'s) and (v_q's)
    v_prev = np.zeros((d, K))
    v_curr = np.zeros((d, K))
    # Initialize the dual variables
    lambdas = np.zeros(K)
    # partial decision vectors x_t(k)
    X_partial_prev = np.zeros((d, Q, K + 1))
    X_partial_curr = np.zeros((d, Q, K + 1))
    # Time permutation
    prev_t = np.array(range(K))
    # Store the decisions to return 
    XX = np.zeros((d, T))
    
    for q in range(1, Q):
        curr_t = np.random.permutation(range((q - 1)*K, q*K))
        X_partial_curr[:, q, 0] = np.zeros(d)
        
        gradGbar_q_minus_one_matrix = np.zeros((d, K))
        for k in range(K):
            # Evaluate gradG_bar_{q-1}(v_{q-1}^{k})
            gradGbar_q_minus_one = np.zeros(d)
            for  i in range(K):
                gradGbar_q_minus_one += gradG(prev_t[i], v_prev[:, k])
            gradGbar_q_minus_one = 1/K*gradGbar_q_minus_one
            gradGbar_q_minus_one_matrix[:, k] = gradGbar_q_minus_one
            
            new_position = v_prev[:, k] + (1/2/alpha)*(V*gradFJoke(prev_t[k], X_partial_prev[:, q - 1, k]) - lambdas[k]*gradGbar_q_minus_one)
            
            # Perform projection back to ground constraint set
            # Projection step 1
            for s in range(d):
                if new_position[s] < 0:
                    new_position[s] = 0
                elif new_position[s] > 1:
                    new_position[s] = 1
            v_curr[:, k] = new_position
            
            X_partial_curr[:, q, k + 1] = X_partial_curr[:, q, k] + 1/K*v_curr[:, k]
        
        for t in curr_t:
            XX[:, t] = roundVec(X_partial_curr[:, q, K])
        
        # Dual update
        for k in range(K):
            # Evaluate g_bar_{q-1}(v_{q-1}^{k})
            gbar_q_minus_one = 0
            for  i in range(K):
                gbar_q_minus_one += g(prev_t[i], v_prev[:, k]) - B
            gbar_q_minus_one = 1/K*gbar_q_minus_one
            
            lambdas[k] = max(0, lambdas[k] + gbar_q_minus_one + np.dot(gradGbar_q_minus_one_matrix[:, k], v_curr[:, k] - v_prev[:, k]))
        
        # Set current values to be the previous ones
        X_partial_prev = X_partial_curr
        v_prev = v_curr
        prev_t = curr_t
    # Return the history
    return XX

# Simple encapsulator functions for running an instance of each algorithm
def runOLFW():
    global X_historyOLFW, f_historyOLFW, g_historyOLFW, f_averageOLFW, g_averageOLFW
    # OLFW
    # Run OLFW Algorithm on the data
    X_historyOLFW = OLFW(T, eps)

    # Get the running cumulative utility data
    f_historyOLFW = np.zeros(T)
    f_historyOLFW[0] = fJoke(0, X_historyOLFW[:, 0])
    for t in range(1, T):
        f_historyOLFW[t] = f_historyOLFW[t-1] + fJoke(t, X_historyOLFW[:, t])

    # Get the running cumulative budget violation data
    g_historyOLFW = np.zeros(T)
    g_historyOLFW[0] = g(0, X_historyOLFW[:, 0]) 
    for t in range(1, T):
        g_historyOLFW[t] = g_historyOLFW[t-1] + np.dot(P[t, :], X_historyOLFW[:, t]) - B

    # Get the running average cumulative utility data
    f_averageOLFW = np.zeros(T)
    f_averageOLFW[0] = fJoke(0, X_historyOLFW[:, 0])
    for t in range(1, T):
        f_averageOLFW[t] = (f_averageOLFW[t - 1]*(t - 1) + fJoke(t, X_historyOLFW[:, t]))/t

    # Get the running average cumulative budget violation data
    g_averageOLFW = np.zeros(T)
    g_averageOLFW[0] = g(0, X_historyOLFW[:, 0]) 
    for t in range(1, T):
        g_averageOLFW[t] = (g_averageOLFW[t-1]*(t - 1) + np.dot(P[t, :], X_historyOLFW[:, t]) - B)/t

def runOSPHG():
    global X_historyOSPHG, f_historyOSPHG, g_historyOSPHG, f_averageOSPHG, g_averageOSPHG
    # OSPHG
    # Run OSPHG Algorithm on the data
    X_historyOSPHG = OSPHG(T)

    # Get the running cumulative utility data
    f_historyOSPHG = np.zeros(T)
    f_historyOSPHG[0] = fJoke(0, X_historyOSPHG[:, 0])
    for t in range(1, T):
        f_historyOSPHG[t] = f_historyOSPHG[t-1] + fJoke(t, X_historyOSPHG[:, t])

    # Get the running cumulative budget violation data
    g_historyOSPHG = np.zeros(T)
    g_historyOSPHG[0] = g(0, X_historyOSPHG[:, 0]) 
    for t in range(1, T):
        g_historyOSPHG[t] = g_historyOSPHG[t-1] + np.dot(P[t, :], X_historyOSPHG[:, t]) - B

    # Get the running average cumulative utility data
    f_averageOSPHG = np.zeros(T)
    f_averageOSPHG[0] = fJoke(0, X_historyOSPHG[:, 0])
    for t in range(1, T):
        f_averageOSPHG[t] = (f_averageOSPHG[t - 1]*(t - 1) + fJoke(t, X_historyOSPHG[:, t]))/t

    # Get the running average cumulative budget violation data
    g_averageOSPHG = np.zeros(T)
    g_averageOSPHG[0] = g(0, X_historyOSPHG[:, 0]) 
    for t in range(1, T):
        g_averageOSPHG[t] = (g_averageOSPHG[t-1]*(t - 1) + np.dot(P[t, :], X_historyOSPHG[:, t]) - B)/t

def runBestOfBoth():
    global alpha, X_historyBOB, f_historyBOB, g_historyBOB, f_averageBOB, g_averageBOB
    # Best-of-Both
    # Parameters
    K = np.int(T**0.5)
    V = np.int(T**0.4)
    alpha = T**1.1
    # Run BestOfBoth Algorithm on the data
    X_BOB = BestOfBoth(T, V, alpha, K)
    X_historyBOB = X_BOB
    # Get the running cumulative utility data
    f_historyBOB = np.zeros(T)
    f_historyBOB[0] = fJoke(0, X_historyBOB[:, 0])
    for t in range(1, T):
        f_historyBOB[t] = f_historyBOB[t-1] + fJoke(t, X_historyBOB[:, t])

    # Get the running cumulative budget violation data
    g_historyBOB = np.zeros(T)
    g_historyBOB[0] = g(0, X_historyBOB[:, 0]) 
    for t in range(1, T):
        g_historyBOB[t] = g_historyBOB[t-1] + np.dot(P[t, :], X_historyBOB[:, t]) - B

    # Get the running average cumulative utility data
    f_averageBOB = np.zeros(T)
    f_averageBOB[0] = fJoke(0, X_historyBOB[:, 0])
    for t in range(1, T):
        f_averageBOB[t] = (f_averageBOB[t - 1]*(t - 1) + fJoke(t, X_historyBOB[:, t]))/t

    # Get the running average cumulative budget violation data
    g_averageBOB = np.zeros(T)
    g_averageBOB[0] = g(0, X_historyBOB[:, 0]) 
    for t in range(1, T):
        g_averageBOB[t] = (g_averageBOB[t-1]*(t - 1) + np.dot(P[t, :], X_historyBOB[:, t]) - B)/t        

def runBestOfBothOneGrad():
    global X_historyBOBOneGrad, f_historyBOBOneGrad, g_historyBOBOneGrad, f_averageBOBOneGrad, g_averageBOBOneGrad
    # Best of Both with One gradient evaluation

    X_historyBOBOneGrad = BestOfBothOneGrad(T, np.int(T**(0.45)), alpha, np.int(T**(1/3)))
    # Get the running cumulative utility data
    f_historyBOBOneGrad = np.zeros(T)
    f_historyBOBOneGrad[0] = fJoke(0, X_historyBOBOneGrad[:, 0])
    for t in range(1, T):
        f_historyBOBOneGrad[t] = f_historyBOBOneGrad[t-1] + fJoke(t, X_historyBOBOneGrad[:, t])

    # Get the running cumulative budget violation data
    g_historyBOBOneGrad = np.zeros(T)
    g_historyBOBOneGrad[0] = g(0, X_historyBOBOneGrad[:, 0]) 
    for t in range(1, T):
        g_historyBOBOneGrad[t] = g_historyBOBOneGrad[t-1] + np.dot(P[t, :], X_historyBOBOneGrad[:, t]) - B

    # Get the running average cumulative utility data
    f_averageBOBOneGrad = np.zeros(T)
    f_averageBOBOneGrad[0] = fJoke(0, X_historyBOBOneGrad[:, 0])
    for t in range(1, T):
        f_averageBOBOneGrad[t] = (f_averageBOBOneGrad[t - 1]*(t - 1) + fJoke(t, X_historyBOBOneGrad[:, t]))/t

    # Get the running average cumulative budget violation data
    g_averageBOBOneGrad = np.zeros(T)
    g_averageBOBOneGrad[0] = g(0, X_historyBOBOneGrad[:, 0]) 
    for t in range(1, T):
        g_averageBOBOneGrad[t] = (g_averageBOBOneGrad[t-1]*(t - 1) + np.dot(P[t, :], X_historyBOBOneGrad[:, t]) - B)/t                             

################################ START PROBLEM ################################

# dimension of ambient space of decision variables 
# (# of different tasks)
d = 13
# Budget for each round B_T/T , we have that E[B_T/T] = 0.86
B = 0.86
# Horizon of online decision making (T) 
# (# of workers coming online)
T = 10000

def generateData():
    global M_matrix, scaling, p, P, u, beta, G
    # Rollout of objective function f_t(x) = sum(h_i(x_i)) + x^TM_tx
    M_matrix = np.random.uniform(-1/15, 0, (d, T*d))   
    # h_i(x_i) = alpha_i*log(1 + x_i), alpha_i = 1 + scaling[i]
    scaling = np.array(range(1, d + 1))

    # Rollout of constraint function <p_t,x> - B.  
    # Note: E[p_t] = p
    # Each row corresponds to one p_t 
    p = 1*np.array([0.01, 0.05, 0.01, 0.08, 0.02, 0.2, 0.05, 0.01, 0.5, 0.01, 0.01, 0.02, 0.25])

    P = np.zeros((T,d))
    for dim in range(d):
        delta_p = 0.5*p[dim]
        P[:, dim] = p[dim] + np.random.uniform(-delta_p, delta_p, T)    

    # Get the F, G, delta, beta values
    u = np.ones(d)
    temp=np.zeros(T)
    temp2=np.zeros(T)
    for t in range(T):
        M = fM(t)
        temp[t]=np.linalg.norm(-np.dot(M,u))
        temp2[t]=np.linalg.norm(P[t,:])

    beta=max(np.max(temp),np.max(temp2))
    G=np.linalg.norm(P,np.inf)-1

# Running the algorithms
# Running all the algorithms and taking the average
# N = the number of times to repeat the experiments
N = 10
f_cumulative = np.zeros((4, T))
g_cumulative = np.zeros((4, T))
f_average = np.zeros((4, T))
g_average = np.zeros((4, T))
for i in range(N):
    # Step 1: Generate a new random data
    generateData()
    # Step 2: Run all 4 algorithms
    runOLFW()
    runOSPHG()
    runBestOfBoth()
    runBestOfBothOneGrad()
    # Step 3: Store the data
    f_cumulative[0, :] += f_historyOLFW
    f_cumulative[1, :] += f_historyOSPHG
    f_cumulative[2, :] += f_historyBOB
    f_cumulative[3, :] += f_historyBOBOneGrad 
    g_cumulative[0, :] += g_historyOLFW
    g_cumulative[1, :] += g_historyOSPHG
    g_cumulative[2, :] += g_historyBOB
    g_cumulative[3, :] += g_historyBOBOneGrad 
    f_average[0, :] += f_averageOLFW
    f_average[1, :] += f_averageOSPHG
    f_average[2, :] += f_averageBOB
    f_average[3, :] += f_averageBOBOneGrad
    g_average[0, :] += g_averageOLFW
    g_average[1, :] += g_averageOSPHG
    g_average[2, :] += g_averageBOB
    g_average[3, :] += g_averageBOBOneGrad
f_cumulative = 1/N*f_cumulative
g_cumulative = 1/N*g_cumulative
f_average = 1/N*f_average
g_average = 1/N*g_average
    

# We plot f_average and g_average in the paper in Figure 1 (b) and compare all the algorithms   