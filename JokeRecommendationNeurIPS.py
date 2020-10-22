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
	This returns Mt matrix in the definition of Jester objective (f_t(x) = R_t^Tx + x^TM_tx)
	"""
	M = np.zeros((d,d))
	for i in range(nBlocks):
		theta_min = np.amin(H[t, i*w:(i+1)*w])
		theta = theta_min/(w**2-w)
		M[i*(w):(i+1)*w, i*w:(i+1)*w] = theta*(np.eye(w) - np.ones((w,w)))
	return M

def fJoke(t, x):
	"""
	This is the objective function oracle.
	input: time t and decision vector x (and function rollout history H)
	output: f_t(x_t)
	"""
	M = fM(t)
	out=np.dot(x.T,np.dot(M,x)) + np.dot(x.T, H[t,:])
	return out

def gradFJoke(t, x):
	"""
	This is the objective function's gradient oracle.
	input: time t and decision vector x (and function rollout history H)
	output: grad(f_t(x_t))
	"""
	out=0
	M = fM(t)
	# print(Ht.shape)
	# print(u.shape)
	out=H[t,:] + np.dot(M+M.T,x)
	return out

def g(t, x):
	"""
	This is the constraint function oracle.
	input: time t and decision vector x (and function rollout history P)
	output: g_t(x_t) = <p_t, x_t> - B
	"""	
	return np.dot(P[t,:], x) 

def roundVec(X):
    """
    Rounds the top KB entries to one
    """
    ind = np.argpartition(X, -KB)[-KB:]
    out = np.zeros(d)
    np.put(out,ind,np.ones(ind.size))
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
		# Projection step 2
		val = np.dot(np.ones(d),self.v)
		if val>KB:
			self.v = self.v -(val-KB)/d*np.ones(d)

def BestOfBoth(T, eps):
    """
    Best-of-both worlds algorithm
    input: T (Horizon), eps (parameter in W = T^{1 - eps}) 
    output: sequence of x_t
    """	
    V = np.int(T**(1 - eps/4))
    K = np.int(T**(1/2))
    alpha = V*(T**0.5)
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
            # Projection step 2
            val = np.dot(np.ones(d), new_position)
            if val > KB:
                new_position = new_position -(val-KB)/d*np.ones(d)
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

def fM(t):
	"""
	This returns Mt matrix in the definition of Jester objective (f_t(x) = R_t^Tx + x^TM_tx)
	"""
	M = np.zeros((d,d))
	for i in range(nBlocks):
		theta_min = np.amin(H[t, i*w:(i+1)*w])
		theta = theta_min/(w**2-w)
		M[i*(w):(i+1)*w, i*w:(i+1)*w] = theta*(np.eye(w) - np.ones((w,w)))
	return M

def fJoke(t, x):
	"""
	This is the objective function oracle.
	input: time t and decision vector x (and function rollout history H)
	output: f_t(x_t)
	"""
	M = fM(t)
	out=np.dot(x.T,np.dot(M,x)) + np.dot(x.T, H[t,:])
	return out

def gradFJoke(t, x):
	"""
	This is the objective function's gradient oracle.
	input: time t and decision vector x (and function rollout history H)
	output: grad(f_t(x_t))
	"""
	out=0
	M = fM(t)
	# print(Ht.shape)
	# print(u.shape)
	out=H[t,:] + np.dot(M+M.T,x)
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
    Rounds the top KB entries to one
    """
    ind = np.argpartition(X, -KB)[-KB:]
    out = np.zeros(d)
    np.put(out,ind,np.ones(ind.size))
    return out

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
            # Projection step 2
            val = np.dot(np.ones(d), new_position)
            if val > KB:
                new_position = new_position -(val-KB)/d*np.ones(d)
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

def OSPHG(T, W):
    """
    OSPHG implementation
    input: T (Horizon), W (Window size) 
    output: sequence of x_t
    """	
    # Number of times each oracle is called 
    K = np.int(T**0.5)
    # Step size of Online Gradient Ascent
    mu = 1.0 / (W*T)**0.5
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

# Trade-off experiments for different Window sizes
eps_vec = [0.05, 0.25, 0.5, 0.75, 1]
num = 5
final_utilityOSPHG = np.zeros(num)
final_utilityBOB = np.zeros(num)
final_budgetViolationOSPHG = np.zeros(num)
final_budgetViolationBOB = np.zeros(num)
for i in range(5):
    print(i)
    eps = eps_vec[i]
    W = np.int(T**(1 - eps))
    # Run the algorithms
    X_historyOSPHG = OSPHG(T, W)
    X_historyBOB = BestOfBoth(T, eps)
    # Calculate the utility and budget violation
    #    For OSPHG
    f_historyOSPHG = fJoke(0, X_historyOSPHG[:, 0])
    for t in range(1, T):
        f_historyOSPHG = f_historyOSPHG + fJoke(t, X_historyOSPHG[:, t])
    final_utilityOSPHG[i] = f_historyOSPHG
    
    g_historyOSPHG = g(0, X_historyOSPHG[:, 0]) 
    for t in range(1, T):
        g_historyOSPHG = g_historyOSPHG + np.dot(P[t, :], X_historyOSPHG[:, t]) - B    
    final_budgetViolationOSPHG[i] = g_historyOSPHG
    
    #    For Best-of-Both
    f_historyBOB = fJoke(0, X_historyBOB[:, 0])
    for t in range(1, T):
        f_historyBOB = f_historyBOB + fJoke(t, X_historyBOB[:, t])
    final_utilityBOB[i] = f_historyBOB
    
    g_historyBOB = g(0, X_historyBOB[:, 0]) 
    for t in range(1, T):
        g_historyBOB = g_historyBOB + np.dot(P[t, :], X_historyBOB[:, t]) - B            
    final_budgetViolationBOB[i] = g_historyBOB

# We plot final_utilityOSPHG and final_utlityBOB one subplot, and final_budgetViolationOSPHG and final_budgetViolationBOB on the other in Figure 1 (a) in the paper.                	