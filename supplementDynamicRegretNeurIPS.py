import numpy as np
import cvxpy as cp

############################################## Problem Paratmeters ##############################################

# dimension of ambient space of decision variables 
# (# of different tasks)
d = 13
# Budget for each round B_T/T , we have that E[B_T/T] = 0.86
B = 0.86
# Horizon of online decision making (T) 
# (# of workers coming online)
T = 10000

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

############################################## Run the Offline algorithm ##############################################
# Generate the data
generateData()

# Start the offline meta-Frank-Wolfe

max_power = 13
T_final = 2**max_power
pow_vec = [i for i in range(max_power + 1)]
utility_vec = np.zeros(max_power + 1)
counter = 0
for power in range(max_power + 1):
    num_shifts = 2**power
    time_width = np.int(T_final/num_shifts)
    for shift in range(num_shifts):
        time_vec = shift*time_width + np.arange(0, time_width)
        K = np.int(T**0.5)
        X = np.zeros((d, K+1))
        for k in range(K):
            x=cp.Variable(d)
            u_time_width = np.ones(time_width)
            u = np.ones(d)
            currGradF = np.zeros(d)
            for time in time_vec:
                currGradF += gradFJoke(time, X[:, k])

            currP_sum = np.zeros(d)
            for time in time_vec:
                currP_sum += P[time, :]
                
            objective=cp.Maximize(cp.matmul(currGradF.T, x))
            constraints=[cp.matmul(currP_sum, x)<=B*time_width, 0<=x, x<=u]
            prob=cp.Problem(objective,constraints)
            prob.solve(solver=cp.ECOS)
            v = x.value
            X[:,k+1]=X[:,k]+v/K
        
        for t in time_vec:
            utility_vec[counter] += fJoke(t, X[:, K])  
    counter += 1

# We plot the benchmark values of the dynamic regret in the supplement   