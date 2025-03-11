# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:43:15 2025

@author: liqunwei
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt

class system_model:
    def __init__(self,time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA):
        self.time_scale = time_scale
        self.N = N
        self.G_t = G_t
        self.I_t_list = I_t_list
        self.n_I_delays = n_I_delays
        self.I_u_list = I_u_list
        self.n_Idose_delays = n_Idose_delays
        self.GAMMA = GAMMA
        
        self.BETA = 0
        self.Km = 2300
        self.m = 60 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        self.mb = 60 /self.time_scale # scaled by 5 to convert to 5 minute time increments
        self.s = 0.0072 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        self.Vmax = 100 *self.time_scale #150 *time_scale # scaled by 5 to convert to 5 minute time increments

    

    def f1(self,I):
    	Rg = 180 *self.time_scale # scaled by 5 to convert to 5 minute time increments
    	alpha = 0.29
    	Vp = 3
    	C5 = 26
    	return Rg/(1+np.exp(alpha*(I/Vp - C5)))
    
    def f2(self,G):
    	Ub = 72 *self.time_scale # scaled by 5 to convert to 5 minute time increments
    	C2 = 144
    	Vg = 10
    	return Ub*(1-np.exp(-G/(C2*Vg)))
    	
    def f3(self,G):
    	C3 = 1000
    	Vg = 10
    	return G/(C3*Vg)
    
    def f4(self,I):
    	U0 = 40 *self.time_scale #scaled by 5 to convert to 5 minute time increments
    	Um = 940 *self.time_scale #scaled by 5 to convert to 5 minute time increments
    	beta = 1.77
    	C4 = 80
    	Vi = 11
    	E = 0.2 *self.time_scale # scaled by 5 to convert to 5 minute time increments
    	ti = 100 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
    	return U0 + (Um-U0)/(1+np.exp(-beta*np.log(0.0000001+I/C4*(1/Vi+1/(E*ti)))));
    
    def f5(self,G):
    	Rm = 210 *self.time_scale # scaled by 5 to convert to 5 minute time increments
    	C1 = 2000
    	Vg = 10
    	a1 = 300
    	return Rm/(1+np.exp((C1-G/Vg)/a1))
    
    # iLoad is the effective infused insulin at the moment(infusion could take time)
    def iLoad(self,I_u_list):
        return I_u_list[self.n_Idose_delays-1]
    
    # iTau is the previous inlusion dose in the body that makes liver produce glucose(Tau: the process takes time)
    def iTau(self,I_t_list):
        return I_t_list[self.n_I_delays-1]
    
    def t_next(self,tau_next):
        #update current glucose level g_t
        dG_t = self.f1( self.iTau(self.I_t_list))-self.f2(self.G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(self.G_t)*self.f4(self.I_t_list[0])
        self.G_t += dG_t
        #update insulin level list
        dI_t = self.iLoad(self.I_u_list)-self.Vmax*self.I_t_list[0]/(self.Km+self.I_t_list[0])
        self.I_t_list = np.insert(self.I_t_list, 0, dI_t+self.I_t_list[0])[:-1]
        # [dI_t+self.I_t_list[0]]+self.I_t_list[:-1]
        #update injected insulin level
        self.I_u_list = np.insert(self.I_u_list,0,tau_next)[:-1]
        
    def predicted_g(self,tau):
        G_N = []
        I_t_list = self.I_t_list
        G_t = self.G_t
        I_u_list = self.I_u_list
        for i in range(self.N):
            I_t = I_t_list[0]
            dG_t = self.f1( self.iTau(I_t_list))-self.f2(G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(G_t)*self.f4(I_t)
            G_t = G_t + dG_t
            # dI_t =I_load[n_Idose_delays-1]+BETA*f5(G_tau[n_G_delays-1],time_scale)-Vmax*I_t/(Km+I_t)
            dI_t = self.iLoad(I_u_list)-self.Vmax*I_t/(self.Km+I_t)
            
            I_t_list = np.insert(I_t_list, 0, dI_t+I_t_list[0])[:-1]
            
            I_u_list = np.insert(I_u_list, 0,tau[i])[:-1]
            
            G_N.append(G_t)
        return G_N
# Step of the system

def l_loss(gt, g, target=100, safe_high=140, safe_low=70, alert_low=60, alert_high=150):
    l_safe_high = (gt-target*g)**2*(gt>safe_high*g)
    l_safe_low = (gt-target*g)**2*(gt<safe_low*g)
    l_alert_high = (gt-target*g)**2*(gt>alert_high*g)
    l_alert_low = (gt-target*g)**2*( gt<alert_low*g)
    return l_safe_high+l_safe_low+l_alert_high+l_alert_low


def solve_mpc(tau_ini,N,system_model,g,G_t_ref):

    # Linear constraints on the rate of change of tau: -delta_tau_max <= tau[i+1] - tau[i] <= delta_tau_max for all i in the N - 1
    # Implemented using LinearConstraint as -delta_tau_max <= delta_tau_matrix * tau <= delta_tau_max
    delta_tau_max = 100
    tau_max = 100
    delta_tau_matrix = np.eye(N) - np.eye(N, k=1)
    constraint1 = LinearConstraint(delta_tau_matrix, 0, delta_tau_max)

    # # We need a constraint on the rate of change of tau[0] respect to its previous value, which is tau_ini[0]
    first_element_matrix = np.zeros([N, N])
    first_element_matrix[0, 0] = 1
    constraint2 = LinearConstraint(first_element_matrix, 0, tau_ini[0]+delta_tau_max)
    # # G_t_constraint = predicted_g(N,tau,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA, time_scale)
    # system_model = system_model(time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA)
    G_t_constraint = NonlinearConstraint(system_model.predicted_g, 0.0, 200*g)
    # Add constraints
    delta_tau_constraint = [constraint1, constraint2,G_t_constraint]
    # delta_tau_constraint = [G_t_constraint]
    # Bounds --> -tau_max <= tau[idx] <= tau_max for idx = 0 to N-1
    bounds = [(0, tau_max) for idx in range(N)]

    # Starting optimisation point for theta and dtheta are the current measurements

    # Minimization
    result = minimize(mpc_cost, tau_ini, args=(tau_ini, system_model, N, G_t_ref*g), bounds=bounds, constraints=delta_tau_constraint)

    # Extract the optimal control sequence
    tau_mpc = result.x

    return tau_mpc

# Cost function to be minimized
def mpc_cost(tau, tau_ini, system_model, N, G_t_ref):

    # Initialise cost = 0 and states to current measured states
    # cost = 0
    G_t_N = system_model.predicted_g(tau)
    return np.sum(np.square(np.array(G_t_N)-G_t_ref))


# ---------- SIMULATION INITIALISATION ----------

# Simulation time
time_range = 10
# Simulation steps
L = round(100)
# Init time
g=3
# Arrays for logging
tau = np.zeros(L)
# ---------- CONTROL SYSTEM CALIBRATION ----------
# Model predictive control horizon
N = 20

tau_ini = np.zeros(N)
# Array for values of theta and tau for each simulation
time_scale = 5
G_t = 200
I_t_list = np.zeros(N)
n_I_delays = 10
I_u_list = np.zeros(N)
n_Idose_delays = 5
GAMMA = 0.5
# ---------- SIMULATION ----------
system = system_model(time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA)
# Init time

G_t_ref=100
G_t = []
for idx in range(L):
    tau_mpc = solve_mpc(tau_ini,N,system,g,G_t_ref)
    # Use first element of control input optimal solution
    tau[idx] = tau_mpc[0]
    
    # Initial solution for next step = current solution
    tau_ini = tau_mpc
    
    # ---------- SIMULATION LOOP  ----------
    system.t_next(tau_mpc[0])
    G_t.append(system.G_t)
# Append result for this simulation
print(G_t)
print(tau)

plt.plot(range(L),G_t)
plt.plot(range(L),tau)
plt.show()

