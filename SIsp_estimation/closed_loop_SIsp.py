# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:43:15 2025

@author: liqunwei
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from math import floor
from sklearn.svm import OneClassSVM

from insulin_sensitivity import detect_meal#, estimate_SI
from scipy.stats import gamma

class system_model:
    def __init__(self,time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA):
        self.meal = True
        self.time_scale = time_scale
        self.N = N
        self.G_t = G_t
        self.G_t_past = G_t
        self.I_t_list = I_t_list
        self.n_I_delays = n_I_delays
        self.I_u_list = I_u_list
        self.n_Idose_delays = n_Idose_delays
        self.GAMMA = GAMMA
        self.pred_next_g = G_t
        self.calib = 0

        self.SIsp = None
        self.G_dose = 0
        
        self.BETA = 0
        self.Km = 2300
        # self.m = 60 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        # self.mb = 60 /self.time_scale # scaled by 5 to convert to 5 minute time increments
        # self.s = 0.0072 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        # self.Vmax = 100 *self.time_scale #150 *time_scale # scaled by 5 to convert to 5 minute time increments
        self.m = 60 # scaled by 1/5 to convert to 5 minute time increments
        self.mb = 60 # scaled by 5 to convert to 5 minute time increments
        self.s = 0.0072  # scaled by 5 to convert to 5 minute time increments
        self.Vmax = 150  #150 *time_scale # scaled by 5 to convert to 5 minute time increments

    

    def f1(self,I):
        # Rg = 180 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Rg = 180
        alpha = 0.29
        Vp = 3
        C5 = 26
        x = alpha*(I/Vp - C5)
        # y = x.copy()
        # y[x>=0] = Rg*np.exp(-x[x>=0]) / (np.exp(-x[x>=0]) + 1)
        # y[x<0] = Rg/(1+np.exp(x[x<0]))
        if x>=0:
            y = Rg*np.exp(-x) / (np.exp(-x) + 1)
        else:
            y = Rg/(1+np.exp(x))
        return y
    
    def f2(self,G):
        # Ub = 72 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Ub = 72
        C2 = 144
        Vg = 10
        return Ub*(1-np.exp(-G/(C2*Vg)))
    
    def f3(self,G):
        C3 = 1000
        Vg = 10
        return G/(C3*Vg)
    
    def f4(self,I):
        # U0 = 40 *self.time_scale #scaled by 5 to convert to 5 minute time increments
        U0 = 40
        # Um = 940 *self.time_scale #scaled by 5 to convert to 5 minute time increments
        Um = 940
        beta = 1.77
        C4 = 80
        Vi = 11
        # E = 0.2 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        E = 0.2
        # ti = 100 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        ti = 100
        return U0 + (Um-U0)/(1+np.exp(-beta*np.log(0.0000001+I/C4*(1/Vi+1/(E*ti)))));
    
    def f5(self,G):
        # Rm = 210 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Rm = 210
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
    
    def t_next(self,tau):
        #update current glucose level g_t
        self.calib = self.G_t - self.pred_next_g
        self.G_t_past = self.G_t
         # dI_t = self.iLoad(self.I_u_list)-0.06*self.I_t_list[0]*self.time_scale
         # [dI_t+self.I_t_list[0]]+self.I_t_list[:-1]
        #update injected insulin level
        GAMMA = (np.random.rand()-self.GAMMA)/self.GAMMA*0.2+self.GAMMA
        dI_t = self.iLoad(self.I_u_list)-self.time_scale*self.Vmax*self.I_t_list[0]/(self.Km+self.I_t_list[0])
        dG_t = self.f1( self.iTau(self.I_t_list))-self.f2(self.G_t)-GAMMA*(1+self.s*(self.m-self.mb))*self.f3(self.G_t)*self.f4(self.I_t_list[0])
        
        self.pred_next_g = self.G_t + self.f1( self.iTau(self.I_t_list))-self.f2(self.G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(self.G_t)*self.f4(self.I_t_list[0])
        
        self.I_t_list = np.insert(self.I_t_list, 0,  dI_t+self.I_t_list[0])[:-1]
        self.I_u_list = np.insert(self.I_u_list,0,tau[0])[:-1]
        
        self.G_t += dG_t * self.time_scale
        
    def predicted_g(self,tau,extended_len=0):
        G_N = np.zeros(self.N+extended_len)
        I_t_list = self.I_t_list
        G_t = self.G_t
        
        I_u_list = self.I_u_list
        for i in range(self.N):
            I_t = I_t_list[0]
            dG_t = self.f1( self.iTau(I_t_list))-self.f2(G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(G_t)*self.f4(I_t)
            
            # dI_t =I_load[n_Idose_delays-1]+BETA*f5(G_tau[n_G_delays-1],time_scale)-Vmax*I_t/(Km+I_t)
            dI_t = self.iLoad(I_u_list)-self.time_scale*self.Vmax*I_t/(self.Km+I_t)
            # dI_t = self.iLoad(I_u_list)-self.time_scale*0.06*I_t
            G_t = G_t + dG_t * self.time_scale
            I_t_list = np.insert(I_t_list, 0, dI_t+I_t_list[0])[:-1]
            
            I_u_list = np.insert(I_u_list, 0,tau[i])[:-1]
            
            G_N[i] = G_t + self.calib
        
        return G_N
    def discrete_tau(self,tau):
        return tau-np.array(tau)

    def estimate_SI(self, G_t, I_t, I_u, t_est, g):
        # parameters
        Gb = 100*g # 100 mg/dL or 5.6 mmol/L
        GEZI = 0.01 # deciliters per kilogram per minute; fixed to 0.01 dL/kg/min for diabetic ubjects (10,12)
        Vg = 1.45 # deciliters per kilogram; fixed to 1.45 dL/kg, according to Dalla Man et al. (3)
        BW = 79 # kg
        height = 1.77 # 1.77 m 5'8''
        age = 30 # years
        CL = 1.7 # liters/min 1.8 for male, 1.57 for female (Capmioni et al), 
        Vi = 9.31 # liters for all, 10.11 for male, 8.24 for female, 10.3 for obese (Campioni et al)

        G_t = G_t/g #convert from mg to mg/dL

        # varaibles 
        t_ingest, t_end, G_dose = detect_meal(G_t, t_est)

        # check for domain of validity
            # 1) CGM > 150 mg/dL 6 hr after meal ingestion
            # 2) CGM at meal time < 60 mg/dL or > 200 mg/dL
            # 3) |dCGM| at meal time > 2 mg/dL/min

        # check for possible estimation
            # 1) 90+ minutes since meal (180 to 360+ minutes has good accuracy)

        # sensitivitiy of insulin from sensor-pump data
        # def COB(G_dose, t_ingest, t_end):
        #     # cob = s
        #     cob = G_dose*(1-(t_end-t_ingest)/(360/timescale))
        #     # f_t = np.log(1+t_end-t_ingest)/np.log(t_end*1.1)
        #     return cob #(self.G_t-G_t[t_ingest])/1000

        def dCGM(cgm):
            return np.diff(cgm, n=1)/self.time_scale
            # return cgm - Gb*g

        if (t_ingest and t_end) and (t_est>t_ingest+1) :

            aoc = self.G_dose #COB(G_dose, t_ingest, t_end)
            s_dcgm = sum(dCGM(G_t[t_ingest:t_end+1]))
            delta_cgm = G_t[t_end]-G_t[t_ingest] 
            # delta_cgm = -15 + (G_t[t_end]-G_t[t_ingest])/4
            num =  (aoc/BW)-GEZI*s_dcgm*self.time_scale-Vg*(delta_cgm)

            s_It = sum(I_u[t_ingest:t_end]) #exclusive of t_end
            iob_0 = I_t[t_ingest-1]
            iob_1 = 0#I_t[t_end] #inclusive of t_end
            s_adcgm = sum(abs(dCGM(G_t[t_ingest:t_end+1])))
            denom = (s_It/Vi/CL + iob_0 - iob_1)*(s_adcgm/(t_end - t_ingest))
            
            self.SIsp = num / denom

            pack = {'aoc':aoc, 'sum dCGM': s_dcgm, 'delta CGM': delta_cgm, 'numerator':num,
                    'sum I': s_It, 'IOB start': iob_0, 'IOB end': iob_1, 'sum abs dCGM':s_adcgm,
                    'delta t': t_end-t_ingest, 'denominator': denom,
                    'SIsp': self.SIsp ,'t end':t_end}
        else:
            pack = {'aoc':None, 'sum dCGM': None, 'delta CGM': None, 'numerator':None,
                    'sum I': None, 'IOB start': None, 'IOB end': None, 'sum abs dCGM':None,
                    'delta t': None, 'denominator': None,
                    'SIsp': None ,'t end':None}

        return pack

# Step of the system

def l_cost(tau, system_model, g, target=105, safe_high=120, safe_low=90, alert_low=60, alert_high=150):
    gt = system_model.predicted_g(tau)
    l_safe_high = np.square(gt-target*g)*(gt>safe_high*g)
    l_safe_low = np.square(gt-target*g)*(gt<safe_low*g)
    l_alert_high = np.square(gt-target*g)*(gt>alert_high*g)
    l_alert_low = np.square(gt-target*g)*( gt<alert_low*g)
    return np.sum(l_safe_high+l_safe_low+l_alert_high+l_alert_low)

def max_cost(tau, system_model,g, G_t_max=140):
    G_t_N = system_model.predicted_g(tau)
    return np.max(np.square(np.array(G_t_N)-G_t_ref*g))

def mpc_cost(tau, system_model,g, G_t_ref=105):
    G_t_N = system_model.predicted_g(tau)
    return np.sum(np.square(np.array(G_t_N)-G_t_ref*g))/len(tau)

def m_cost(tau, system_model,g, G_t_ref=105, G_low=70):
    G_t_N = system_model.predicted_g(tau)-system.calib
    mse = np.mean(np.square(np.array(G_t_N)-G_t_ref*g))
    # max = np.max(np.square(np.array(G_t_N)-G_t_ref*g))
    # G_t_N_extended = system_model.predicted_g(tau,12)-system.calib
    # extended_max = np.max(np.square(np.array(G_t_N_extended)-G_t_ref*g))
    # extended_low = np.sum(np.square(np.array(G_t_N_extended)-G_t_ref*g)*(G_t_N_extended<G_low*g))
    safe_penalty = np.sum((G_t_N > 140*g) | (G_t_N < 70*g)) * 1e4
    return mse+safe_penalty

def solve_mpc(tau_ini,N,system_model,g,G_t_ref):

    # Linear constraints on the rate of change of tau: -delta_tau_max <= tau[i+1] - tau[i] <= delta_tau_max for all i in the N - 1
    # Implemented using LinearConstraint as -delta_tau_max <= delta_tau_matrix * tau <= delta_tau_max
    delta_tau_max = 500 #200
    tau_max = 1000 #300
    delta_tau_matrix = np.eye(N) - np.eye(N, k=1)
    constraint1 = LinearConstraint(delta_tau_matrix, -delta_tau_max, delta_tau_max)

    # # # We need a constraint on the rate of change of tau[0] respect to its previous value, which is tau_ini[0]
    first_element_matrix = np.zeros([N, N])
    first_element_matrix[0, 0] = 1
    constraint2 = LinearConstraint(first_element_matrix, 0, tau_ini[0]+delta_tau_max)
    # # G_t_constraint = predicted_g(N,tau,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA, time_scale)
    # system_model = system_model(time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA)
    G_t_constraint = NonlinearConstraint(system_model.predicted_g, 0.0, 200*g)
    # discrete_constraint = NonlinearConstraint(system_model.discrete_tau, 0.0, 0.0)

    # Add constraints
    delta_tau_constraint = [constraint1,constraint2,G_t_constraint]
    # delta_tau_constraint = [G_t_constraint]
    # Bounds --> -tau_max <= tau[idx] <= tau_max for idx = 0 to N-1
    bounds = [(0, tau_max) for idx in range(N)]

    # Starting optimisation point for theta and dtheta are the current measurements

    # Minimization
    result = minimize(m_cost, tau_ini, args=(system_model, g), bounds=bounds, constraints=delta_tau_constraint)

    # Extract the optimal control sequence
    tau_mpc = result.x

    return tau_mpc

# Cost function to be minimized


# ---------- SIMULATION INITIALISATION ----------
timescale = 5
HOURS = 12 #7am to 9pm
N_iterations = int(floor(HOURS*60/timescale))

# n_G_delays = int(floor(5/timescale)) removed for typeI
n_I_delays = int(floor(15/timescale))  #tau2
n_Idose_delays = int(floor(5/timescale)) #tau1

g=100
# Arrays for logging
tau = np.zeros(N_iterations)
# ---------- CONTROL SYSTEM CALIBRATION ----------
# Model predictive control horizon
N = 10
tau_ini = np.zeros(N)
# Array for values of theta and tau for each simulation

G_t_start = 100*g
I_t_record = np.ones(N)*50

I_u_record = np.ones(N)*50

GAMMA = 0.5
# ---------- SIMULATION ----------
system = system_model(timescale,N,G_t_start,I_t_record,n_I_delays,I_u_record,n_Idose_delays, GAMMA)
# Init time

G_t_ref=100
G_t = np.zeros(N_iterations)
G_predicted = np.zeros(N_iterations)
I_ob = np.zeros(N_iterations)
calib = np.zeros(N_iterations)
calib_anamoly =np.zeros(N_iterations)
moving_average = np.empty(0)
compensation = 0

bfast_start = int(floor(2*60/timescale)) #
bfast_end = int(floor(4.5*60/timescale))
x = np.arange(bfast_end-bfast_start)
alpha, beta = 10, 1
bfast = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)

lunch_start = int(floor(5*60/timescale))
lunch_end = int(floor(8*60/timescale))
x = np.arange(lunch_end-lunch_start)
alpha, beta = 10, 1
lunch = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)

dinner_start = int(floor(9*60/timescale))
dinner_end = int(floor(11.5*60/timescale))
x = np.arange(lunch_end-lunch_start)
alpha, beta = 10, 1
dinner = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)


model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# Define the window size for the moving average
window_size = 5
meal_compensation = 200
threshold = 500

compensation_times = 0
compensation_times_thrd = 2
compensation_gap = 0
compensation_gap_sleep = 12

SIsp = np.zeros(N_iterations)
pack = {}

for idx in range(N_iterations):

    tau_mpc = solve_mpc(tau_ini,N,system,g,G_t_ref)
    # Use first element of control input optimal solution
    
    # Initial solution for next step = current solution
    
    if idx in (bfast_start, lunch_start, dinner_start):
        tau_mpc[0] += compensation    
    
    #manual manipulation
    # if idx > 10 and idx -window_size <= 10:
    moving_average = np.append(moving_average, system.calib)
    if idx -window_size > 10:
            
            if compensation_times < compensation_times_thrd:
                if system.calib - np.mean(moving_average) > threshold:

                    calib_anamoly[idx] = 1
                    tau_mpc[0] += meal_compensation
                    compensation_times += 1
            elif compensation_times > 0:
                if compensation_gap < compensation_gap_sleep:
                    compensation_gap += 1
                else:
                    compensation_times = 0
                    compensation_gap = 0
    
    tau[idx] = tau_mpc[0]
    # ---------- SIMULATION LOOP  ----------
    G_predicted[idx] = system.predicted_g(tau_mpc)[0]
    system.t_next(tau_mpc)
    
    tau_ini = tau_mpc
    if system.meal:
        # system.G_t += 540 #540 mg = 1.08mg/dL/min constant infusion
        if (idx>=bfast_start) & (idx<bfast_end):
            system.G_t += bfast[idx-bfast_start]*85_000#(1/(bfast_end-bfast_start))*50_000 #50*g # 300 mg/dl of glucose infused of 15 mins
            system.G_dose = bfast[idx-bfast_start]*85_000
        if (idx>=lunch_start) & (idx<lunch_end): 
            system.G_t += lunch[idx-lunch_start]*80_000#(1/(lunch_end-lunch_start))*40_000 #100 *g # 
        if (idx>=dinner_start) & (idx<dinner_end): 
            system.G_t += dinner[idx-dinner_start]*95_000#(1/(dinner_end-dinner_start))*50_000 #150 *g
    G_t[idx] = system.G_t
    I_ob[idx] = system.I_t_list[0]
    
    calib[idx] = system.calib

    if system.meal and idx>0:
        pack[idx] = system.estimate_SI(G_t=G_t[:idx+1],
                                       I_t=I_ob[:idx+1], 
                                       I_u=tau[:idx+1],
                                       t_est=idx, 
                                       g=g)
        SIsp[idx] = pack[idx]['SIsp']

print(sum(tau))
    
# Append result for this simulation
plt.subplot(3, 1, 1)
plt.plot(np.arange(N_iterations)*timescale/60, np.array(G_t)/g, '--', label="Glucose Level")
plt.plot(np.arange(N_iterations)*timescale/60, np.array(G_predicted)/g, '--', label="Predicted Glucose Level")
plt.ylabel(r"G_t, mg/dL")
plt.legend()
plt.gca().set_ylim(bottom=-1)
plt.axhline(70,color='green',linestyle='--',linewidth=0.75)
plt.axhline(140,color='green',linestyle='--',linewidth=0.75)
# plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    
    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    

plt.subplot(3, 1, 2)
plt.plot(np.arange(N_iterations)*timescale/60, tau, linewidth=0,color='r',marker='.',label="Insulin Infusion")
plt.ylabel(r"I_u, mU")
plt.xlabel("Time")
plt.legend()
plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    
    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.text(bfast_start*timescale/60+0.1, 110, 'Breakfast', rotation=90)
    plt.text(lunch_start*timescale/60+0.1, 110, 'Lunch', rotation=90)
    plt.text(dinner_start*timescale/60+0.1, 110, 'Dinner', rotation=90)

plt.subplot(3, 1, 3)
num = [pack[i]['numerator'] for i in pack.keys()]
denom = [pack[i]['denominator'] for i in pack.keys()]
aoc = [pack[i]['aoc'] for i in pack.keys()]
s_dcgm = [pack[i]['sum dCGM'] for i in pack.keys()]
delta_cgm = [pack[i]['delta CGM'] for i in pack.keys()]

plt.scatter(np.arange(N_iterations)*timescale/60, SIsp, marker='')
plt.plot(pack.keys(), num, linewidth=0.8,label='numerator')
plt.plot(pack.keys(), denom, linewidth=0.8,label='denominator')
plt.plot(pack.keys(), aoc, linewidth=0.8,label='aoc')
plt.plot(pack.keys(), s_dcgm, linewidth=0.8,label='s_dcgm')
plt.plot(pack.keys(), delta_cgm, linewidth=0.8,label='delta_cgm')
plt.plot(np.arange(N_iterations)*timescale/60, SIsp/0.006, linewidth=0.8,color='grey',label="Insulin Sensitivity")
plt.axhline(GAMMA,linewidth=0.75,color='green',linestyle='--')
plt.ylabel(r"SIsp, %")
plt.xlabel("Time")
plt.legend()
# plt.ylim([-0.5,1.5])
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    
    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.text(bfast_start*timescale/60+0.1, 110, 'Breakfast', rotation=90)
    plt.text(lunch_start*timescale/60+0.1, 110, 'Lunch', rotation=90)
    plt.text(dinner_start*timescale/60+0.1, 110, 'Dinner', rotation=90)
plt.show()



