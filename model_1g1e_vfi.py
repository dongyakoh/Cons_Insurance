# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:02:55 2023

@author: dkoh
"""

import numpy as np
from scipy.optimize import fminbound,fmin
import sys
import matplotlib.pyplot as plt  


# Define household class
class household:
    
    # Initialize a household object.
    # Provide predetermined parameter values
    def __init__(self, σ=2, 𝛽=0.94, Na=500, T=6):
        
        # Store the characteristics of household in the class object
        self.σ          = σ                   # Coefficient of RRA
        self.𝛽          = 𝛽                   # Discount factor
        self.r          = 0.04
        self.w          = 5
        self.b          = 1
        self.Na         = Na                  # Number of grid points for a0 state
        self.Ne         = 2
        self.T          = T                  # 
        self.a_min      = 0
        self.a_max      = 10
        self.a0_state   = np.linspace(self.a_min,self.a_max,self.Na) # Discretize a0 state
        self.e0_state   = np.asarray([0, 1]) # Construct e0 state as ndarray type instead of list type
        self.Π          = np.asarray([[0.5,0.5],[0.5,0.5]])   # Stochastic matrix, Prob(e1|e0), as ndarray type
        self.Vf         = np.zeros((self.Na,self.Ne,T))   # Value function, a1
        self.ap         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1


    def get_Vf(self,age):
        '''
        This function updates the value function
        '''
        return self.Vf[:,:,age]
    
    # Update the policy function, a1
    def set_Vf(self, V0, age):
        self.Vf[:,:,age] = V0
    
    def get_ap(self,age):
        '''
        This function updates the value function
        '''
        return self.ap[:,:,age]

    # Update the policy function, a1
    def set_ap(self, a1, age):
        self.ap[:,:,age] = a1
        
        
        
#-----------------------------------------
#   Several functions
#-----------------------------------------
def uu(cons,σ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    if σ == 1.0:
        return np.log(cons)
    else:
        return cons**(1-σ)/(1-σ)
        
def bellman(ap,hh,a0,e0,VV,P):
    '''
    This function computes bellman equation for a given state (a0,e0).
    Input:
        ap: evaluating point
        hh: household object
        (a0,e0) state
        V1: value function at age t+1 evaluated at (a',e')
        P1: probability distribution of e' conditional on the current e0
    Output:
        -vv: bellman equation
    ''' 

    a0_state   = hh.a0_state
    r,w,b,β,σ  = hh.r,hh.w,hh.b,hh.β,hh.σ
    

    # Interpolate next period's value function evaluated at (a',e')
    # using 1-dimensional interpolation function in numpy
    V0      = np.interp(ap,a0_state,VV[:,0])
    V1      = np.interp(ap,a0_state,VV[:,1])
    EV      = P[0]*V0 + P[1]*V1
        
    # Interpolated value cannot be NaN or Inf
    if np.isnan(V0) or np.isinf(V0): print("bellman: V0 is NaN.")
    if np.isnan(V1) or np.isinf(V1): print("bellman: V1 is NaN.")

    # Compute consumption at a given (a0,e0) and a'       
    cons = w*e0 + b*(1-e0) + (1 + r)*a0 - ap
    
    # Consumption must be non-negative
    if cons<=0:
        return 1e7
    else:
        # Compute value function
        return -(uu(cons,σ) + β*EV)
    



#-----------------------------------------
#   Run the program
#-----------------------------------------
              
# Create a household instance
hh     = household()

r,w,b,Π,σ = hh.r,hh.w,hh.b,hh.Π,hh.σ
Na        = hh.Na
Ne        = hh.Ne
a0_state  = hh.a0_state
e0_state  = hh.e0_state


# Backward induction: solve an individual's problem from the last period
#for age in reversed(range(hh.T)):
for age in reversed(range(hh.T)):
    sys.stdout.write('age = %d.\n' % age)
    
    # Value function at age t+1
    V0    = np.zeros((Na,Ne))
    a1    = np.zeros((Na,Ne))
    c0    = np.zeros((Na,Ne))

    # Iterate for each state (a0,e0)
    # enumerate() gives an index and value of the list
    for ind in range(Na*Ne):
        
        ia  = ind // Ne
        ie  = ind % Ne
        
        a0 = a0_state[ia]
        e0 = e0_state[ie]

        if age==hh.T-1:   # Find an optimal saving for t=T
            a1[ia,ie] = 0.0
            c0[ia,ie] = w*e0 + b*(1-e0) + (1+r)*a0 - a1[ia,ie]
            V0[ia,ie] = uu(c0[ia,ie],σ)
    
        else:   # Find an optimal saving by using a root-finding method for t<T
            V1      = hh.get_Vf(age+1)
            P       = Π[ie,:]
            amin    = 0
            amax    = w*e0 + b*(1-e0) + (1+r)*a0 - 1e-2
            a1[ia,ie] = fminbound(bellman,amin,amax,args=(hh,a0,e0,V1,P))
            c0[ia,ie] = w*e0 + b*(1-e0) + (1+r)*a0 - a1[ia,ie]
            V0[ia,ie] = -bellman(a1[ia,ie],hh,a0,e0,V1,P)

            
    # Store the policy function in the class object
    hh.set_Vf(V0,age)
    hh.set_ap(a1,age)


 #-----------------------------------------
 #   Plot value/policy functions
 #-----------------------------------------

    # Plot Value Function
    fig, ax = plt.subplots(figsize=(5, 3))
# =============================================================================
#     plt.plot(a0_state, V0[:,0],label="bad shock")
#     plt.plot(a0_state, V0[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Value function V' ")
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================
    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(a0_state, a1[:,0],label="bad shock")
    plt.plot(a0_state, a1[:,1],label="good shock")
    plt.plot(a0_state, a0_state,'k--',label="$45^{\circ}$ line")
    plt.xlabel("State space, a")
    plt.ylabel("Policy function a' ")
    plt.legend(loc='upper left', fontsize = 10)
    plt.show()


# =============================================================================
#     plt.plot(a0_state, c0[:,0],label="bad shock")
#     plt.plot(a0_state, c0[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function c ")
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================


