# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:02:55 2023

@author: dkoh
"""

import numpy as np
from scipy.optimize import fminbound,fmin
import sys
import matplotlib.pyplot as plt  
import scipy.optimize as opt


# Define household class
class household:
    
    # Initialize a household object.
    # Provide predetermined parameter values
    def __init__(self):
        
        # Store the characteristics of household in the class object
        self.γ          = 0.578                   # Coefficient of RRA
        self.𝛽          = 0.95                  # Discount factor
        self.σ          = 0.85
        self.ψ          = 1.95
        self.f          = 0.0306
        self.r          = 0.02
        self.w1         = 1
        self.w2         = 0.8
        self.b          = 0.37
        self.Na         = 500                  # Number of grid points for a0 state
        self.Ne         = 2
        self.T          = 6                  # 
        self.a_min      = 0
        self.a_max      = 10
        self.a0_state   = np.linspace(self.a_min,self.a_max,self.Na) # Discretize a0 state
        self.e0_state   = np.asarray([0, 1]) # Construct e0 state as ndarray type instead of list type
        self.Π          = np.asarray([[0.9,0.1],[0.1,0.9]])   # Stochastic matrix, Prob(e1|e0), as ndarray type
        self.VF         = np.zeros((self.Na,self.Ne,self.T))   # Value function, a1
        self.dVF        = np.zeros((self.Na,self.Ne,self.T))   # Value function, a1
        self.ap         = np.zeros((self.Na,self.Ne,self.T))   # Policy function, a1
        self.cp         = np.zeros((self.Na,self.Ne,self.T))   # Policy function, a1
        self.hp         = np.zeros((self.Na,self.Ne,self.T))   # Policy function, a1


    def get_VF(self,age):
        '''
        This function retrieves the current value function
        '''
        return self.VF[:,:,age]

    def set_VF(self, VF, age):
        '''
        This function sets a new value function
        '''
        self.VF[:,:,age] = VF

    def get_dVF(self,age):
        '''
        This function retrieves the current value function
        '''
        return self.dVF[:,:,age]
    def set_dVF(self, dVF, age):
        '''
        This function sets a new value function
        '''
        self.dVF[:,:,age] = dVF
    
    # Update the policy function, a1
    def set_ap(self, a1, age):
        '''
        This function sets a new policy function
        '''
        self.ap[:,:,age] = a1
        
    # Update the policy function, a1
    def set_cp(self, c0, age):
        '''
        This function sets a new policy function
        '''
        self.cp[:,:,age] = c0

    # Update the policy function, a1
    def set_hp(self, h2, age):
        '''
        This function sets a new policy function
        '''
        self.hp[:,:,age] = h2
        
        
#-----------------------------------------
#   Several functions
#-----------------------------------------
def uu(c0,h2,γ,ψ,σ,f):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    h2_temp = h2 + np.equal(h2,0.0)*0.1
    return c0**(1-1/γ)/(1-1/γ) - np.not_equal(h2,0.0)*(ψ*h2_temp**(1+1/σ)/(1+1/σ) + f)
    

def duc(c0,γ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return c0**(-1/γ)

def duh(h2,ψ,σ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return -ψ*h2**(1/σ)
        



#-----------------------------------------
#   Run the program
#-----------------------------------------
              
# Create a household instance
hh     = household()

r,w1,w2,b,Π,γ,σ,β,ψ,f = hh.r,hh.w1,hh.w2,hh.b,hh.Π,hh.γ,hh.σ,hh.β,hh.ψ,hh.f
Na        = hh.Na
Ne        = hh.Ne
a0_state  = hh.a0_state
a1_state  = hh.a0_state
e0_state  = hh.e0_state


# Backward induction: solve an individual's problem from the last period
for age in reversed(range(hh.T)):
#for age in [hh.T-1]:
    sys.stdout.write('age = %d.\n' % age)
    
    # Value function at age t+1
    VF    = np.zeros((Na,Ne))
    dVF   = np.zeros((Na,Ne))
    a1    = np.zeros((Na,Ne))
    c0    = np.zeros((Na,Ne))
    h2    = np.zeros((Na,Ne))

    # Iterate for each state (a0,e0)
    # enumerate() gives an index and value of the list
    if age==hh.T-1:   # Find an optimal saving for t=T
        for ind in range(Na*Ne):
            ia  = ind // Ne
            ie  = ind % Ne
            a0_st = a0_state[ia]
            e0_st = e0_state[ie]
            a1[ia,ie] = 0.0
            
            # Case1: h2>0
            income_t = w1*e0_st + b*(1-e0_st) + (1+r)*a0_st
            _f = lambda cc: cc - w2*(w2/ψ * cc**(-1/γ))**σ - income_t
            c0_temp1 = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            h2_temp1  = (w2/ψ * c0_temp1**(-1/γ))**σ
            VF_temp1  = uu(c0_temp1,h2_temp1,γ,ψ,σ,f)

            # Case2: h2=0
            h2_temp2  = 0.0
            c0_temp2  = w1*e0_st + b*(1-e0_st) + (1+r)*a0_st
            VF_temp2  = uu(c0_temp2,h2_temp2,γ,ψ,σ,f)

            # Choose either case 1 or 2 whichever gives a higher value function
            h2[ia,ie] = h2_temp1*(VF_temp1>=VF_temp2) + h2_temp2*(VF_temp2>VF_temp1)
            c0[ia,ie] = c0_temp1*(VF_temp1>=VF_temp2) + c0_temp2*(VF_temp2>VF_temp1)
            VF[ia,ie] = VF_temp1*(VF_temp1>=VF_temp2) + VF_temp2*(VF_temp2>VF_temp1)
    
    else:   # Find an optimal saving by using a root-finding method for t<T
        a0_temp1     = np.zeros((Na,Ne))
        c0_temp1     = np.zeros((Na,Ne))
        h2_temp1     = np.zeros((Na,Ne))
        a0_temp2     = np.zeros((Na,Ne))
        c0_temp2     = np.zeros((Na,Ne))
        h2_temp2     = np.zeros((Na,Ne))
        
        dVFt1       = hh.get_dVF(age+1)
        for ind in range(Na*Ne):
            ia      = ind // Ne
            ie      = ind % Ne
            a1_st   = a1_state[ia]
            e0_st   = e0_state[ie]
            dVFt1_0    = np.interp(a1_st,a1_state,dVFt1[:,0])
            dVFt1_1    = np.interp(a1_st,a1_state,dVFt1[:,1])
            dEVF    = Π[ie,0]*dVFt1_0 + Π[ie,1]*dVFt1_1
            if np.isnan(dEVF) or np.isinf(dEVF) or dEVF<=0: print("bellman: dEVF=%f" % dEVF)

            # Case1: h2>0
            c0_temp1[ia,ie] = (β*dEVF)**(-γ)
            h2_temp1[ia,ie] = (w2/ψ * c0_temp1[ia,ie]**(-1/γ))**σ
            a0_temp1[ia,ie] = (c0_temp1[ia,ie] + a1_st - \
                              w1*e0_st - b*(1-e0_st) - w2*h2_temp1[ia,ie])/(1+r)

            # Case1: h2=0
            c0_temp2[ia,ie] = (β*dEVF)**(-γ)
            h2_temp2[ia,ie] = 0.0
            a0_temp2[ia,ie] = (c0_temp2[ia,ie] + a1_st - \
                              w1*e0_st - b*(1-e0_st) - w2*h2_temp2[ia,ie])/(1+r)

        # Interpolate value and policy functions
        VFt1  = hh.get_VF(age+1)
        for ind in range(Na*Ne):
            ia  = ind // Ne
            ie  = ind % Ne
            a0_st = a0_state[ia]
            e0_st = e0_state[ie]
            income_t        = w1*e0_st + b*(1-e0_st) + (1+r)*a0_st
            
            # Case1: h2>0
            a1_hat1         = np.interp(a0_st,a0_temp1[:,ie],a1_state)
            if a1_hat1<0:   a1_hat1     = 0.0
            _f = lambda cc: cc - w2*(w2/ψ * cc**(-1/γ))**σ - income_t + a1_hat1 
            c0_hat1         = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            if c0_hat1<=0:   c0_hat1     = 1e-2
            h2_hat1         = (w2/ψ * c0_hat1**(-1/γ))**σ
            VFt1_0_hat1     = np.interp(a1_hat1,a1_state,VFt1[:,0])
            VFt1_1_hat1     = np.interp(a1_hat1,a1_state,VFt1[:,1])
            EVF_hat1        = Π[ie,0]*VFt1_0_hat1 + Π[ie,1]*VFt1_1_hat1
            VF_hat1         = uu(c0_hat1,h2_hat1,γ,ψ,σ,f) + β*EVF_hat1

            # Case2: h2=0
            h2_hat2         = 0.0
            a1_hat2         = np.interp(a0_st,a0_temp2[:,ie],a1_state)
            if a1_hat2<0:   a1_hat2     = 0.0
            c0_hat2         = income_t - a1_hat2
            if c0_hat2<=0:   c0_hat2    = 1e-2
            VFt1_0_hat2     = np.interp(a1_hat2,a1_state,VFt1[:,0])
            VFt1_1_hat2     = np.interp(a1_hat2,a1_state,VFt1[:,1])
            EVF_hat2        = Π[ie,0]*VFt1_0_hat2 + Π[ie,1]*VFt1_1_hat2
            VF_hat2         = uu(c0_hat2,h2_hat2,γ,ψ,σ,f) + β*EVF_hat2

            # Choose either case 1 or 2 whichever gives a higher value function
            VF[ia,ie] = VF_hat1*(VF_hat1>=VF_hat2) + VF_hat2*(VF_hat2>VF_hat1)
            a1[ia,ie] = a1_hat1*(VF_hat1>=VF_hat2) + a1_hat2*(VF_hat2>VF_hat1)
            c0[ia,ie] = c0_hat1*(VF_hat1>=VF_hat2) + c0_hat2*(VF_hat2>VF_hat1)
            h2[ia,ie] = h2_hat1*(VF_hat1>=VF_hat2) + h2_hat2*(VF_hat2>VF_hat1)
    
            
    # Store the policy function in the class object
    dVF     = duc(c0,γ)*(1+r)
    hh.set_dVF(dVF,age)
    hh.set_VF(VF,age)
    hh.set_ap(a1,age)
    hh.set_cp(c0,age)
    hh.set_hp(h2,age)


 #-----------------------------------------
 #   Plot value/policy functions
 #-----------------------------------------

# =============================================================================
#     # Plot Value Function
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, VF[:,0],label="bad shock")
#     plt.plot(a0_state, VF[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Value function V' ")
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================
    


# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, a1[:,0],label="bad shock")
#     plt.plot(a0_state, a1[:,1],label="good shock")
#     plt.plot(a0_state, a0_state,'k--',label="$45^{\circ}$ line")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function a' ")
#     plt.legend(loc='upper left', fontsize = 10)
#     plt.show()
# =============================================================================
    
    
# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, c0[:,0],label="bad shock")
#     plt.plot(a0_state, c0[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function c ")
# #    ax.set_ylim([0, 5])
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================
    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(a0_state, h2[:,0],label="bad shock")
    plt.plot(a0_state, h2[:,1],label="good shock")
    plt.xlabel("State space, a")
    plt.ylabel("Policy function h2 ")
    plt.title("Age: %d" % age)
    ax.set_ylim([0, 5])
    plt.legend(loc='upper right', fontsize = 14)
    plt.show()


