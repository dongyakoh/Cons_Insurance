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
    def __init__(self, γ=0.5, σ=1.5, ξ=0.8, 𝛽=0.94, pn=1.5, Na=500, T=6):
        
        # Store the characteristics of household in the class object
        self.γ          = γ                   # Coefficient of RRA
        self.σ          = σ                   # Coefficient of RRA
        self.ξ          = ξ
        self.pn         = pn                   # Discount factor
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
        self.VF         = np.zeros((self.Na,self.Ne,T))   # Value function, a1
        self.dVF        = np.zeros((self.Na,self.Ne,T))   # Value function, a1
        self.ap         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1
        self.cf         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1
        self.cn         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1


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
    
    def get_ap(self,age):
        '''
        This function retrieves the current policy function
        '''
        return self.ap[:,:,age]

    # Update the policy function, a1
    def set_ap(self, a1, age):
        '''
        This function sets a new policy function
        '''
        self.ap[:,:,age] = a1
        
    def get_cf(self,age):
        '''
        This function retrieves the current policy function
        '''
        return self.cf[:,:,age]

    # Update the policy function, a1
    def set_cf(self, c0, age):
        '''
        This function sets a new policy function
        '''
        self.cf[:,:,age] = c0

    def get_cn(self,age):
        '''
        This function retrieves the current policy function
        '''
        return self.cn[:,:,age]

    # Update the policy function, a1
    def set_cn(self, c1, age):
        '''
        This function sets a new policy function
        '''
        self.cn[:,:,age] = c1
        
        
#-----------------------------------------
#   Several functions
#-----------------------------------------
def uu(cf,cn,γ,σ,ξ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return ξ*cf**(1-1/γ)/(1-1/γ) + (1-ξ)*cn**(1-1/σ)/(1-1/σ)

def duf(cf,γ,ξ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return ξ*cf**(-1/γ)

def dun(cn,σ,ξ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return (1-ξ)*cn**(-1/σ)
        



#-----------------------------------------
#   Run the program
#-----------------------------------------
              
# Create a household instance
hh     = household()

r,w,b,Π,γ,σ,ξ,β,pn = hh.r,hh.w,hh.b,hh.Π,hh.γ,hh.σ,hh.ξ,hh.β,hh.pn
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
    cf    = np.zeros((Na,Ne))
    cn    = np.zeros((Na,Ne))

    # Iterate for each state (a0,e0)
    # enumerate() gives an index and value of the list
    if age==hh.T-1:   # Find an optimal saving for t=T
        for ind in range(Na*Ne):
            ia  = ind // Ne
            ie  = ind % Ne
            a0_st = a0_state[ia]
            e0_st = e0_state[ie]
            a1[ia,ie] = 0.0
            income_t = w*e0_st + b*(1-e0_st) + (1+r)*a0_st
            _f = lambda cc: (cc + pn*(ξ/(1-ξ)*pn*cc**(-1/γ))**(-σ)) - income_t
            _cf = opt.brentq(_f, 1e-5, income_t - 1e-2, rtol=1e-12)
            cf[ia,ie] = _cf
            cn[ia,ie] = (ξ/(1-ξ)*pn*_cf**(-1/γ))**(-σ)
            VF[ia,ie] = uu(cf[ia,ie],cn[ia,ie],γ,σ,ξ)
    
    else:   # Find an optimal saving by using a root-finding method for t<T
        a0_temp     = np.zeros((Na,Ne))
        cf_temp     = np.zeros((Na,Ne))
        cn_temp     = np.zeros((Na,Ne))
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
            cf_temp[ia,ie] = (β*dEVF / ξ)**(-γ)
            cn_temp[ia,ie] = (pn*β*dEVF / (1-ξ))**(-σ)
            a0_temp[ia,ie] = (cf_temp[ia,ie] + pn*cn_temp[ia,ie] + a1_st - \
                              w*e0_st - b*(1-e0_st))/(1+r)

        # Interpolate value and policy functions
        VFt1  = hh.get_VF(age+1)
        for ie in range(Ne):
            a1[:,ie]    = np.interp(a0_state,a0_temp[:,ie],a1_state)
            a1[a1[:,ie]<0,ie]    = 0.0
            cf[:,ie]    = np.interp(a0_state,a0_temp[:,ie],cf_temp[:,ie])
            cf[cf[:,ie]<=0,ie] = 1e-2
            cn[:,ie]= w*e0_state[ie] + b*(1-e0_state[ie]) + (1+r)*a0_state - a1[:,ie] - cf[:,ie]
            cn[cn[:,ie]<=0,ie] = 1e-2
            VFt1_0      = np.interp(a1[:,ie],a1_state,VFt1[:,0])
            VFt1_1      = np.interp(a1[:,ie],a1_state,VFt1[:,1])
            EVF         = Π[ie,0]*VFt1_0 + Π[ie,1]*VFt1_1
            VF[:,ie]    = uu(cf[:,ie],cn[:,ie],γ,σ,ξ) +  β*EVF
    
            
    # Store the policy function in the class object
    dVF     = duf(cf,γ,ξ)*(1+r)
    hh.set_dVF(dVF,age)
    hh.set_VF(VF,age)
    hh.set_ap(a1,age)
    hh.set_cf(cf,age)
    hh.set_cn(cn,age)


 #-----------------------------------------
 #   Plot value/policy functions
 #-----------------------------------------

    # Plot Value Function
# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, VF[:,0],label="bad shock")
#     plt.plot(a0_state, VF[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Value function V' ")
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
#     
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
#     plt.plot(a0_state, cf[:,0],label="bad shock")
#     plt.plot(a0_state, cf[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function c ")
#     ax.set_ylim([0, 5])
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================
    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(a0_state, cn[:,0],label="bad shock")
    plt.plot(a0_state, cn[:,1],label="good shock")
    plt.xlabel("State space, a")
    plt.ylabel("Policy function c ")
    ax.set_ylim([0, 5])
    plt.legend(loc='lower right', fontsize = 14)
    plt.show()


