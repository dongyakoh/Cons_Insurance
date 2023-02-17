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
        self.γf         = 0.578                   # Coefficient of RRA
        self.γn         = 1.2                   # Coefficient of RRA
        self.ξ          = 0.8
        self.𝛽          = 0.95                  # Discount factor
        self.σ          = 0.85
        self.ψ          = 1.95
        self.f          = 0.0306
        self.pn         = 1.2
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
        self.cf         = np.zeros((self.Na,self.Ne,self.T))   # Policy function, a1
        self.cn         = np.zeros((self.Na,self.Ne,self.T))   # Policy function, a1
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
    def set_cf(self, cf0, age):
        '''
        This function sets a new policy function
        '''
        self.cf[:,:,age] = cf0

    # Update the policy function, a1
    def set_cn(self, cn0, age):
        '''
        This function sets a new policy function
        '''
        self.cn[:,:,age] = cn0

    # Update the policy function, a1
    def set_hp(self, h2, age):
        '''
        This function sets a new policy function
        '''
        self.hp[:,:,age] = h2
        
        
#-----------------------------------------
#   Several functions
#-----------------------------------------
def uu(cf,cn,h2,γf,γn,ξ,ψ,σ,f):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    uc = ξ*cf**(1-1/γf)/(1-1/γf) + (1-ξ)*cn**(1-1/γn)/(1-1/γn)
    h2_temp = h2 + np.equal(h2,0.0)*0.1
    return uc - np.not_equal(h2,0.0)*(ψ*h2_temp**(1+1/σ)/(1+1/σ) + f)
    

def ducf(cf,γf,ξ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return ξ*cf**(-1/γf)

def ducn(cn,γn,ξ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    return (1-ξ)*cn**(-1/γn)

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

r,w1,w2,b,Π,γf,γn,ξ,pn,σ,β,ψ,f = hh.r,hh.w1,hh.w2,hh.b,hh.Π,hh.γf,hh.γn,hh.ξ,hh.pn,hh.σ,hh.β,hh.ψ,hh.f
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
            _f = lambda cc: cc + pn*(pn*ξ/(1-ξ)*cc**(-1/γf))**(-γn) - w2*(w2/ψ * ξ*cc**(-1/γf))**σ - income_t
            cf_temp1    = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            cn_temp1    = (pn*ξ/(1-ξ)*cf_temp1**(-1/γf))**(-γn)
            h2_temp1    = (w2/ψ * ξ*cf_temp1**(-1/γf))**σ
            VF_temp1    = uu(cf_temp1,cn_temp1,h2_temp1,γf,γn,ξ,ψ,σ,f)

            # Case2: h2=0
            _f = lambda cc: cc + pn*(pn*ξ/(1-ξ)*cc**(-1/γf))**(-γn) - income_t
            cf_temp2    = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            cn_temp2    = (pn*ξ/(1-ξ)*cf_temp2**(-1/γf))**(-γn)
            h2_temp2    = 0.0
            VF_temp2    = uu(cf_temp2,cn_temp2,h2_temp1,γf,γn,ξ,ψ,σ,f)

            # Choose either case 1 or 2 whichever gives a higher value function
            h2[ia,ie] = h2_temp1*(VF_temp1>=VF_temp2) + h2_temp2*(VF_temp2>VF_temp1)
            cf[ia,ie] = cf_temp1*(VF_temp1>=VF_temp2) + cf_temp2*(VF_temp2>VF_temp1)
            cn[ia,ie] = cn_temp1*(VF_temp1>=VF_temp2) + cn_temp2*(VF_temp2>VF_temp1)
            VF[ia,ie] = VF_temp1*(VF_temp1>=VF_temp2) + VF_temp2*(VF_temp2>VF_temp1)
    
    else:   # Find an optimal saving by using a root-finding method for t<T
        a0_temp1     = np.zeros((Na,Ne))
        cf_temp1     = np.zeros((Na,Ne))
        cn_temp1     = np.zeros((Na,Ne))
        h2_temp1     = np.zeros((Na,Ne))
        a0_temp2     = np.zeros((Na,Ne))
        cf_temp2     = np.zeros((Na,Ne))
        cn_temp2     = np.zeros((Na,Ne))
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
            cf_temp1[ia,ie] = (β*dEVF/ξ)**(-γf)
            cn_temp1[ia,ie] = (pn*ξ/(1-ξ)*cf_temp1[ia,ie]**(-1/γf))**(-γn)
            h2_temp1[ia,ie] = (w2/ψ * ξ*cf_temp1[ia,ie]**(-1/γf))**σ
            a0_temp1[ia,ie] = (cf_temp1[ia,ie] + pn*cn_temp1[ia,ie] + a1_st - \
                              w1*e0_st - b*(1-e0_st) - w2*h2_temp1[ia,ie])/(1+r)

            # Case1: h2=0
            cf_temp2[ia,ie] = (β*dEVF/ξ)**(-γf)
            cn_temp2[ia,ie] = (pn*ξ/(1-ξ)*cf_temp2[ia,ie]**(-1/γf))**(-γn)
            h2_temp2[ia,ie] = 0.0
            a0_temp2[ia,ie] = (cf_temp2[ia,ie] + pn*cn_temp2[ia,ie] + a1_st - \
                              w1*e0_st - b*(1-e0_st) - w2*h2_temp2[ia,ie])/(1+r)
#        print(np.concatenate((a0_temp1,a0_temp2), axis=1))
        
        # Interpolate value and policy functions
        VFt1  = hh.get_VF(age+1)
        print(VFt1)
        for ind in range(Na*Ne):
            ia  = ind // Ne
            ie  = ind % Ne
            a0_st = a0_state[ia]
            e0_st = e0_state[ie]
            income_t        = w1*e0_st + b*(1-e0_st) + (1+r)*a0_st
            
            # Case1: h2>0
            a1_hat1         = np.interp(a0_st,a0_temp1[:,ie],a1_state)
            if a1_hat1<0:   a1_hat1     = 0.0
            _f = lambda cc: cc + pn*(pn/(1-ξ)*ξ*cc**(-1/γf))**(-γn) + a1_hat1  - \
                            w2*(w2/ψ*ξ*cc**(-1/γf))**σ - income_t
            cf_hat1         = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            if cf_hat1<=0:   cf_hat1     = 1e-2
            cn_hat1         = (pn/(1-ξ)*ξ*cf_hat1**(-1/γf))**(-γn)
            if cn_hat1<=0:   cn_hat1     = 1e-2
            h2_hat1         = (w2/ψ*ξ*cf_hat1**(-1/γf))**σ
            VFt1_0_hat1     = np.interp(a1_hat1,a1_state,VFt1[:,0])
            VFt1_1_hat1     = np.interp(a1_hat1,a1_state,VFt1[:,1])
            EVF_hat1        = Π[ie,0]*VFt1_0_hat1 + Π[ie,1]*VFt1_1_hat1
            VF_hat1         = uu(cf_hat1,cn_hat1,h2_hat1,γf,γn,ξ,ψ,σ,f) + β*EVF_hat1

            # Case2: h2=0
            h2_hat2         = 0.0
            a1_hat2         = np.interp(a0_st,a0_temp2[:,ie],a1_state)
            if a1_hat2<0:   a1_hat2     = 0.0
            _f = lambda cc: cc + pn*(pn/(1-ξ)*ξ*cc**(-1/γf))**(-γn) + a1_hat2 - income_t
            cf_hat2         = opt.brentq(_f, 1e-5, income_t+10, rtol=1e-12)
            if cf_hat2<=0:   cf_hat2     = 1e-2
            cn_hat2         = (pn/(1-ξ)*ξ*cf_hat2**(-1/γf))**(-γn)
            if cn_hat2<=0:   cn_hat2     = 1e-2
            VFt1_0_hat2     = np.interp(a1_hat2,a1_state,VFt1[:,0])
            VFt1_1_hat2     = np.interp(a1_hat2,a1_state,VFt1[:,1])
            EVF_hat2        = Π[ie,0]*VFt1_0_hat2 + Π[ie,1]*VFt1_1_hat2
            VF_hat2         = uu(cf_hat2,cn_hat2,h2_hat2,γf,γn,ξ,ψ,σ,f) + β*EVF_hat2
            
#            print(VF_hat1,VF_hat2)
            
            # Choose either case 1 or 2 whichever gives a higher value function
            VF[ia,ie] = VF_hat1*(VF_hat1>=VF_hat2) + VF_hat2*(VF_hat2>VF_hat1)
            a1[ia,ie] = a1_hat1*(VF_hat1>=VF_hat2) + a1_hat2*(VF_hat2>VF_hat1)
            cf[ia,ie] = cf_hat1*(VF_hat1>=VF_hat2) + cf_hat2*(VF_hat2>VF_hat1)
            cn[ia,ie] = cn_hat1*(VF_hat1>=VF_hat2) + cn_hat2*(VF_hat2>VF_hat1)
            h2[ia,ie] = h2_hat1*(VF_hat1>=VF_hat2) + h2_hat2*(VF_hat2>VF_hat1)
    
            
    # Store the policy function in the class object
    dVF     = ducf(cf,γf,ξ)*(1+r)
    hh.set_dVF(dVF,age)
    hh.set_VF(VF,age)
    hh.set_ap(a1,age)
    hh.set_cf(cf,age)
    hh.set_cn(cn,age)
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
#     plt.legend(loc='upper left', fontsize = 14)
#     plt.show()
# =============================================================================
    

# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, a1[:,0],label="bad shock")
#     plt.plot(a0_state, a1[:,1],label="good shock")
#     plt.plot(a0_state, a0_state,'k--',label="$45^{\circ}$ line")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function a' ")
#     plt.title("Age: %d" % age)
#     plt.legend(loc='upper left', fontsize = 10)
#     plt.show()
# =============================================================================
    
    
# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, cf[:,0],label="bad shock")
#     plt.plot(a0_state, cf[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function cf ")
#     plt.ylim(0,5)
#     plt.title("Age: %d" % age)
#     plt.legend(loc='upper left', fontsize = 14)
#     plt.show()
# =============================================================================


# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_state, cn[:,0],label="bad shock")
#     plt.plot(a0_state, cn[:,1],label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function cn ")
#     plt.ylim(0,5)
#     plt.title("Age: %d" % age)
#     plt.legend(loc='upper left', fontsize = 14)
#     plt.show()
# =============================================================================

    
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(a0_state, h2[:,0],label="bad shock")
    plt.plot(a0_state, h2[:,1],label="good shock")
    plt.xlabel("State space, a")
    plt.ylabel("Policy function h2 ")
    plt.title("Age: %d" % age)
    ax.set_ylim([0, 5])
    plt.legend(loc='upper left', fontsize = 14)
    plt.show()


