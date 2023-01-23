# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:50:02 2023

@author: dkoh
"""

import sys
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from complete_poly import complete_polynomial,complete_polynomial_der
import quantecon as qe



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
        self.a0_grid    = np.linspace(self.a_min,self.a_max,self.Na) # Discretize a0 state
        self.e0_grid    = np.asarray([0, 1]) # Construct e0 state as ndarray type instead of list type
        self.Π          = np.asarray([[1.0,0.0],[0.,1.0]])   # Stochastic matrix, Prob(e1|e0), as ndarray type
        self.poly_deg   = 3
        self.Np         = np.sum(np.asarray([i+1 for i in range(self.poly_deg+1)]))
        self.v_coef     = np.zeros((self.Np,T))   # Value function, a1
        self.a1_coef    = np.zeros((self.Np,T))   # Policy function, a1

        self.init_grid()


    def init_grid(self):
        self.state      = qe.gridtools.cartesian([self.a0_grid, self.e0_grid])
        return
        
        
    def get_v(self,age):
        '''
        This function updates the value function
        '''
        return self.v_coef[:,age]
    
    # Update the policy function, a1
    def set_v(self, vv, age):
        self.v_coef[:,age] = vv
    
    def get_a1(self,age):
        '''
        This function updates the policy function
        '''
        return self.a1_coef[:,age]

    # Update the policy function, a1
    def set_a1(self, a1, age):
        self.a1_coef[:,age] = a1
        

#-----------------------------------------
#   Several functions
#-----------------------------------------
def uu(cons,σ):
    '''
    This function returns the value of CRRA utility with ssigma
    u(c) = c**(1-ssigma)/(1-ssigma)
    '''
    if σ == 1:
        return -1e10 if cons < 1e-10 else np.log(cons)
    else:
        return -1e10 if cons < 1e-10 else (cons**(1-σ) - 1.0)/(1-σ)

def du(cons,σ):
    '''Derivative of CRRA utility function'''
    if σ == 1:
        return -1e10 if cons < 1e-10 else 1/cons
    else:
        return -1e10 if cons < 1e-10 else cons**(-σ)

def compute_EV_scalar(hh, istate, ap, age):
    state,e1,P = hh.state,hh.e0_grid,hh.Π

    # All possible exogenous states tomorrow
    v_coef      = hh.get_v(age+1)   # value function coefficient
    e0_state    = state[:,1]     # Current state of e
    weights     = P[int(e0_state[istate]), :] # Transition probability conditional on e0
    phi         = complete_polynomial(np.vstack([np.ones(Ne)*ap, e1]),
                                  hh.poly_deg).T
    val         = weights@(phi @ v_coef)

    return val

def compute_dEV_scalar(hh, istate, ap, age):
    state,e1,P = hh.state,hh.e0_grid,hh.Π

    # All possible exogenous states tomorrow
    v_coef      = hh.get_v(age+1)   # value function coefficient
    e0_state    = state[:,1]     # Current state of e
    weights     = P[int(e0_state[istate]), :] # Transition probability conditional on e0
    dphi         = complete_polynomial_der(np.vstack([np.ones(Ne)*ap, e1]),
                                  hh.poly_deg, 0).T
    val         = weights@(dphi @ v_coef)

    return val


#-----------------------------------------
#   Run the program
#-----------------------------------------
              
# Create a household instance
hh     = household()

r,w,b,β,σ   = hh.r,hh.w,hh.b,hh.β,hh.σ
Na          = hh.Na
Ne          = hh.Ne
poly_deg    = hh.poly_deg
state       = hh.state
a0_state    = state[:,0]
e0_state    = state[:,1]

# Backward induction: solve an individual's problem from the last period
#for age in reversed(range(hh.T)):
for age in reversed(range(hh.T)):
    sys.stdout.write('age = %d.\n' % age)
    
    a1      = np.empty(Na*Ne)
    c0      = np.empty(Na*Ne)
    VF      = np.empty(Na*Ne)
    # Iterate for each state (a0,e0)
    for ik in range(Na*Ne):
        
        a0 = a0_state[ik]
        e0 = e0_state[ik]

        if age==hh.T-1:   # Find an optimal saving for t=T
            a1[ik] = 0
            c0[ik] = w*e0 + b*(1-e0) + (1+r)*a0 - a1[ik]
            VF[ik] = uu(c0[ik],σ)
    
        else:   # Find an optimal saving by using a root-finding method for t<T
            amin = 0
            amax = w*e0 + b*(1-e0) + (1+r)*a0 - 1e-2
            _f = lambda ap: (du(w*e0 + b*(1-e0) + (1+r)*a0 - ap,σ) - \
                             β*compute_dEV_scalar(hh,ik,ap,age))
            _ap = opt.brentq(_f, -amax*10, amax, rtol=1e-12)

            a1[ik] = _ap
            c0[ik] = w*e0 + b*(1-e0) + (1+r)*a0 - a1[ik]
            VF[ik] = uu(c0[ik],σ)+β*compute_EV_scalar(hh,ik,_ap,age)

    Phi     = complete_polynomial(state.T, poly_deg).T
    a1_coef = la.lstsq(Phi, a1)[0]
    v_coef  = la.lstsq(Phi, VF)[0]
    
    hh.set_a1(a1_coef,age)
    hh.set_v(v_coef,age)


    #-----------------------------------------
    #   Plot policy function
    #-----------------------------------------
    a0_grid = hh.a0_grid
    e0_grid = hh.e0_grid
    phi0        = complete_polynomial(np.vstack([a0_grid, np.ones(Na)*e0_grid[0]]),
                                       hh.poly_deg).T
    phi1        = complete_polynomial(np.vstack([a0_grid, np.ones(Na)*e0_grid[1]]),
                                       hh.poly_deg).T
    
    a1_pf0  = phi0 @ a1_coef
    a1_pf0[a1_pf0 < 0]  = 0
    c0_pf0  = w*e0_grid[0] + b*(1-e0_grid[0]) + (1+r)*a0_grid - a1_pf0
    a1_pf1  = phi1 @ a1_coef
    a1_pf1[a1_pf1 < 0]  = 0
    c0_pf1  = w*e0_grid[1] + b*(1-e0_grid[1]) + (1+r)*a0_grid - a1_pf0
    vf_pf0  = phi0 @ v_coef
    vf_pf1  = phi1 @ v_coef

    # Plot Value Function
# =============================================================================
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_grid, vf_pf0,label="bad shock")
#     plt.plot(a0_grid, vf_pf1,label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Value function v' ")
#     plt.legend(loc='lower right', fontsize = 14)
#     plt.show()
# =============================================================================

    # Plot Policy Function
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.plot(a0_grid, a1_pf0,label="bad shock")
    plt.plot(a0_grid, a1_pf1,label="good shock")
    plt.plot(a0_grid, a0_grid,'k--',label="$45^{\circ}$ line")
    plt.xlabel("State space, a")
    plt.ylabel("Policy function a' ")
    plt.legend(loc='upper left', fontsize = 10)
    plt.show()

# =============================================================================
#     # Plot Policy Function
#     fig, ax = plt.subplots(figsize=(5, 3))
#     plt.plot(a0_grid, c0_pf0,label="bad shock")
#     plt.plot(a0_grid, c0_pf1,label="good shock")
#     plt.xlabel("State space, a")
#     plt.ylabel("Policy function a' ")
#     plt.legend(loc='upper left', fontsize = 10)
#     plt.show()
# =============================================================================








