# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:35:21 2022

@author: Don Koh
"""


import numpy as np
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt

# Define household class
class household:
    
    # Initialize a household object.
    # Provide predetermined parameter values
    def __init__(self, σ=2, 𝛽=0.985,𝜂=0.09, 𝛾=0, Na=50, T=6):

        # Store the characteristics of household in the class object
        self.σ          = σ                   # Coefficient of RRA
        self.𝛽          = 𝛽                   # Discount factor
        self.𝜂          = 𝜂                   # Income shock
        self.𝛾          = 𝛾
        self.r          = 0.011
        self.w          = 1
        self.Na         = Na                  # Number of grid points for a0 state
        self.T          = T                  # 
        self.a_min      = 0
        self.a_max      = 20
        self.h_min      = 0
        self.h_max      = 40
        self.a0_state   = np.linspace(self.a_min,self.a_max,self.Na) # Discretize a0 state
        self.e0_state   = np.asarray([1-𝜂, 1+𝜂]) # Construct e0 state as ndarray type instead of list type
        self.Ne         = len(self.e0_state)
        self.Π          = np.asarray([[(1+𝛾)/2,(1-𝛾)/2],[(1-𝛾)/2,(1+𝛾)/2]])   # Stochastic matrix, Prob(e1|e0), as ndarray type
        self.Vf         = np.zeros((self.Na,self.Ne,T+1))   # Value function, a1
        self.ap         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1
        self.hp         = np.zeros((self.Na,self.Ne,T))  # Policy function, current hours
        self.b          = 0.3
        self.kappa      = 2.5
        self.eta        = 0.5

    def util(self,cons,hour):
        '''
        This function returns the value of CRRA utility with ssigma
        u(c) = c**(1-ssigma)/(1-ssigma)
        '''
        if self.σ != 1:
            uu = cons**(1-self.σ)/(1-self.σ) - self.kappa*hour**(1+1/self.eta)/(1+1/self.eta)
        else:
            uu = np.log(cons)
        return uu

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
        
    # Update the policy function, h1
    def set_hp(self, h1, age):
        self.hp[:,:,age] = h1
        
        
if __name__ == "__main__":
    
    # Model parameters 
    𝜂      = 0.09
    𝛾      = 3
    σ      = 2

    # Create a household instance
    hh     = household(𝜂=𝜂,𝛾=𝛾,σ=σ)