#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Auxiliary functions for cubic VLE calculations
def a_lv(x,a):
    xx = np.outer(x,x)
    aa = np.sqrt(np.outer(a,a))
    return np.sum(xx*aa)

def b_lv(x,b): return np.dot(x,b)

def beta_lv(b,P,R,T): return b*P/(R*T)

def q_lv(a_lv,b_lv,R,T): return a_lv/(b_lv*R*T)

def q_bar_lv(x,a,b,R,T,a_lv,b_lv,q_lv): 
    aa = np.outer(a,a)**0.5
    for i in range(np.size(a)):
        aa[i,i] = 0
    a_bar = 2*aa@x + 2*a*x
    return q_lv*(a_bar/a_lv-b/b_lv)

def I_lv(sigma,eps,Z,beta): 
    ans = np.log((Z+sigma*beta)/(Z+eps*beta))/(sigma-eps)
    return ans

def phi_lv(b,q,Z,beta,I,b_lv): 
    ans = np.exp(b*(Z-1)/b_lv - np.log(Z-beta) - q*I)
    return ans

def Z_coeffs(beta,sigma,eps,q):
    ans = [1,beta*(sigma+eps)-(1+beta),
            (sigma*eps*beta+q-(1+beta)*(sigma+eps))*beta,
            -sigma*eps*beta**2*(1+beta)-q*beta**2]
    return ans

# ---------------------------------------------------------------------
# Patrick Heng
# 03/21/25
# Objective function to find bubble point composition and temperature given 
# liquid mole fractions and pressure
#
# INPUTS: 
#      XX: unknown vector, i.e. F(XX) = 0
#      x: liquid phase mole fraction, vector 
#      P: system pressure, scalar
#      w, Tc, Pc: critical constants, vectors
#      R: Gas constant given in a consistent set of units, scalar
#      optional: 
#      Omega, Psi, sigma, eps: Cubic EOS parameters, default are for SRK
#                              equation
# OUTPUTS:
#      sol: solution vector, first N components being the vapor compositions
#           and the last component being the temperature
# ---------------------------------------------------------------------

def obj_func_bub(XX,x,P,w,Tc,Pc,R,Omega=0.08664,Psi=0.42748,sigma=1,eps=0):
    
    # Get physcial variables from input vector
    N = np.size(x)
    y = XX[0:N]
    T = XX[-1]
    
    # Compute cubic EOS parameters
    b = Omega*R*Tc/Pc
    bl = np.dot(x,b)
    a = Psi*(R**2*Tc**2/Pc)*(1+(0.480+1.574*w-0.176*w**2)*(1-(T/Tc)**0.5))**2
    al = a_lv(x,a)
    bl = np.dot(x,b)
    av = a_lv(y,a)
    bv = b_lv(y,b)
    ql = q_lv(al,bl,R,T)
    qv = q_lv(av,bv,R,T)
    betal = beta_lv(bl,P,R,T)
    betav = beta_lv(bv,P,R,T)
    q_bar_l = q_bar_lv(x,a,b,R,T,al,bl,ql)
    q_bar_v = q_bar_lv(y,a,b,R,T,av,bv,qv)
    
    # Compute cubic roots 
    # Zl -> liquid -> smallest root
    coeffs = Z_coeffs(betal,sigma,eps,ql)
    sol = np.roots(coeffs)
    sol = np.min(sol[sol>0])
    Zl = sol
    # Zv -> vapor -> largest root
    coeffs = Z_coeffs(betav,sigma,eps,qv)
    sol = np.roots(coeffs)
    sol = np.max(np.roots(coeffs))
    Zv = sol
    
    # More cubic EOS parameters...
    Il = I_lv(sigma,eps,Zl,betal)
    Iv = I_lv(sigma,eps,Zv,betav)
    # Compute fugacities
    phi_l = phi_lv(b,q_bar_l,Zl,betal,Il,bl)
    phi_v = phi_lv(b,q_bar_v,Zv,betav,Iv,bv)
    # Compute K values from fugacities (finally)
    K = phi_l/phi_v
    
    # Return equilibrium constraints
    F = np.zeros(N+1)
    F[0:N] = y-K*x
    F[-1] = 1 - np.sum(K*x)

    return F

# ---------------------------------------------------------------------
# Patrick Heng
# 03/21/25
# Objective function to find dew point composition and temperature given 
# vapor mole fractions and pressure
#
# INPUTS: 
#      XX: unknown vector, i.e. F(XX) = 0
#      y: vapor phase mole fraction, vector 
#      P: system pressure, scalar
#      w, Tc, Pc: critical constants, vectors
#      R: Gas constant given in a consistent set of units, scalar
#      optional: 
#      Omega, Psi, sigma, eps: Cubic EOS parameters, default are for SRK
#                              equation
# OUTPUTS:
#      sol: solution vector, first N components being the liquid compositions
#           and the last component being the temperature
# ---------------------------------------------------------------------

def obj_func_dew(XX,y,P,w,Tc,Pc,R,Omega=0.08664,Psi=0.42748,sigma=1,eps=0):
    
    # Get physcial variables from input vector
    N = np.size(y)
    x = XX[0:N]
    T = XX[-1]
    
    # Compute cubic EOS parameters
    b = Omega*R*Tc/Pc
    bl = np.dot(x,b)
    a = Psi*(R**2*Tc**2/Pc)*(1+(0.480+1.574*w-0.176*w**2)*(1-(T/Tc)**0.5))**2
    al = a_lv(x,a)
    bl = np.dot(x,b)
    av = a_lv(y,a)
    bv = b_lv(y,b)
    ql = q_lv(al,bl,R,T)
    qv = q_lv(av,bv,R,T)
    betal = beta_lv(bl,P,R,T)
    betav = beta_lv(bv,P,R,T)
    q_bar_l = q_bar_lv(x,a,b,R,T,al,bl,ql)
    q_bar_v = q_bar_lv(y,a,b,R,T,av,bv,qv)
    
    # Compute cubic roots 
    # Zl -> liquid -> smallest root
    coeffs = Z_coeffs(betal,sigma,eps,ql)
    sol = np.roots(coeffs)
    sol = np.min(sol[sol>0])
    Zl = sol
    # Zv -> vapor -> largest root
    coeffs = Z_coeffs(betav,sigma,eps,qv)
    sol = np.max(np.roots(coeffs))
    Zv = sol
    
    # More cubic EOS parameters...
    Il = I_lv(sigma,eps,Zl,betal)
    Iv = I_lv(sigma,eps,Zv,betav)
    # Compute fugacities
    phi_l = phi_lv(b,q_bar_l,Zl,betal,Il,bl)
    phi_v = phi_lv(b,q_bar_v,Zv,betav,Iv,bv)
    # Compute K values from fugacities (finally)
    K = phi_l/phi_v
    
    # Return equilibrium constraints
    F = np.zeros(N+1)
    F[0:N] = y-K*x
    F[-1] = 1 - np.sum(y/K)

    return F

