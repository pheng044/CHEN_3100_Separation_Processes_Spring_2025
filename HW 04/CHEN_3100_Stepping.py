#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Patrick Heng
# 02/01/25
# Function for stepping through an x-y diagram 
#
# INPUTS:
#      x_0: initial liquid composition, scalar
#      x_N: output liquid composition, scalar
#      eq_line: equillibrium line, given as a function of x
#               i.e. y = f(x), (can use lambda function)
#      op_line: operating line, given as a function of x
#               i.e. y = f(x), (can use lambda function)
#      max_iter: optional, maximum number of allowed steps in column
#                default: 30
#
# OUTPUTS:
#     x_eq: vector of x values of intersections with eq. and
#           op. lines, i.e.
#           x_eq = [x_0,x_0,x_1,x_1, ... , x_{N-1}, x_{N-1}, x_N]
#     y_eq: vector of y values of intersections with eq. and
#           op. lines, i.e.
#           y_eq = [y_1,y_1,y_2,y_2, ... , y_N, y_N, y_{N+1}]
#
#     Using matplotlib, graphing the steps simply becomes:
#           plt.plot(x_eq,y_eq)

def stepping(x_0,x_N,eq_line,op_line,max_iter = 30):
    
    import numpy as np
    from scipy.optimize import root_scalar
    
    # Initialize iteration at inlet comp
    x_n = x_0
    y_1 = eq_line(x_0)
    
    # Resulting vapor equilibrium
    y_n = op_line(x_0)
    
    # Initialize equillibirum vectors
    x_eq = [x_0,x_0]
    y_eq = [y_1,y_n]
    
    # Iterate through stepping
    for i in range(max_iter):
        
        # Solve for next x value using secant method, i.e. solve:
        #      f(x_n) =  y_n - eq_line(x_n) = 0 
        # Use previous x value as initial guess for secant
        sol = root_scalar(lambda x: y_n-eq_line(x),method='secant',x0=x_n)
        
        # Next x_n value is the root of the above equation
        x_n = sol.root
        
        # Resulting vapor equilibrium, y_n
        y_n = op_line(x_n)
        
        # Add step coordinates to equillibrium vectors
        x_eq = np.hstack((x_eq,[x_n,x_n]))
        y_eq = np.hstack((y_eq,[eq_line(x_n),y_n]))
        
        # If the outlet compsition is achieved, stop
        if x_n > x_N:
            break
    
    return x_eq[0:-1], y_eq[0:-1]

