#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ---------------------------------------------------------------------
# Patrick Heng
# 02/01/25
# Function for stepping through an absorption/stripping x-y diagram 
#
# INPUTS:
#      x_0: initial liquid composition, scalar
#      x_N: output liquid composition, scalar
#      eq_line: equilibrium line, given as a function of x
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
# ---------------------------------------------------------------------


def stepping_AS(x_0,x_N,eq_line,op_line,max_iter = 30):
    
    import numpy as np
    from scipy.optimize import root_scalar
    
    # Initialize iteration at inlet comp
    x_n = x_0
    y_1 = eq_line(x_0)
    
    # Resulting vapor equilibrium
    y_n = op_line(x_0)
    
    # Initialize equilibrium vectors
    x_eq = [x_0,x_0]
    y_eq = [y_1,y_1]
    
    # Iterate through stepping
    for i in range(max_iter):
        
        # Solve for next x value using secant method, i.e. solve:
        #      f(x_n) =  y_n - eq_line(x_n) = 0 
        # Use previous x value as initial guess for secant method
        sol = root_scalar(lambda x: y_n-eq_line(x),method='secant',x0=x_n)
        
        # Next x_n value is the root of the above equation
        x_n = sol.root
        
        # Resulting vapor equilibrium, y_n
        y_n = op_line(x_n)
        
        # Add step coordinates to equilibrium vectors
        x_eq = np.hstack((x_eq,[x_n,x_n]))
        y_eq = np.hstack((y_eq,[eq_line(x_n),y_n]))
        
        # If the outlet composition is achieved, stop
        if x_n > x_N:
            break
    
    return x_eq[0:-1], y_eq[0:-1]


# ---------------------------------------------------------------------
# Patrick Heng
# 02/22/25
# Function for stepping through a McCabe-Thiele x-y diagram 
#
# INPUTS:
#      x_0: initial liquid composition, scalar
#      x_N: output liquid composition, scalar
#      eq_line: equilibrium line, given as a function of x
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
# ---------------------------------------------------------------------


def stepping_MT(x_D,x_B,eq_line,op_line,max_iter = 30):
    
    import numpy as np
    from scipy.optimize import root_scalar
    
    # Initialize iteration at distillate comp
    x_n = x_D
    y_1 = op_line(x_D)
    
    # Resulting vapor equilibrium
    y_n = op_line(x_D)
    
    # Initialize equilibrium vectors
    x_eq = [x_D,x_D]
    y_eq = [y_1,y_1]
    
    # Iterate through stepping
    for i in range(max_iter):
        
        # Solve for next x value using secant method, i.e. solve:
        #      f(x_n) =  y_n - eq_line(x_n) = 0 
        # Use previous x value as initial guess for secant method
        sol = root_scalar(lambda x: y_n-eq_line(x),x0=x_n)
        
        # Next x_n value is the root of the above equation
        x_n = sol.root
        
        # Resulting vapor equilibrium, y_n
        y_n = op_line(x_n)
        
        # Add step coordinates to equilibrium vectors
        x_eq = np.hstack((x_eq,[x_n,x_n]))
        y_eq = np.hstack((y_eq,[eq_line(x_n),y_n]))

        # If the bottoms composition is achieved, stop
        if x_n < x_B:
            break
    
    return x_eq[0:], y_eq[0:]


