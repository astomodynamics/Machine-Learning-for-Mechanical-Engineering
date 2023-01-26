#####################################################
##  ME8813ML Homework 1: 
##  Implement a quasi-Newton optimization method for data fitting
#####################################################
import numpy as np
import matplotlib.pyplot as plt


########################################################
## Implement a parameter fitting function fit() so that 
##  p = DFP_fit(x,y)
## returns a list of the parameters as p of model:
##  p0 + p1*cos(2*pi*x) + p2*cos(4*pi*x) + p3*cos(6*pi*x)  
########################################################


# Fixing random state for reproducibility
np.random.seed(19680801)

dx = 0.1
x_lower_limit = 0
x_upper_limit = 40                                       
x = np.arange(x_lower_limit, x_upper_limit, dx)
data_size = len(x)                                 # data size
noise = np.random.randn(data_size)                 # white noise

# Original dataset 
y = 2.0 + 3.0*np.cos(2*np.pi*x) + 1.0*np.cos(6*np.pi*x) + noise


###########################################
# p = DFP_fit(x, y)
###########################################


fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y)
axs[0].set_xlim(x_lower_limit, x_upper_limit)
axs[0].set_xlabel('x')
axs[0].set_ylabel('observation')
axs[0].grid(True)


#########################################
## Plot the predictions from your fitted model here
axs[1].set_xlim(x_lower_limit, x_upper_limit)
axs[1].set_xlabel('x')
axs[1].set_ylabel('model prediction')

fig.tight_layout()
plt.show()
