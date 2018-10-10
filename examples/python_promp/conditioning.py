""" Very simple conditioning example

Most of the other examples work with a 7 DoF robot. In this example, we show
simply a Polynomial with fixed parameters on a single degree of freedom, and
show how conditioning work in this scenario.
"""

import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import json

print promp.__file__

full_basis = {
        'conf': [
                {"type": "poly", "nparams": 0, "conf": {"order": 3}}
            ],
        'params': []
        }
dim_basis_fun = promp.dim_comb_basis(**full_basis)
dof = 1
w_dim = dof*dim_basis_fun

p = promp.FullProMP(num_joints=1, basis=full_basis)
p.mu_w = np.zeros(w_dim)
p.Sigma_w = 1e2*np.eye(w_dim)
p.Sigma_y = 1e-4*np.eye(dof)

n_samples = 5 # Number of samples to draw
sample_time = [np.linspace(0,1,200) for i in range(n_samples)]

#1) Condition the ProMP to pass through q=0.5 at time 0, q=-1 at time 0.5 and q=1 at time 1.0

#1.1) Procedure 1: Use the E-step doing all at the same time
e_step = p.E_step(times=[[0.0,0.5,1.0]], Y = np.array([[[0.5],[-1.0],[1.0]]]))

#1.2) Procedure 2: Condition the ProMP one by one obs
p.condition(t=0.0, T=1.0, q=np.array([0.5]), ignore_Sy=False)
p.condition(t=0.5, T=1.0, q=np.array([-1]), ignore_Sy=False)
p.condition(t=1.0, T=1.0, q=np.array([1]), ignore_Sy=False)

# Draw samples of recursive conditioning
recursive_cond_samples = p.sample(sample_time)

# Draw samples of E-step conditioning
p.mu_w = e_step['w_means'][0]
p.Sigma_w = e_step['w_covs'][0]
e_step_cond_samples = p.sample(sample_time)

for i in range(n_samples):
    plt.plot(sample_time[i], recursive_cond_samples[i][:,0], color='green')
    plt.plot(sample_time[i], e_step_cond_samples[i][:,0], color='blue')
plt.title('Samples of the conditioned ProMPs')
plt.show()


