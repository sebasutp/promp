Probabilistic Movement Primitive Library
========================================

A Probabilistic Movement Primitive (ProMP) is a probabilistic generative model used to model movement, it is typically
used in the robotics community to learn movements from a human demonstrator (or teacher) and replicate those
movements in a robotic platform.

This repository contains the implementation in Python and C++ of the Probabilistic Movement Primitive framework as
described in [this paper](https://arxiv.org/pdf/1808.10648.pdf). Typically, the operations we want from a ProMP
are:

1) Learning a ProMP from several human demonstrations. Typically, we consider we learn from trajectories in joint 
space.
2) Conditioning in joint space. For example, force the movement to start in the current position of the robot.
3) Conditioning in task space. For example, conditioning a table tennis strike movement to hit the ball in a
certain position in Cartesian coordinates.
4) Controlling a robot to track a ProMP

We provide code for the first three operations. We assume that the learning is done in Python, and only implemented
the adaptation operators in C++ (They are also provided in Python). We also provide code for the following operations:

* Compute the likelihood of a given trajectory for a given ProMP
* Sample trajectories from a ProMP
* Save and load ProMPs

Code Examples
-------------

Please refer to the example folder in the repository. We just show a simple example
that illustrates how to use the API in general. 

### A very simple example

The following example loads a dataset of strike trajectories from a file called 
"strike_mov.npz" and trains a ProMP with the given trajectories. The example
file is provided as part of the repository. Subsequently, the script draws samples
from the learned ProMP and a ProMP conditioned to start at a particular location.

```
import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np
from matplotlib import pyplot as plt

#1) Take the first 10 striking movements from a file with recorded demonstrations
with open('strike_mov.npz','r') as f:
    data = np.load(f)
    time = data['time'][0:10]
    Q = data['Q'][0:10]

#2) Create a ProMP with basis functions: 3 RBFs with scale 0.25 and 
#   centers 0.25, 0.5 and 0.75. Use also polynomial basis functions of 
#   degree one (constant and linear term)
full_basis = {
        'conf': [
                {"type": "sqexp", "nparams": 4, "conf": {"dim": 3}},
                {"type": "poly", "nparams": 0, "conf": {"order": 1}}
            ],
        'params': [np.log(0.25),0.25,0.5,0.75]
        }
robot_promp = promp.FullProMP(basis=full_basis)

#3) Train ProMP with NIW prior on the covariance matrix (as described in the paper)

dof = 7
dim_basis_fun = 5
inv_whis_mean = lambda v, Sigma: utils.make_block_diag(Sigma, dof)
prior_Sigma_w = {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean}
train_summary = robot_promp.train(time, q=Q, max_iter=10, prior_Sigma_w=prior_Sigma_w,
        print_inner_lb=True)


#4) Plot some samples from the learned ProMP and conditioned ProMP

n_samples = 5 # Number of samples to draw
plot_dof = 3 # Degree of freedom to plot
sample_time = [np.linspace(0,1,200) for i in range(n_samples)]

#4.1) Make some samples from the unconditioned ProMP
promp_samples = robot_promp.sample(sample_time)

#4.2) Condition the ProMP to start at q_cond_init and draw samples from it
q_cond_init = [1.54, 0.44, 0.15, 1.65, 0.01, -0.09, -1.23]
robot_promp.condition(t=0, T=1, q=q_cond_init, ignore_Sy=False)
cond_samples = robot_promp.sample(sample_time)
```

### Using task space conditioning

In order to use task space conditioning you need to implement the forward kinematics of your own robot.
You simply need to implement a class with a method called "position_and_jac(q)" that given a joint space
configuration q produces a tuple (x, jac, ori) representing respectively the cartesian position, the
jacobian and the orientation. We provide an example of the implementation of the kinematics of a Barrett
WAM arm. The following example shows how to condition a ProMP in task space using the Barrett forward
kinematics implementation.

```
import robpy.kinematics.forward as fwd

# Compute the prior distribution in joint space at the desired time
time_cartesian = 0.9
mean_marg_w, cov_marg_w = robot_promp.marginal_w(np.array([0.0,time_cartesian,1.0]), q=True)
prior_mu_q = mean_marg_w[1]
prior_Sigma_q = cov_marg_w[1]

# Compute the posterior distribution in joint space after conditioning in task space
fwd_kin = fwd.BarrettKinematics()
prob_inv_kin = promp.ProbInvKinematics(fwd_kin)

mu_cartesian = np.array([-0.62, -0.44, -0.34])
Sigma_cartesian = 0.02**2*np.eye(3) 

mu_q, Sigma_q = prob_inv_kin.inv_kin(mu_theta=prior_mu_q, sig_theta=prior_Sigma_q,
        mu_x = mu_cartesian, sig_x = Sigma_cartesian)

# Finally, condition in joint space using the posterior joint space distribution

robot_promp.condition(t=time_cartesian, T=1.0, q=mu_q, Sigma_q=Sigma_q, ignore_Sy=False)
task_cond_samples = robot_promp.sample(sample_time)
```

Publications
------------

The implementations provided in this repository are based on the following publications:

1) [Adaptation and Robust Learning of Probabilistic Movement Primitives](https://arxiv.org/pdf/1808.10648.pdf)
2) [Using probabilistic movement primitives for striking movements, IEEE RAS International 
Conference on Humanoid Robots, 2016](https://ieeexplore.ieee.org/abstract/document/7803322/)

Please refer to these papers to understand our implementation, get general information about
probabilistic movement primitives and see the evaluation of the implemented methods in real
robotic platforms. We also have a [C++ implementation](https://github.com/sebasutp/promp-cpp) of
the ProMP conditioning in joint and task space and other operators that might require real-time
execution performance.

