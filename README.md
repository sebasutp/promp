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

#### A very simple example

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

Please refer to the example folder in the repository. We mention some of the example files
here along with what exactly the example shows:

#### A toy example

The file "examples/python_promp/toy_example.py" 

Publications
------------

The implementations provided in this repository are based on the following publications:

1) [Adaptation and Robust Learning of Probabilistic Movement Primitives](https://arxiv.org/pdf/1808.10648.pdf)
2) [Using probabilistic movement primitives for striking movements, IEEE RAS International 
Conference on Humanoid Robots, 2016](https://ieeexplore.ieee.org/abstract/document/7803322/)

Please refer to these papers to understand our implementation, get general information about
probabilistic movement primitives and see the evaluation of the implemented methods in real
robotic platforms.


