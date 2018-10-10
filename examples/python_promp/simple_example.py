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
inv_whis_mean = lambda v, Sigma: 9e-1*utils.make_block_diag(Sigma, dof) + 1e-1*np.eye(dof*dim_basis_fun)
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

#5) An example of conditioning in Task space

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

#6) Plot all samples

for i in range(n_samples):
    plt.plot(sample_time[i], promp_samples[i][:,plot_dof], color='green')
    plt.plot(sample_time[i], cond_samples[i][:,plot_dof], color='blue')
    plt.plot(sample_time[i], task_cond_samples[i][:,plot_dof], color='red')
plt.title('Samples on DoF {} for the learned and conditioned ProMPs'.format(plot_dof))
plt.show()

