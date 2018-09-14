""" Toy example of a Probabilistic Movement Primitive

"""

import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import json


full_basis = {
        'conf': [
                {"type": "sqexp", "nparams": 6, "conf": {"dim": 5}},
                {"type": "poly", "nparams": 0, "conf": {"order": 0}}
            ],
        'params': [np.log(0.1),0.0,0.25,0.5,0.75,1.0]
        }
dim_basis_fun = promp.dim_comb_basis(**full_basis)
dof = 1
w_dim = dof*dim_basis_fun
test_mu_w = np.array([-10,20,-12,15,-13,-5])
test_sig_w = 9*np.eye(w_dim)

inv_whis_mean = lambda v, Sigma: np.diag(np.diag(Sigma))
params = {
        'new_kernel': full_basis,
        #'prior_mu_w': {"m0": np.zeros(5*7), "k0": 1},
        #'prior_Sigma_w': {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean},
        'model_fname': "/tmp/promp.json",
        'diag_sy': True,
        'opt_basis_pars': False,
        'print_inner_lb': True,
        'no_Sw': False, #Approx E-Step with Dirac delta?
        'num_joints': dof,
        'max_iter': 30,
        'init_params': {'mu_w': np.zeros(w_dim), 
            'Sigma_w': 1e8*np.eye(w_dim),
            'Sigma_y': np.eye(dof)}
        }

def create_toy_data(n=100, T=30, missing_obs=[40,40]):
    p = promp.FullProMP(model={'mu_w': test_mu_w,
        'Sigma_w': test_sig_w,
        'Sigma_y': np.eye(1)}, num_joints=1, basis=full_basis)
    times = [np.linspace(0,1,T) for i in xrange(n)]
    if missing_obs is not None:
        for i in range(missing_obs[0]): times[i] = np.delete(times[i], range(1,T/2))
        for i in range(missing_obs[1]): times[i+missing_obs[0]] = np.delete(times[i+missing_obs[0]], range(T/2,T-1))
        #times = [np.delete(times[i], range(1 + (i % (T/2)),T/2 - 1 + (i % (T/2)))) for i in range(n)]
    Phi = []
    X = p.sample(times, Phi=Phi, q=True)
    return times, Phi, X

def trivial_train(times, Phi, X):
    W = []
    for i, phi in enumerate(Phi):
        y = X[i]
        phi_a = np.array(phi)
        w = np.dot(np.linalg.pinv(phi_a[:,0,:]),y[:,0])
        W.append(w)
    mu_w = np.mean(W,axis=0)
    Sigma_w = np.cov(W, rowvar=0)
    return mu_w, Sigma_w

def promp_train(times, X):
    p = promp.FullProMP(num_joints=1, basis=full_basis)
    p.train(times=times, q=X, **params)
    return p.mu_w, p.Sigma_w

times, Phi, X = create_toy_data()

for i,t in enumerate(times):
    plt.plot(t, X[i])
plt.show()

mu_w_t, sig_w_t = trivial_train(times, Phi, X)
mu_w_p, sig_w_p = promp_train(times, X)

print "mu_w=", test_mu_w
print "mu_w_trivial=", mu_w_t
print "mu_w_em=", mu_w_p

print "sig_w=", test_sig_w
print "sig_w_trivial=", sig_w_t
print "sig_w_em=", sig_w_p
