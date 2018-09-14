""" Train a Probabilistic Movement Primitive

This script can load JSON data and meta data to train a ProMP.
"""

import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import json
import logging

def trainFullPromp(time, Q, Qdot=None, plot_likelihoods=True, **args):
    args.setdefault('print_lowerbound', plot_likelihoods)
    args.setdefault('print_inner_lb', False)
    args.setdefault('max_iter', 10)
    if 'init_promp' in args:
        robot_promp = promp.FullProMP.load(args['init_promp'])
    else:
        robot_promp = promp.FullProMP(basis=args['new_kernel'])
    if Qdot is not None:
        train_summary = robot_promp.train(time, q=Q, qd=Qdot, **args)
    else:
        train_summary = robot_promp.train(time, q=Q, **args)
    np.set_printoptions(precision=4, linewidth=200)
    print "Mean Weights:\n", robot_promp.mu_w
    print "Stdev Weights:\n", np.sqrt(np.diag(robot_promp.Sigma_w))
    print "Noise stdev:\n", np.sqrt(np.diag(robot_promp.Sigma_y))
    print "Basis function params: ", robot_promp.get_basis_pars()
    if plot_likelihoods:
        lhoods = train_summary['likelihoods']
        #lhoods -= lhoods[0] #Plot improvement from first iteration
        plt.plot(lhoods)
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood')
        if args['save_lh_plot']:
            tikz_save(args['save_lh_plot'],
                figureheight = '\\figureheight',
                figurewidth = '\\figurewidth')
        plt.show()
    stream = robot_promp.to_stream()
    fout = file(args['model_fname'], 'w')
    json.dump(stream, fout)
    return robot_promp

def main(args):
    logging.basicConfig(filename='/tmp/promp_train.log',level=logging.DEBUG)
    with open(args.data,'r') as f:
        data = np.load(args.data)
        time = data['time']
        Q = data['Q']
        Qdot = data['Qdot']
    durations = [t[-1]-t[0] for t in time]
    dm = np.mean(durations)
    dstd = np.std(durations)
    print("Durations {} +/- {}".format(dm, dstd))

    if args.trajlim:
        time = time[0:args.trajlim]
        Q = Q[0:args.trajlim]
        Qdot = Qdot[0:args.trajlim]

    full_basis = {
            'conf': [
                    {"type": "sqexp", "nparams": 4, "conf": {"dim": 3}},
                    {"type": "poly", "nparams": 0, "conf": {"order": 1}}
                ],
            'params': [np.log(0.25),0.25,0.5,0.75]
            }
    dof = np.shape(Q[0])[1]
    dim_basis_fun = promp.dim_comb_basis(**full_basis)
    w_dim = dof*dim_basis_fun
    inv_whis_mean = lambda v, Sigma: utils.make_block_diag(Sigma, dof) + args.pdiag_sw*np.eye(w_dim)
    params = {
            'new_kernel': full_basis,
            #'prior_mu_w': {"m0": np.zeros(5*7), "k0": 1},
            'model_fname': "/tmp/promp.json",
            'diag_sy': not args.full_sy,
            'joint_indep': args.joint_indep,
            'use_velocity': args.use_vel,
            'opt_basis_pars': False,
            'print_inner_lb': False,
            'no_Sw': False, #Approx E-Step with Dirac delta?
            'num_joints': dof,
            'max_iter': args.training_iter,
            'save_lh_plot': args.save_lh_plot,
            'init_params': {'mu_w': np.zeros(w_dim), 
                'Sigma_w': 1e8*np.eye(w_dim),
                'Sigma_y': np.eye(dof)},
            'print_inner_lb': False
            }
    if args.init_promp:
        params['init_promp'] = args.init_promp
    if not args.max_lh:
        params['prior_Sigma_w'] = {'v':dim_basis_fun*dof, 'mean_cov_mle': inv_whis_mean}
    if args.use_vel:
        model = trainFullPromp(time, Q, Qdot, **params)
    else:
        model = trainFullPromp(time, Q, **params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', help="File with the trajectories used for training "
            "as a dictionary with keys {time, Q, Qdot}")
    parser.add_argument('--full_sy', action="store_true", help="Make the noise matrix full rank instead of diagonal")
    parser.add_argument('--joint_indep', action="store_true", help="Make a ProMP with independent joints")
    parser.add_argument('--use_vel', action="store_true", help="Should the velocity be used to train the ProMP?")
    parser.add_argument('--max_lh', action="store_true", help="Do not use priors")
    parser.add_argument('--trajlim', type=int, help="Use for training only the given number of trajectories,"
            " use it to split training and validation or to test training with fewer data examples")
    parser.add_argument('--init_promp', help="Optional starting values for the ProMP parameters")
    parser.add_argument('--training_iter', default=10, type=int, help="Number of training iterations")
    parser.add_argument('--pdiag_sw', type=float, default=1e-4, help="Prior diagonal element for Sigma_w")
    parser.add_argument('--save_lh_plot', help="File where the likelihood w.r.t iterations plot should be saved")

    args = parser.parse_args()
    main(args)
