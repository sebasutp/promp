""" Computes and plots the marginal joint distribution of a ProMP

This script computes, plots and optionally saves a distribution of
joint angles given a ProMP. If you wish to see the also plot recorded
distributions along with the marginal please provide the paths to the
data and meta-data as well.

"""

import robpy.full_promp as promp
import robpy.utils as utils
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import json

def plot_promp(promp, marginal, **params):
    """ Plots independently each of the outputs of the ProMP
    """
    params.setdefault("joint", 0)

    d = params['joint']
    dvel = d + promp.num_joints
    dist_z = marginal['time']
    means = np.array(marginal['means'])
    stdevs = np.array(map(lambda x: np.sqrt(np.diag(x)), marginal['covs']))

    tot_plots = 2 if params['plot_vel'] else 1
    fig = plt.figure()
    ax1 = fig.add_subplot(tot_plots,1,1)
    ax1.plot(dist_z, means[:,d], 'r')
    ax1.fill_between(dist_z, means[:,d]-2*stdevs[:,d], means[:,d]+2*stdevs[:,d], facecolor='red', alpha=0.3)
    if params['plot_vel']:
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(dist_z, means[:,dvel], 'r')
        ax2.fill_between(dist_z, means[:,dvel]-2*stdevs[:,dvel], means[:,dvel]+2*stdevs[:,dvel], facecolor='red', alpha=0.3)

    if 'data' in params and 'time' in params['data']:
        for ix,t in enumerate(params['data']['time']):
            Tn = len(t)
            T = (t[-1] - t[0])
            z = np.linspace(0,1,Tn)
            ax1.plot(z, params['data']['q'][ix][:,d], 'g')
            if params['plot_vel']:
                ax2.plot(z, T*params['data']['qd'][ix][:,d], 'g')
        ax1.set_ylabel('$q_{0}$'.format(params['joint']))
        ax1.set_xlabel('time')
        if params['plot_vel']:
            ax2.set_ylabel('$\dot{q}_{0}$'.format(params['joint']))
            ax2.set_xlabel('time')
    if params['tikz']:
        from matplotlib2tikz import save as tikz_save
        tikz_save(params['tikz'], figureheight = '\\figureheight', figurewidth = '\\figurewidth')
    plt.show()

def main(args):
    data = {}
    if args.data:
        with open(args.data,'r') as f:
            ldata = np.load(f)
            data['time'] = ldata['time']
            data['q'] = ldata['Q']
            data['qd'] = ldata['Qdot']
            if args.start_ix or args.end_ix:
                a = args.start_ix if args.start_ix is not None else 0
                b = args.end_ix if args.end_ix is not None else len(data['time'])
                for k in data:
                    data[k] = data[k][a:b]
    model = promp.FullProMP.load(args.promp)
    mlh = model.log_likelihood(data['time'], q=data['q']) 
    print("Avg. Log-Likelihood: {}".format(mlh / len(data['time'])))
    dist_z = np.linspace(0,1,50)
    means,covs = model.marginal_w(dist_z, q=True, qd=True)
    marginal = {'time': dist_z, 'means': means, 'covs': covs}
    if args.save_dist:
        with open(args.save_dist,'w') as f: 
            json.dump(utils.numpy_serialize(marginal), f)
    if args.plot_marg:
        for d in xrange(7):
            plot_promp(model, marginal, joint=d, tikz=args.tikz, plot_vel=args.plot_vel, data=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('promp', help="Path to the stored JSON ProMP file")
    parser.add_argument('--data', help="File with the trajectories used for training "
            "as a dictionary with keys {time, Q, Qdot}")
    parser.add_argument('--start_ix', type=int, help="From which data index")
    parser.add_argument('--end_ix', type=int, help="To which data index")
    parser.add_argument('--plot_marg', action="store_true", help="Plot marginal distributions")
    parser.add_argument('--plot_vel', action="store_true", help="Plot the velocities and the learned distribution")
    parser.add_argument('--save_dist', help="File where the computed marginal distribution is stored")
    parser.add_argument('--tikz', help="Name of a latex file to export the produced figure")
    args = parser.parse_args()
    main(args)
