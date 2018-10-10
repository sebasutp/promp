""" Probabilistic Movement Primitive module
"""

import scipy.optimize as opt
import scipy.linalg
import autograd.numpy as np
import autograd
import json
import robpy.utils as utils

def joint_train_data(tt_data):
    """ Extracts the joint training data for ProMPs

    This method takes a table tennis training set builder object and
    returns the joint angles W and velocities Wdot required by the
    ProMP class.
    """
    data = tt_data.strike_trajectories()
    time = []
    W = []
    Wdot = []
    for instance in data:
        time_n = []
        Wn = []
        Wn_dot = []
        for elem in instance:
            time_n.append(elem["t"])
            Wn.append(elem["q"])
            Wn_dot.append(elem["qd"])
        time.append(time_n)
        W.append(np.array(Wn))
        Wdot.append(np.array(Wn_dot))
    return (time, W, Wdot)

def sqexp(t, params, **args):
    """ A set of radial basis functions in one dimension
    """
    sigma_sq = np.exp(params[0])**2
    centers = params[1:]
    ans = np.exp(-0.5*(t - centers)**2 / sigma_sq)
    return ans

def poly(t, params, **args):
    """ Polynomial with order equal to dim-1
    """
    order = args['conf']['order']
    basis_f = map(lambda ix: t**ix, range(order+1))
    return np.array(basis_f)

def comb_basis(t, params, **args):
    basis = {"sqexp": sqexp, "poly": poly}
    conf = args['conf']
    ans = []
    start = 0
    for c in conf:
        end = start + c['nparams']
        ans.append(basis[c['type']](t, params[start:start+end], conf=c['conf']))
        start = end
    return np.concatenate(ans)

def dim_comb_basis(**args):
    bdim = {'poly': lambda x: x['order'] + 1, 'sqexp': lambda x: x['dim']}
    conf = args['conf']
    dim = 0
    for c in conf:
        dim = dim + bdim[c['type']](c['conf'])
    return dim

def quad(a,X):
    """ Computes a quadratic form as a^T X a
    """
    return np.dot(a, np.dot(X, a))

def cov_mat_precomp(cov_mat):
    tmp, log_det = np.linalg.slogdet(cov_mat)
    return {'inv': np.linalg.inv(cov_mat), 
            'log_det': log_det}

def lambda_debug(f, x, name):
    print "Evaluating {1}({0})".format(x,name)
    ans = f(x)
    print "Ans=", ans
    return ans

def get_bfun_lambdas(basis_params, basis_fun, q=False, qd=False):
    f = lambda z: basis_fun(z, basis_params)
    bfun = {}
    if q:
        bfun['fpos'] = f
    if qd:
        bfun['fvel'] = autograd.jacobian(f)
    return bfun

def get_Phi_t(t, T, num_joints=1, pos=None, vel=None, acc=None):
    """ Builds the matrix Phi_t with the same format as the C++ implementation
    """
    assert t>=0 and t<=T
    vel_fac = 1.0/T
    pos_t = []
    vel_t = []
    acc_t = []
    for d in xrange(num_joints):
        if pos is not None: pos_t.append( pos )
        if vel is not None: vel_t.append( vel_fac * vel )
        if acc is not None: acc_t.append( vel_fac**2 * acc )
    ans = []
    if pos is not None: ans.append(scipy.linalg.block_diag(*pos_t))
    if vel is not None: ans.append(scipy.linalg.block_diag(*vel_t))
    if acc is not None: ans.append(scipy.linalg.block_diag(*acc_t))
    return np.concatenate(ans, axis=0)

def comp_Phi_t(t, T, num_joints=1, fpos=None, fvel=None, facc=None):
    """ Builds the matrix Phi_t with the same format as the C++ implementation
    """
    vals = {}
    if fpos is not None: vals['pos'] = fpos(t/T)
    if fvel is not None: vals['vel'] = fvel(t/T)
    if facc is not None: vals['acc'] = facc(t/T)
    return get_Phi_t(t,T,num_joints,**vals)

def get_Phi(self, times, bfun):
    """ Computes a list with all the matrices Phi_t
    """
    Phi = []
    for time in times:
        Tn = len(time)
        duration = time[-1] - time[0]
        Phi_n = []
        for t in xrange(Tn):
            curr_time = time[t] - time[0]
            phi_nt = self.__comp_Phi_t(curr_time, duration, **bfun)
            Phi_n.append(phi_nt)
        Phi.append(Phi_n)
    return Phi

def get_y_t(self, q=None, qd=None, qdd=None):
    """ Builds the vector y_t to be compatible with the matrix Phi_t 

    This method builds a vector y_t with any valid combination of 
    joint position, velocity and acceleration. 
    """
    y_t = []
    if 'q' in params: y_t.extend(params['q'])
    if 'qd' in params: y_t.extend(params['qd'])
    if 'qdd' in params: y_t.extend(params['qdd'])
    return np.array(y_t)

def __get_Y(self, times, **args):
    Y = []
    N = len(times)
    for n in xrange(N):
        y_n = []
        for t in xrange(len(times[n])):
            inst = {}
            if 'q' in args:
                inst['q'] = args['q'][n][t,:]
            if 'qd' in args:
                inst['qd'] = args['qd'][n][t,:]
            y_n.append(self.__get_y_t(**inst))
        Y.append(np.array(y_n))
    return Y

class FullProMP:

    def __get_bfun_lambdas(self, **args):
        args.setdefault('basis_params', self.get_basis_pars())
        args.setdefault('basis_fun', self.basis_fun)
        q = False if 'q' in args and isinstance(args['q'],bool) and not args['q'] else True
        qd = args['qd'] if 'qd' in args else False
        return get_bfun_lambdas(args['basis_params'], args['basis_fun'], q=q, qd=qd)

    def __get_bfun_grad_lambdas(self, **args):
        pass

    def __get_Phi_t(self, t, T, **args):
        """ Builds the matrix Phi_t with the same format as the C++ implementation
        """
        assert t>=0 and t<=T
        vel_fac = 1.0/T
        pos_t = []
        vel_t = []
        acc_t = []
        for d in xrange(self.num_joints):
            if 'pos' in args: pos_t.append( args['pos'] )
            if 'vel' in args: vel_t.append( vel_fac * args['vel'] )
            if 'acc' in args: acc_t.append( vel_fac**2 * args['acc'] )
        ans = []
        if 'pos' in args: ans.append(scipy.linalg.block_diag(*pos_t))
        if 'vel' in args: ans.append(scipy.linalg.block_diag(*vel_t))
        if 'acc' in args: ans.append(scipy.linalg.block_diag(*acc_t))
        return np.concatenate(ans, axis=0)

    def __comp_Phi_t(self, t, T, **args):
        """ Builds the matrix Phi_t with the same format as the C++ implementation
        """
        vals = {}
        if 'fpos' in args: vals['pos'] = args['fpos'](t/T)
        if 'fvel' in args: vals['vel'] = args['fvel'](t/T)
        if 'facc' in args: vals['acc'] = args['facc'](t/T)
        return self.__get_Phi_t(t,T,**vals)

    def __get_Phi(self, times, **args):
        """ Builds a list with all the matrices Phi_t already pre-computed
        """
        if not 'basis_params' in args and self.__Phi: 
            return self.__Phi
        args.setdefault('bfun', self.__get_bfun_lambdas(**args))
        bfun = args['bfun']
        Phi = []
        for time in times:
            Tn = len(time)
            duration = time[-1] - time[0]
            Phi_n = []
            for t in xrange(Tn):
                curr_time = time[t] - time[0]
                phi_nt = self.__comp_Phi_t(curr_time, duration, **bfun)
                Phi_n.append(phi_nt)
            Phi.append(Phi_n)
        if not 'basis_params' in args:
            self.__Phi = Phi
        return Phi

    def get_Phi(self, times, **args):
        return self.__get_Phi(times, **args)

    def __get_y_t(self, **params):
        """ Builds the vector y_t to be compatible with the matrix Phi_t 

        This method builds a vector y_t with any valid combination of 
        joint position, velocity and acceleration. 
        """
        y_t = []
        if 'q' in params: y_t.extend(params['q'])
        if 'qd' in params: y_t.extend(params['qd'])
        if 'qdd' in params: y_t.extend(params['qdd'])
        return np.array(y_t)

    def __get_Y(self, times, **args):
        Y = []
        N = len(times)
        for n in xrange(N):
            y_n = []
            for t in xrange(len(times[n])):
                inst = {}
                if 'q' in args:
                    inst['q'] = args['q'][n][t,:]
                if 'qd' in args:
                    inst['qd'] = args['qd'][n][t,:]
                y_n.append(self.__get_y_t(**inst))
            Y.append(np.array(y_n))
        return Y

    def get_Y(self, times, **args):
        return self.__get_Y(times, **args)

    def set_internal_params(self, **args):
        if 'mu_w' in args: self.mu_w = args['mu_w']
        if 'Sigma_w' in args: self.Sigma_w = args['Sigma_w']
        if 'Sigma_y' in args: self.Sigma_y = args['Sigma_y']
    
    def __init__(self, **args):
        args.setdefault("basis", {
            "conf": [
                {"type": "sqexp", "nparams": 4, "conf": {"dim": 3}},
                {"type": "poly", "nparams": 0, "conf": {"order": 1}}
                ],
            "params": np.array([np.log(0.25),0.0,0.5,1.0])
                })
        self.__basis_conf = args['basis']['conf']
        self.set_basis_pars( args['basis']['params'] )
        if 'basis_fun' in args: 
            self.basis_fun = args['basis_fun']
        else:
            self.basis_fun = lambda t,params: comb_basis(t, params, conf=args["basis"]["conf"])
        if 'num_joints' in args: self.num_joints = args['num_joints']
        if 'model' in args:
            self.mu_w = np.array(args['model']['mu_w'])
            self.Sigma_w = np.array(args['model']['Sigma_w'])
            self.Sigma_y = np.array(args['model']['Sigma_y'])

    def get_basis_pars(self):
        return self.__basis_pars

    def set_basis_pars(self, basis_pars):
        self.__basis_pars = np.array(basis_pars)
        self.__Phi = None

    def __em_lb_likelihood(self, times, Y, expectations, **args):
        #1) Set default values to some variables
        args.setdefault('Sigma_w', self.Sigma_w)
        args.setdefault('Sigma_y', self.Sigma_y)
        args.setdefault('Sigma_w_val', cov_mat_precomp(args['Sigma_w']))
        args.setdefault('Sigma_y_val', cov_mat_precomp(args['Sigma_y']))
        args.setdefault('mu_w', self.mu_w)
        #2) Load values in some variables
        inv_sig_w = args['Sigma_w_val']['inv']
        log_det_sig_w = args['Sigma_w_val']['log_det']
        inv_sig_y = args['Sigma_y_val']['inv']
        log_det_sig_y = args['Sigma_y_val']['log_det']
        mu_w = args['mu_w']
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        Phi = self.__get_Phi(times, **args)
        #3) Actually compute lower bound
        ans = 0.0
        for n in xrange(len(times)):
            Tn = len(times[n])
            lpw = log_det_sig_w + np.trace(np.dot(inv_sig_w,w_covs[n])) + quad(w_means[n]-mu_w, inv_sig_w)
            lhood = 0.0
            for t in xrange(Tn):
                phi_nt = Phi[n][t] 
                y_nt = Y[n][t]
                lhood = lhood + log_det_sig_y + quad(y_nt-np.dot(phi_nt,w_means[n]),inv_sig_y) + \
                        np.trace(np.dot(inv_sig_y, np.dot(phi_nt, np.dot(w_covs[n], phi_nt.T))))
            ans = ans + lpw + lhood
        return -0.5*ans

    def __em_lb_grad_basis_pars(self, times, Y, expectations, **args):
        #1) Set default values to some variables
        args.setdefault('Sigma_y', self.Sigma_y)
        args.setdefault('Sigma_y_val', cov_mat_precomp(args['Sigma_y']))
        #2) Load values in some variables
        inv_sig_y = args['Sigma_y_val']['inv']
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        bfun = args['bfun']
        Phi = self.__get_Phi(times, **args)
        pars = args['basis_params']
        #3) Actually compute the gradient
        for n in xrange(len(times)):
            Tn = len(times[n])
            for t in xrange(Tn):
                #3.1) Compute the derivative of lower-bound w.r.t phi_nt
                A = np.outer( np.dot(Phi[n][t], w_means[n]), w_means[n] ) + \
                        np.dot(Phi[n][t], w_covs[n]) - np.outer(Y[n][t], w_means[n])
                d_lb_phi_nt = 2*np.dot(inv_sig_y, A)
                #3.1) Compute the derivatives of phi_nt w.r.t each parameter
                for d in xrange(len(pars)):
                    pass



    def __EM_lowerbound(self, times, Y, expectations, **args):
        """ Computes the EM lowerbound
        Receives a list of time vectors from the training set, the expectations computed in the
        E-step of the algorithm, and a list of optional arguments. As an optional argument eigther
        the angle positions, velocities or accelerations of the training set should be included.
        The optional arguments can also specify any of the parameters that are being optimized as
        a special value.
        """
        #1) Load default values
        args.setdefault('Sigma_w', self.Sigma_w)
        args.setdefault('Sigma_y', self.Sigma_y)
        args.setdefault('Sigma_w_val', cov_mat_precomp(args['Sigma_w']))
        args.setdefault('mu_w', self.mu_w)
        #2) Load useful variables (Including likelihood)
        inv_sig_w = args['Sigma_w_val']['inv']
        log_det_sig_w = args['Sigma_w_val']['log_det']
        lhood_lb = self.__em_lb_likelihood(times, Y, expectations, **args)
        #3) Compute prior log likely-hood
        lprior = 0.0
        if 'prior_mu_w' in args:
            m0 = args['prior_mu_w']['m0']
            inv_V0 = args['prior_mu_w']['k0']*inv_sig_w #Normal-Inverse-Wishart prior
            lprior = lprior + quad(args['mu_w']-m0, inv_V0)
        if 'prior_Sigma_w' in args:
            prior_Sigma_w = args['prior_Sigma_w']
            v0 = prior_Sigma_w['v']
            D = np.shape(self.Sigma_w)[0]
            if 'mean_cov_mle' in prior_Sigma_w:
                S0 = prior_Sigma_w['mean_cov_mle'](v0, self.__Sigma_w_mle) * (v0 + D + 1)
            else:
                S0 = prior_Sigma_w['invS0']
            lprior = lprior + (v0 + D + 1)*log_det_sig_w + np.trace(np.dot(S0, inv_sig_w))
        #4) Compute full lower bound
        return -0.5*lprior + lhood_lb

    def __E_step(self, times, Y, **args):
        #1) Set up default values
        args.setdefault('Sigma_w', self.Sigma_w)
        args.setdefault('Sigma_y', self.Sigma_y)
        args.setdefault('Sigma_w_val', cov_mat_precomp(args['Sigma_w']))
        args.setdefault('Sigma_y_val', cov_mat_precomp(args['Sigma_y']))
        args.setdefault('mu_w', self.mu_w)
        #2) Load some variables
        inv_sig_w = args['Sigma_w_val']['inv']
        inv_sig_y = args['Sigma_y_val']['inv']
        mu_w = args['mu_w']
        Phi = self.__get_Phi(times, **args)
        #3) Compute expectations
        w_means = []
        w_covs = []
        for n,time in enumerate(times):
            Tn = len(Y[n])
            sum_mean = np.dot(inv_sig_w, mu_w)
            sum_cov = inv_sig_w
            for t in xrange(Tn):
                phi_nt = Phi[n][t]
                tmp1 = np.dot(np.transpose(phi_nt),inv_sig_y)
                sum_mean = sum_mean + np.dot(tmp1, Y[n][t])
                sum_cov = sum_cov + np.dot(tmp1, phi_nt)
            Swn = utils.force_sym(np.linalg.inv(sum_cov))
            wn = np.dot(Swn, sum_mean)
            w_means.append(wn)
            w_covs.append(Swn)
        return {'w_means': w_means, 'w_covs': w_covs}

    def E_step(self, times, Y, **args):
        return self.__E_step(times, Y, **args)

    def __M_step(self, times, Y, expectations, **args):
        Phi = self.__get_Phi(times, **args)
        N = len(times)
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        n_var = lambda X: sum(map(lambda x: np.outer(x,x), X))

        #1) Optimize mu_w
        wn_sum = sum(w_means)
        if 'prior_mu_w' in args: 
            prior_mu_w = args['prior_mu_w']
            mu_w = (wn_sum + prior_mu_w['k0']*prior_mu_w['m0'])/(N + prior_mu_w['k0'])
        else: 
            mu_w = (wn_sum) / N

        #2) Optimize Sigma_w
        diff_w = map(lambda x: x - mu_w, w_means)
        if 'no_Sw' in args and args['no_Sw']==True:
            sw_sum = utils.force_sym(n_var(diff_w))
        else:
            sw_sum = utils.force_sym(sum(w_covs) + n_var(diff_w))

        self.__Sigma_w_mle = sw_sum / N  # Maximum likelyhood estimate for Sigma_w
        if 'prior_Sigma_w' in args:
            prior_Sigma_w = args['prior_Sigma_w']
            v0 = prior_Sigma_w['v']
            D = np.shape(self.Sigma_w)[0]
            if 'mean_cov_mle' in prior_Sigma_w:
                S0 = prior_Sigma_w['mean_cov_mle'](v0, self.__Sigma_w_mle) * (v0 + D + 1)
            else:
                S0 = prior_Sigma_w['invS0']
            Sigma_w = (S0 + sw_sum) / (N + v0 + D + 1)
        else:
            Sigma_w = self.__Sigma_w_mle

        #3) Optimize Sigma_y
        diff_y = []
        uncert_w_y = []
        for n in xrange(N):
            for t in xrange(len(times[n])):
                diff_y.append(Y[n][t] - np.dot(Phi[n][t], w_means[n]))
                uncert_w_y.append(np.dot(np.dot(Phi[n][t],w_covs[n]),Phi[n][t].T))
        if 'no_Sw' in args and args['no_Sw']==True:
            Sigma_y = (n_var(diff_y)) / len(diff_y)
        else:
            Sigma_y = (n_var(diff_y) + sum(uncert_w_y)) / len(diff_y)

        #4) Update
        self.mu_w = mu_w
        if args['print_inner_lb']:
            print 'lb(mu_w)=', self.__EM_lowerbound(times, Y, expectations, **args)

        self.Sigma_w = utils.force_sym(Sigma_w)
        if args['joint_indep']: self.Sigma_w = utils.make_block_diag(self.Sigma_w, args['num_joints'])
        if args['print_inner_lb']:
            print 'lb(Sigma_w)=', self.__EM_lowerbound(times, Y, expectations, **args)

        if args['diag_sy']:
            self.Sigma_y = np.diag(np.diag(Sigma_y))
        else:
            self.Sigma_y = utils.force_sym(Sigma_y)
        if args['print_inner_lb']:
            print 'lb(Sigma_y)=', self.__EM_lowerbound(times, Y, expectations, **args)

        #5) Update optional parameters
        if args['opt_basis_pars']:
            obj = lambda pars: -self.__em_lb_likelihood(times, Y, expectations, mu_w=mu_w, \
                    Sigma_w=Sigma_w, Sigma_y=Sigma_y, basis_params=pars, q=True)
            obj_debug = lambda x: lambda_debug(obj, x, "cost")
            jac = autograd.grad(obj)
            #print "Objective at x0: ", obj(self.get_basis_pars())
            #print "Gradient at x0: ", jac(self.get_basis_pars())
            #o_basis_pars = opt.minimize(lambda x: lambda_debug(obj,x,"cost"), self.get_basis_pars(), method="CG", jac=lambda x: lambda_debug(jac,x,"grad"))
            o_basis_pars = opt.minimize(obj, self.get_basis_pars(), method="Powell")
            #o_basis_pars = opt.minimize(obj, self.get_basis_pars(), method="Nelder-Mead")
            if o_basis_pars.success:
                self.set_basis_pars(o_basis_pars.x)
            else:
                print "Warning: The optimization of the basis parameters failed. Message: ", o_basis_pars.message
            if args['print_inner_lb']:
                print 'lb(basis_params)=', self.__EM_lowerbound(times, Y, expectations, **args)

    def __EM_training(self, times, **args):
        #1) Initialize state before training
        args.setdefault('opt_basis_pars', False)
        args.setdefault('max_iter', 10)
        args.setdefault('print_lowerbound', False)
        args.setdefault('print_inner_lb', False)
        args.setdefault('print_inner_params', False)
        args.setdefault('diag_sy', True)
        args.setdefault('joint_indep', False)
        args.setdefault('bfun', self.__get_bfun_lambdas(**args))
        if args['opt_basis_pars']:
            args.setdefault('bfun_grad', self.__get_bfun_grad_lambdas(**args))
        Y = self.__get_Y(times, **args)
        bfun = args['bfun']
        if 'q' in args:
            self.num_joints = np.shape(args['q'][0])[1]
        elif 'qd' in args:
            self.num_joints = np.shape(args['qd'][0])[1]
        else:
            raise AttributeError("The positions q or velocities qd are expected but not found")
        y_dim, w_dim = np.shape( self.__comp_Phi_t(0.0,1.0,**bfun) )
        is_initalized = hasattr(self, 'mu_w') and hasattr(self,'Sigma_w') and hasattr(self,'Sigma_y')
        if not is_initalized:
            #Only set internal parameters if they are missing
            args.setdefault('init_params', {'mu_w': np.zeros(w_dim), 
                'Sigma_w': np.eye(w_dim),
                'Sigma_y': np.eye(y_dim)})
            self.set_internal_params(**args['init_params'])
        self.__Sigma_w_mle = self.Sigma_w

        likelihoods = []
        #2) Train
        for it in xrange(args['max_iter']):
            expectations = self.__E_step(times, Y, **args)
            if ('early_quit' in args and args['early_quit']()): break
            if args['print_lowerbound']: 
                lh = self.__EM_lowerbound(times, Y, expectations, **args)
                print 'E-step LB:', lh
                likelihoods.append(lh)
            self.__M_step(times, Y, expectations, **args)
            if ('early_quit' in args and args['early_quit']()): break
            if args['print_lowerbound']: 
                print 'M-step LB:', self.__EM_lowerbound(times, Y, expectations, **args)
        return {'likelihoods': likelihoods}

    def log_likelihood(self, times, Sigma_w_mle=None, **args):
        self.__Sigma_w_mle = self.Sigma_w if Sigma_w_mle is None else Sigma_w_mle
        Y =  self.__get_Y(times, **args)
        expectations = self.__E_step(times, Y, **args)
        return self.__EM_lowerbound(times, Y, expectations, **args)

    def train(self, times, **args):
        return self.__EM_training(times, **args)

    def sample(self, times, Phi=None, **args):
        N = len(times)
        args.setdefault('weights', np.random.multivariate_normal(self.mu_w, self.Sigma_w, N))
        if not 'noise' in args:  args['noise'] = np.diag(self.Sigma_y)
        W = args['weights']
        noise = args['noise']
        _Phi = self.__get_Phi(times, basis_params=self.get_basis_pars(), **args)
        if Phi is None:
            Phi = _Phi 
        elif isinstance(Phi,list) and len(Phi)==0:
            Phi.extend(_Phi)
        ans = []
        for n in xrange(N):
            Tn = len(times[n])
            w = W[n]
            curr_sample = []
            for t in xrange(Tn):
                y = np.dot(Phi[n][t], w) + np.multiply(np.random.standard_normal(len(noise)), noise)
                curr_sample.append(y)
            ans.append(np.array(curr_sample))
        return ans

    def marginal_w(self, time, **args):
        phi_n = self.__get_Phi([time], basis_params=self.get_basis_pars(), **args)[0]
        means = []
        covs = []
        for phi_nt in phi_n:
            means.append(np.dot(phi_nt, self.mu_w))
            covs.append(np.dot(np.dot(phi_nt,self.Sigma_w),phi_nt.T))
        return means, covs

    def to_stream(self):
        model = {"mu_w": self.mu_w.tolist(), "Sigma_w": self.Sigma_w.tolist(), "Sigma_y": self.Sigma_y.tolist()}
        basis = {"conf": self.__basis_conf, "params": self.get_basis_pars().tolist()}
        ans = {"model": model, "basis": basis, "num_joints": self.num_joints}
        return ans

    def condition(self, t, T, q, Sigma_q=None, ignore_Sy = True):
        """ Conditions the ProMP

        Condition the ProMP to pass be at time t with some desired position and velocity. If there is
        uncertainty on the conditioned point pass it as the optional matrices Sigma_q,
        Sigma_qd.
        """
        times = [[0,t,T]]
        _Phi = self.__get_Phi(times, basis_params=self.get_basis_pars())
        phi_t = _Phi[0][1]
        d,lw = phi_t.shape
        mu_q = self.__get_y_t(q=q)
        if ignore_Sy:
            tmp1 = np.dot(self.Sigma_w, phi_t.T)
            tmp2 = np.dot(phi_t, np.dot(self.Sigma_w, phi_t.T))
            tmp2 = np.linalg.inv(tmp2)
            tmp3 = np.dot(tmp1,tmp2)
            mu_w = self.mu_w + np.dot(tmp3, (mu_q - np.dot(phi_t, self.mu_w)))
            tmp4 = np.eye(d)
            if Sigma_q is not None:
                tmp4 -= np.dot(Sigma_q, tmp2)
            Sigma_w = self.Sigma_w - np.dot(tmp3, np.dot(tmp4, tmp1.T))
        else:
            inv_Sig_w = np.linalg.inv(self.Sigma_w)
            inv_Sig_y = np.linalg.inv(self.Sigma_y)
            Sw = np.linalg.inv(inv_Sig_w + np.dot(phi_t.T,np.dot(inv_Sig_y, phi_t)))
            A = np.dot(np.dot(Sw, phi_t.T), inv_Sig_y)
            b = np.dot(Sw, np.dot(inv_Sig_w, self.mu_w))
            mu_w = np.dot(A, mu_q) + b
            if Sigma_q is not None:
                Sigma_w = Sw + np.dot(A,np.dot(Sigma_q,A.T))
            else:
                Sigma_w = Sw

        self.mu_w = mu_w
        self.Sigma_w = Sigma_w


    @classmethod
    def load(cls, f_name):
        f = open(f_name, 'r')
        args = json.load(f)
        f.close()
        return cls(**args)


class ProbInvKinematics:
    #params:
    #fwd_k: A forward kinematics object

    def __laplace_cost_and_grad(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x):
        f_th, jac_th, ori = self.fwd_k.position_and_jac(theta)
        jac_th = jac_th[0:3,:]
        diff1 = theta - mu_theta
        tmp1 = np.dot(inv_sigma_theta, diff1)
        diff2 = f_th - mu_x
        tmp2 = np.dot(inv_sigma_x, diff2)
        nll = 0.5*(np.dot(diff1,tmp1) + np.dot(diff2,tmp2))
        grad_nll = tmp1 + np.dot(jac_th.T,tmp2)
        return nll, grad_nll

    def __init__(self, fwd_kinematics):
        self.fwd_k = fwd_kinematics

    def inv_kin(self, mu_theta, sig_theta, mu_x, sig_x):
        inv_sig_theta = np.linalg.inv(sig_theta)
        inv_sig_x = np.linalg.inv(sig_x)
        cost_grad = lambda theta: self.__laplace_cost_and_grad(theta, mu_theta, inv_sig_theta, mu_x, inv_sig_x)
        cost = lambda theta: cost_grad(theta)[0]
        grad = lambda theta: cost_grad(theta)[1]
        res = opt.minimize(cost, mu_theta, method='BFGS', jac=grad)
        post_mean = res.x
        post_cov = res.hess_inv
        return post_mean, post_cov
