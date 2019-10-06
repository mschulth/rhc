import numpy as np
import casadi.casadi as cas


class Planning:

    def __init__(self, env, model):
        self.env = env
        self.model = model

        self.opt_maxiter = 500  # maximal number of iterations for the optimizer
        self.model_diff = True  # whether the model predicts the difference to the current state
        self.x_lim = None       # the state limits to impose

    def plan_multiple_shooting(self, t, cost, x0, x_lim=None, x_init=None):
        """
        Optimizes for a trajectory using multiple shooting method
        :param t: the horizon to plan for
        :param cost: the objective function. May be a casadi function of the form (cas: [s, u, v] -> cost) or a function
                  of the form t -> (cas: [si, ui, vi] -> cost), so it returns a casadi function given the time step.
        :param x0: the starting state
        :param x_lim: The limits for states (dimension [t, s_dim, 2] or [s_dim, 2] last index 0 is lower limit,
                      index 1 the upper. If none, self.x_lim is taken
        :param x_init: The inital guess for the trajectory. If None, uniformly random samples from [-2, 2] are taken.
        :return: The optimized trajectory actions, states, variances, and the reached cost
        """
        if x_lim is None:
            x_lim = self.x_lim

        s_dim = self.env.o_dim
        a_dim = self.env.a_dim
        x_dim = self.env.x_dim
        a_min = self.env.a_lim[:, 0]
        a_max = self.env.a_lim[:, 1]

        m = self.model.predict_casf(ret_var=True)
        x0 = cas.DM(np.atleast_2d(x0))
        xu = cas.MX.sym("x", t, x_dim)  # states and action variables to optimize for

        # full state and action variables
        x = cas.vcat((x0, xu[:, :s_dim]))
        u = xu[:, -a_dim:]

        # define standard bounds on variables (action bounds + no state bounds)
        xu_l = np.ones((t, x_dim)) * -np.inf
        xu_u = np.ones((t, x_dim)) * np.inf
        xu_l[:, -a_dim:] = a_min
        xu_u[:, -a_dim:] = a_max

        # include given task bounds if provided
        if x_lim is not None:
            if x_lim.ndim == 2:
                x_lim = np.tile(x_lim[None, :, :], (t, 1, 1))
            xu_l[:, :] = x_lim[:, :, 0]
            xu_u[:, :] = x_lim[:, :, 1]
            assert np.shape(x_lim) == (t, x_dim, 2)

        step_based_objective = not isinstance(cost, cas.Function)

        # equality constraints (model) and objective function
        obj = 0
        g = cas.MX()
        _, var_mx = m.mx_out()
        v = cas.MX()
        for i in range(t):
            xi = x[i, :]  # current state
            ui = u[i, :]  # current action

            pred_res = m(cas.hcat((xi, ui)))
            xj = pred_res[0]  # next state
            vi = pred_res[1]  # current predictive variance
            v = cas.vcat((v, vi))

            # update constraints
            gi = x[i+1, :] - xj
            if self.model_diff:
                gi -= xi
            g = cas.horzcat(g, gi)

            if step_based_objective and cost(i) is not None:
                obj += cost(i)(xi, ui, vi)

        # if episode-based reward, add the full cost
        if not step_based_objective:
            obj = cost(x, u, v)
        # otherwise add cost of last time step with zero action
        elif cost(t) is not None:
            obj += cost(t)(x[-1, :], 0, vi)

        # create NLP
        xu_flat = cas.reshape(xu, -1, 1)
        nlp = {'x': xu_flat, 'f': obj, 'g': g}
        solver = cas.nlpsol("solver", "ipopt", nlp, self.opt_dict)

        # set initial trajectory
        if x_init is not None:
            xopt_0 = cas.reshape(x_init, -1, 1)
        else:
            xopt_0 = (np.random.random_sample(xu_flat.shape[0]) - .5) * 4

        # solve nlp
        sol = solver(x0=xopt_0, lbx=xu_l.T.flatten(), ubx=xu_u.T.flatten(), lbg=0, ubg=0)
        opt_xu = np.array(cas.reshape(sol['x'], xu.shape[0], xu.shape[1]))
        opt_a = np.array(opt_xu[:, -a_dim:])
        opt_x = np.array(cas.vcat((x0, opt_xu[:, :s_dim])))
        cost = float(sol['f'])

        # compute variance of trajectory
        var_f = cas.Function("variance", [xu], [v])
        opt_varx = np.array(var_f(opt_xu))

        return opt_a, opt_x, opt_varx, cost

    @property
    def opt_dict(self):
        """
        Returs the dictionary for the casadi optimization
        """
        opts = {
            # "ipopt.print_level": 1,
            # "ipopt.sb": "yes",
            # "print_time": 0
            # "ipopt.derivative_test": "first-order"
        }
        if self.opt_maxiter is not None:
            opts['ipopt.max_iter'] = self.opt_maxiter
        else:
            opts = {}
        return opts
