from scipy.stats import qmc
import numpy as np
import GPy
import scipy

class RTRunningCost:
    def __init__(self, penalty=1):

        """
        Initialize the RTRunningCost with a penalty coefficient.

        Parameters:
        penalty (float): Penalty coefficient (scalar).
        """

        self.penalty = penalty

    def cost(self, B, Price, B_DA):

        """
        Calculate the real-time profit based on battery action, price, and state of charge.

        Parameters:
        B (float): Battery action (scalar).
        Price (float): Real-time price (scalar).
        SoC (float): State of charge (scalar).

        Returns:
        float: Real-time profit (scalar).
        """
        return B * Price +  (1/2) * self.penalty * (B)**2 

    def derivative(self, B, Price, B_DA):

        """
        Calculate the derivative of the real-time cost with respect to B.

        Parameters:
        B (float): Battery action (scalar).
        Price (float): Real-time price (scalar).
        SoC (float): State of charge (scalar).

        Returns:
        float: Derivative (scalar).
        """

        return Price + self.penalty * (B)


class softSoC_Constraint():
    def __init__(self, charging_efficiency, SoC_max, SoC_target, penalty_coeff=100.0):
        """
        Initialize the soft state of charge constraint.

        Parameters:
        SoC_min (float): Minimum state of charge (scalar).
        SoC_max (float): Maximum state of charge (scalar).
        penalty_coeff (float): Penalty coefficient for constraint violation (scalar).
        """
        self.SoC_max = SoC_max
        self.penalty_coeff = penalty_coeff
        self.SoC_target = SoC_target
        self.charging_eff = charging_efficiency
        assert 0<= SoC_target <= SoC_max, "SoC_target must be less than or equal to SoC_max"

    def cost(self,SoC):
        """
        Calculates the cost of the penalty term with respect to battery power (B) and state of charge (SoC).

        Parameters:
            B (float): Battery power. Positive for charging, negative for discharging.
            SoC (float): Current state of charge of the battery.

        Returns:
            float: The computed quadratic cost.
        """
        return (self.penalty_coeff/2) * (SoC - self.SoC_target)**2
    
    def derivative(self,SoC):
        """
        Calculates the derivative of the penalty term with respect to battery power (B) and state of charge (SoC).

        Parameters:
            B (float): Battery power. Positive for charging, negative for discharging.
            SoC (float): Current state of charge of the battery.

        Returns:
            float: The computed derivative value of quadratic cost.
        """
        return self.penalty_coeff * (SoC- self.SoC_target) 



class LHS_2D_design():
    """
    Generates a 2D Latin Hypercube Sampling (LHS) design for two variables: price and state of charge (SoC).

    Parameters
    ----------
    N_samples : int
        Number of LHS samples to generate (must be greater than 0).
    Price_lb : float
        Lower bound for the price variable.
    Price_ub : float
        Upper bound for the price variable.
    SoC_lb : float
        Lower bound for the state of charge (SoC) variable.
    SoC_ub : float
        Upper bound for the state of charge (SoC) variable.
    seed_num : int, optional
        Random seed for reproducibility. Default is None.
    fence : int, optional
        Number of points to use for boundary sampling along each edge. Default is 10.

    Returns:
        Generates and returns the sampled points as two arrays: X (price) and I (SoC).
        The samples include both LHS points and boundary points along the edges of the domain.
    """
    def __init__(self, N_samples, Price_lb, Price_ub, SoC_lb, SoC_ub, seed_num=None, fence=10):
        assert N_samples > 0, "Number of samples cannot be 0 or less"
        self.N_samples = N_samples
        self.Price_lb = Price_lb
        self.Price_ub = Price_ub
        self.SoC_lb = SoC_lb
        self.SoC_ub = SoC_ub
        self.seed = seed_num
        self.fence = fence
        self._cached_design = None

    @property
    def create_samples(self):
        if self._cached_design is not None:
            return self._cached_design

        eps = 1e-10
        assert np.abs(self.Price_lb - self.Price_ub) > eps, "Price_lb and Price_ub cannot be the same"

        sampler = qmc.LatinHypercube(d=2, scramble=True, seed=self.seed)
        lhs_samples = sampler.random(n=self.N_samples)
        lower_bounds = [self.Price_lb, self.SoC_lb]
        upper_bounds = [self.Price_ub, self.SoC_ub]
        lhs_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)

        price_grid = np.linspace(self.Price_lb, self.Price_ub, self.fence)
        soc_lower_boundary = np.column_stack((price_grid, np.full(len(price_grid), self.SoC_lb)))
        soc_upper_boundary = np.column_stack((price_grid, np.full(len(price_grid), self.SoC_ub)))

        boundary_points = np.row_stack((soc_lower_boundary, soc_upper_boundary))
        combined_design = np.row_stack((lhs_samples, boundary_points))
        Price_samples = combined_design[:, 0]
        SoC_samples = combined_design[:, 1]

        self._cached_design = (Price_samples, SoC_samples)
        return Price_samples, SoC_samples

class costToGo_opt():
    """
    Class to compute the cost-to-go value and its derivative for battery operation optimization.

    Parameters:
    charing_efficiency (float): Battery charging efficiency.
    running_cost (object): Instance of RTRunningCost or similar cost function.
    B_DA (float): Day-ahead battery action.
    Delta_t (float): Time step size.
    continuation_map (object): Either a GPy GP model or a cost/derivative function for continuation value.
    """
    def __init__(self, charing_efficiency, running_cost, B_DA, Delta_t, continuation_map):
        self.charging_eff = charing_efficiency
        self.running_cost = running_cost
        self.B_DA = B_DA
        self.Delta_t = Delta_t
        self.continuation_map = continuation_map

    def compute_costToGo_value(self, B, cur_Price, cur_SoC):
        """
        Compute the cost-to-go value for a given battery action, price, and state of charge.

        Parameters:
        B (array-like): Battery action (array of length 1).
        cur_Price (float): Current price.
        cur_SoC (float): Current state of charge.

        Returns:
        float: Cost-to-go value.
        """
        assert isinstance(B, (list, np.ndarray)), "B must be array of length 1."
        B_total = B[0] + self.B_DA
        # Compute next SoC based on battery action and efficiency
        curPrice_nextSoC = np.array([
            cur_Price,
            cur_SoC + self.Delta_t * (
                self.charging_eff * (B_total >= 0) +
                1 / self.charging_eff * (B_total < 0)
            ) * B_total
        ])
        # Get continuation value from GP or cost function
        if isinstance(self.continuation_map, GPy.core.GP):
            continuation_value = self.continuation_map.predict(curPrice_nextSoC.reshape(1, -1))[0].flatten()[0]
        else:
            continuation_value = self.continuation_map.cost(curPrice_nextSoC[1])

        # Total cost-to-go value
        costToGo_value = self.running_cost.cost(B[0], cur_Price, self.B_DA) * self.Delta_t + continuation_value
        return costToGo_value

    def compute_costToGo_derivative(self, B, cur_Price, cur_SoC):
        """
        Compute the derivative of the cost-to-go value with respect to battery action.

        Parameters:
        B (array-like): Battery action (array of length 1).
        cur_Price (float): Current price.
        cur_SoC (float): Current state of charge.

        Returns:
        float: Derivative of cost-to-go value.
        """
        B_total = B[0] + self.B_DA
        # Compute next SoC based on battery action and efficiency
        curPrice_nextSoC = np.array([
            cur_Price,
            cur_SoC + self.Delta_t * (
                self.charging_eff * (B_total >= 0) +
                1 / self.charging_eff * (B_total < 0)
            ) * B_total
        ])
        # Get derivative of continuation value from GP or derivative function
        if isinstance(self.continuation_map, GPy.core.GP):
            continuation_derivative = self.continuation_map.predictive_gradients(curPrice_nextSoC.reshape(1, -1))[0][:, 1].flatten()[0]
            continuation_derivative = continuation_derivative * \
                self.Delta_t * (self.charging_eff * (B_total >= 0) + 1 / self.charging_eff * (B_total < 0))
        else:
            continuation_derivative = self.continuation_map.derivative(curPrice_nextSoC[1]) * \
                self.Delta_t * (self.charging_eff * (B_total >= 0) + 1 / self.charging_eff * (B_total < 0))

        # Total derivative of cost-to-go value
        costToGo_derivative = self.running_cost.derivative(B[0], cur_Price, self.B_DA) * self.Delta_t + continuation_derivative
        return costToGo_derivative


class OU_w_derivative:
    """
    Ornstein–Uhlenbeck process with time-varying mean reversion M(t) and derivative dM/dt.
    SDE:
        dX_t = [ α(M(t) - X_t) + dM/dt(t) ] dt + σ dW_t
    """

    def __init__(self, X0, nstep, nsim, maturity,
                 alpha, meanRevRate_func, dmeanRevRate_func, sigma,
                 noises, seed=None):
        """
        Parameters
        ----------
        X0 : array-like (nsim,)  – initial values
        nstep : int              – number of time steps
        nsim : int               – number of simulations
        maturity : float         – total time horizon
        alpha : float            – mean reversion rate
        meanRevRate_func : func  – function M(t)
        dmeanRevRate_func : func – function dM/dt(t)
        sigma : float            – volatility
        UB, LB : unused (kept for compatibility)
        noises : array or None   – optional pre-generated noises
        seed : int or None       – random seed for reproducibility
        """
        self.X0 = X0
        self.nstep = nstep
        self.nsim = nsim
        self.maturity = maturity
        self.dt = maturity / nstep
        self.t = np.linspace(0, maturity, nstep + 1)

        self.alpha = alpha
        self.sigma = sigma
        self.meanRevRate_func = meanRevRate_func
        self.dmeanRevRate_func = dmeanRevRate_func

        # Fix random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate noise (fixed by seed if given)
        if noises is None:
            self.noises = np.random.normal(0, 1, size=(nsim, nstep))
        else:
            self.noises = noises

        # Simulate all trajectories
        self.sim_trajectories = self.simulate()

    def drift(self, t, X):
        """Drift = α(M(t) − X) + dM/dt(t)"""
        return self.alpha * (self.meanRevRate_func(t) - X) + self.dmeanRevRate_func(t)

    def simulate(self, new_sim=False, seed=None):
        """
        Simulate full trajectories.
        If seed is provided, regenerate deterministic noise and simulation.
        """
        if seed is not None:
            np.random.seed(seed)
            self.noises = np.random.normal(0, 1, size=(self.nsim, self.nstep))
        elif new_sim:
            self.noises = np.random.normal(0, 1, size=(self.nsim, self.nstep))

        X = np.zeros((self.nsim, self.nstep + 1))
        X[:, 0] = self.X0

        for k in range(self.nstep):
            t_k = self.t[k]
            drift = self.drift(t_k, X[:, k])
            X[:, k + 1] = X[:, k] + drift * self.dt + self.sigma * X[:, k] * np.sqrt(self.dt) * self.noises[:, k]

        return X

    def one_step_simulate(self, step_idx, X_start, nsim, seed=None):
        
        """
        One-step forward simulation from X_start at time step `step_idx`.
        Fixable with a seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        dW = np.random.normal(0, 1, size=nsim)
        t_k = self.t[step_idx]
        drift = self.drift(t_k, X_start)
        X_next = X_start + drift * self.dt + self.sigma *X_start* np.sqrt(self.dt) * dW
        return X_next
    
    def multi_step_simulate(self, start_step_idx, X_start, n_substeps, nsim=None, seed=None):
        """
        Evolve X forward by n_substeps of length self.dt, starting from time index start_step_idx.

        Parameters
        ----------
        start_step_idx : int
            Index k such that we start at time t[k].
        X_start : array-like, shape (nsim,)
            Starting values at t[start_step_idx].
        n_substeps : int
            How many fine-grid steps to move forward.
        nsim : int or None
            Number of simulations (defaults to len(X_start)).
        seed : int or None
            Random seed for this evolution.

        Returns
        -------
        X : np.ndarray, shape (nsim,)
            Values at t[start_step_idx + n_substeps].
        """
        X = np.array(X_start, copy=True)
        if nsim is None:
            nsim = len(X)

        if seed is not None:
            np.random.seed(seed)

        noises = np.random.normal(0, 1, size=(nsim, n_substeps))

        for j in range(n_substeps):
            t_k = self.t[start_step_idx + j]
            drift = self.drift(t_k, X)
            dW = noises[:, j]
            X = X + drift * self.dt + self.sigma * X * np.sqrt(self.dt) * dW

        return X
class valueGP(GPy.core.GP):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = GPy.kern.Matern52(input_dim=X.shape[1],ARD= True)
        assert isinstance(kernel, GPy.kern.Kern)

        likelihood = GPy.likelihoods.Gaussian(variance=noise_var)

        super(valueGP, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

class policyGP(GPy.core.GP):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        if kernel is None:
            kernel = GPy.kern.Matern32(input_dim=X.shape[1],ARD= True)

        assert isinstance(kernel, GPy.kern.Kern), "Kernel must be GPy kernel."

        likelihood = GPy.likelihoods.Gaussian(variance=noise_var)

        super(policyGP, self).__init__(X, Y, kernel, likelihood, name='GP regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)

            
class ShadowGPTrainer:
    """
    SHADOw-GP style backward trainer with decoupled simulation and decision grids.

    - OU process is simulated on a fine grid: process.nstep (e.g., 96 15-min steps for 24h).
    - Decisions are made on a coarser (or equal) grid: ndecisions (e.g., 24 hourly decisions).
    - We build one policy and one continuation map per *decision* time.
    - Terminal time T is handled via final_cost (e.g., softSoC_Constraint).

    Parameters
    ----------
    process : OU_w_derivative (or compatible)
        Must expose .nstep, .dt, .maturity, .t, and .multi_step_simulate().
    BESSparameters : tuple
        (Bmax_scalar, Bmin_scalar, B_DA_vec, Imax, charging_eff)
        B_DA_vec must have length = ndecisions.
    running_cost : object
        Must have .cost(B, Price, B_DA).
    final_cost : object or GPy.core.GP
        Terminal cost; if not GP, must have .cost(SoC).
    nsim_design : int
        Number of design points per decision step for GP training.
    continuation_kernel, policy_kernel : str or None
        Names in {"RBF", "Matern52", "Matern32", None}.
    batch_size : int
        If >0, use batched MC within each design point in _pointwise_values_decision.
    ndecisions : int or None
        Number of decision times over [0,T). If None, defaults to process.nstep.
    """

    def __init__(self,
                 process,
                 BESSparameters,
                 running_cost,
                 final_cost,
                 nsim_design,
                 continuation_kernel="Matern52",
                 policy_kernel="Matern32",
                 batch_size=0,
                 ndecisions=None):

        self.process = process
        (self.Bmax_scalar,
         self.Bmin_scalar,
         self.B_DA,
         self.Imax,
         self.charging_eff) = BESSparameters

        self.running_cost  = running_cost
        self.final_cost    = final_cost
        self.nsim_design   = nsim_design
        self.batch_size    = batch_size

        # underlying simulation grid (fine)
        self.nstep    = process.nstep          # e.g. 96
        self.dt_sim   = process.dt             # e.g. 0.25 if 24h / 96
        self.T        = process.maturity       # total horizon
        self.t_sim    = process.t              # fine time grid (length nstep+1)

        # decision grid
        if ndecisions is None:
            # old behavior: one decision per OU step
            self.ndecisions = self.nstep
        else:
            self.ndecisions = ndecisions

        assert self.nstep % self.ndecisions == 0, "nstep must be a multiple of ndecisions"
        self.step_factor = self.nstep // self.ndecisions  # #fine steps between decisions

        # decision indices in the simulation grid:
        # decisions at t=0,1,...,T-dt_opt → indices 0,step_factor,...,nstep-step_factor
        self.decision_idx = np.arange(0, self.nstep, self.step_factor)
        assert len(self.decision_idx) == self.ndecisions, "one sim index per decision time"

        # terminal index in the simulation grid (not a decision time)
        self.terminal_idx = self.nstep     # e.g. 96, corresponding to t = T

        # optimization dt at decision level (e.g. 1h if T=24, ndecisions=24)
        self.dt_opt = self.T / self.ndecisions
        # decision times: 0, 1, ..., 23  (does NOT include T)
        self.t_dec  = np.arange(self.ndecisions) * self.dt_opt
        self.t_terminal = self.T

        # kernels
        self.kernel_dic = {
            "RBF":      GPy.kern.RBF,
            "Matern52": GPy.kern.Matern52,
            "Matern32": GPy.kern.Matern32,
            None:       None
        }
        self.q_kern_cls = self.kernel_dic[continuation_kernel]
        self.p_kern_cls = self.kernel_dic[policy_kernel]

        # state envelopes only at decision times
        # sim_trajectories: (nsim, nstep+1); pick decision indices
        sim_dec = self.process.sim_trajectories[:, self.decision_idx]  # (nsim, ndecisions)
        self.X_lowers, self.X_uppers = np.quantile(sim_dec, q=[0.005, 0.995], axis=0)

        # outputs defined on decision grid
        # continuation_maps[k] corresponds to cost-to-go at decision time k
        # with continuation_maps[-1] = final_cost (terminal)
        self.continuation_maps = [None] * self.ndecisions
        self.policy_maps       = [None] * self.ndecisions
        self.continuation_mse  = [None] * self.ndecisions
        self.policy_mse        = [None] * self.ndecisions

    # -------- public API -----------------------------------------------------
    def fit(self, design_seed=None, one_step_seed=None, fence=10):
        """
        Train backward on the decision grid and populate maps + MSEs.

        design_seed : int or None
            Base seed for LHS designs at each decision.
        one_step_seed : int or None
            Base seed for OU evolution between decisions.
        fence : int
            Boundary points per edge for LHS_2D_design.
        """

        # terminal continuation value at k = ndec-1 (interpreted as t = T)
        self.continuation_maps[-1] = self.final_cost

        # last decision index: k = ndec-1 (time t = T - dt_opt)
        last_k = self.ndecisions - 1
        print(f"training index[{last_k}]")

        X_B, I_B = self._make_design(
            self.nsim_design,
            self.X_lowers[last_k], self.X_uppers[last_k],
            0.0, self.Imax,
            seed=self._derive_seed(design_seed, last_k),
            fence=fence
        )

        pmap_T, qmse_T = self._train_policy_simple(
            X_B, I_B,
            self.B_DA[last_k],
            self.continuation_maps[last_k]    # final_cost at terminal
        )
        self.policy_maps[last_k] = pmap_T
        self.policy_mse[last_k]  = qmse_T
        if qmse_T >= 0.1:
            print(f"ERROR MSE greater for policy_map[{last_k}]: {qmse_T}")

        # backward over decision indices k = ndec-1, ..., 1
        for k in range(self.ndecisions - 1, 0, -1):

            print(f"training index[{k-1}]")

            # design at decision time k-1
            X_prev, I_next = self._make_design(
                self.nsim_design,
                self.X_lowers[k-1], self.X_uppers[k-1],
                0.0, self.Imax,
                seed=self._derive_seed(design_seed, k-1),
                fence=fence
            )

            # pointwise cost-to-go using:
            # - policy at time k
            # - continuation at time k
            pv = self._pointwise_values_decision(
                k_dec=k-1,
                X_prev=X_prev,
                I_next=I_next,
                policy_map=self.policy_maps[k],
                continuation_map=self.continuation_maps[k],
                B_DA_t=self.B_DA[k],
                one_step_seed=self._derive_seed(one_step_seed, k-1)
            )

            # fit continuation value map at decision time k-1
            qmap, qmse = self._train_continuation_gp(X_prev, I_next, pv)
            self.continuation_maps[k-1] = qmap
            self.continuation_mse[k-1]  = qmse
            if qmse >= 0.1:
                print(f"ERROR MSE greater for continuation_map[{k-1}]: {qmse}")

            # now fit policy at decision time k-1
            X_B, I_B = self._make_design(
                self.nsim_design,
                self.X_lowers[k-1], self.X_uppers[k-1],
                0.0, self.Imax,
                seed=self._derive_seed(design_seed, k-1),
                fence=fence
            )

            pmap, pmse = self._train_policy_simple(
                X_B, I_B,
                self.B_DA[k-1],
                self.continuation_maps[k-1]
            )
            self.policy_maps[k-1] = pmap
            self.policy_mse[k-1]  = pmse
            if pmse >= 0.1:
                print(f"ERROR MSE greater for policy_map[{k-1}]: {pmse}")

        return {
            "t_dec": self.t_dec.tolist(),          # decision times 0,...,T-dt_opt
            "t_terminal": self.t_terminal,         # T
            "continuation_mse_by_step": self.continuation_mse,
            "policy_mse_by_step": self.policy_mse,
            "continuation_maps": self.continuation_maps,
            "policy_maps": self.policy_maps
        }

    # -------- seeding helpers ------------------------------------------------
    def _derive_seed(self, seed, step):
        assert seed is not None, "At least one seed must be provided."
        return seed + int(step)

    # -------- design + training internals -----------------------------------
    def _make_design(self, n, x_lb, x_ub, i_lb, i_ub, seed=None, fence=10):
        return LHS_2D_design(
            n, x_lb, x_ub, i_lb, i_ub,
            seed_num=seed, fence=fence
        ).create_samples

    def _train_continuation_gp(self, X_train, I_train, y):
        cin  = np.column_stack([X_train, I_train])
        cout = y.reshape(-1, 1)
        kern = None if self.q_kern_cls is None else self.q_kern_cls(cin.shape[1], ARD=True)
        cont_map = valueGP(cin, cout, kernel=kern, normalizer=True)
        cont_map.optimize()
        pred = cont_map.predict(cin)[0].flatten()
        mse  = np.mean((y - pred)**2)
        return cont_map, mse

    def _train_policy_simple(self, X_train, I_train, B_DA_t, continuation_map):
        """Unconstrained L-BFGS policy training using costToGo_opt with dt_opt."""
        assert len(X_train) == len(I_train)
        B_hat = np.zeros(len(X_train))

        def mk_objs(x, i):
            ctg = costToGo_opt(
                self.charging_eff,
                self.running_cost,
                B_DA_t,
                self.dt_opt,          # <<< use optimization dt, NOT simulation dt
                continuation_map
            )

            def f(b_arr):
                return ctg.compute_costToGo_value(b_arr, x, i)

            def g(b_arr):
                return ctg.compute_costToGo_derivative(b_arr, x, i)

            return f, g

        for k, (x, i) in enumerate(zip(X_train, I_train)):
            f, g = mk_objs(x, i)
            res = scipy.optimize.minimize(
                f, jac=g, x0=0.0,
                method="L-BFGS-B",
                bounds=None
            )
            B_hat[k] = res.x[0]

        control_in  = np.column_stack([X_train, I_train])
        control_out = B_hat.reshape(-1, 1)
        kern = None if self.p_kern_cls is None else self.p_kern_cls(control_in.shape[1], ARD=True)
        pol_map = policyGP(control_in, control_out, kernel=kern, normalizer=True)
        pol_map.optimize()
        pred = pol_map.predict(control_in)[0].flatten()
        mse  = np.mean((B_hat - pred)**2)
        return pol_map, mse

    # -------- continuation evaluation on decision grid ----------------------
    def _pointwise_values_decision(self, k_dec, X_prev, I_next,
                                   policy_map, continuation_map,
                                   B_DA_t, one_step_seed=None):
        """
        Compute pointwise cost-to-go at decision index k_dec using:

        - underlying OU evolution from t_dec[k_dec] to t_dec[k_dec+1] via fine grid,
        - policy at time k_dec+1,
        - continuation map at time k_dec+1 (or final_cost at terminal),
        - SoC dynamics and running cost over dt_opt.
        """

        # map decision index to fine simulation index
        start_idx  = self.decision_idx[k_dec]
        n_substeps = self.step_factor

        if self.batch_size != 0:
            X_prev = np.repeat(X_prev, self.batch_size)
            I_next = np.repeat(I_next, self.batch_size)

        # evolve OU from t_dec[k_dec] to t_dec[k_dec+1] using fine steps
        X_next = self.process.multi_step_simulate(
            start_step_idx=start_idx,
            X_start=X_prev,
            n_substeps=n_substeps,
            nsim=len(X_prev),
            seed=one_step_seed
        )

        # Policy at (X_next, I_next)
        RT_adj = policy_map.predict(
            np.column_stack([X_next, I_next])
        )[0].flatten()

        # Physical SoC window [0, Imax]
        Ilb, Iub = 0.0, self.Imax

        # use dt_opt for SoC-step constraints (decision dt, not sim dt)
        LB = np.maximum(
            self.Bmin_scalar - B_DA_t,
            (self.charging_eff*(Ilb - I_next) / self.dt_opt) - B_DA_t
        )
        UB = np.minimum(
            self.Bmax_scalar - B_DA_t,
            ((Iub - I_next) / (self.charging_eff * self.dt_opt)) - B_DA_t
        )
        RT_adj = np.maximum(LB, np.minimum(RT_adj, UB))

        P_rt = RT_adj + B_DA_t
        eff  = self.charging_eff*(P_rt >= 0) + (1.0/self.charging_eff)*(P_rt < 0)
        I_nextnext = I_next + P_rt * eff * self.dt_opt
        I_nextnext = np.clip(I_nextnext, Ilb, Iub)

        # running cost over dt_opt
        run_cost = self.running_cost.cost(RT_adj, X_next, B_DA_t) * self.dt_opt

        if isinstance(continuation_map, GPy.core.GP):
            cont = continuation_map.predict(
                np.column_stack([X_next, I_nextnext])
            )[0].flatten()
        else:
            cont = continuation_map.cost(I_nextnext)

        costToGo_values = run_cost + cont

        if self.batch_size != 0:
            return np.mean(costToGo_values.reshape(-1, self.batch_size), axis=1)
        else:
            return costToGo_values
