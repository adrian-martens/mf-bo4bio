import numpy as np
from scipy.integrate import solve_ivp

import mfbo4bio.conditions_data as data


class EXPERIMENT:
    def __init__(
        self,
        T=32,
        pH=7.2,
        cell_type="celltype_1",
        reactor="MTP",
        feeding=[(10, 0), (20, 0), (30, 0)],
        time=150,
        clone_distribution="alpha",
    ):
        """
        EXPERIMENT Class simulates CHO kinetics depending on the initialization:

        T: Temperature of the reactor in type float

        pH: pH of the reactor in type float

        cell_type: Used cell culture in the experiment (e.g. 'celltype_1')

        reactor: Used reactor scale and type (e.g. 'MTP')

        feeding: Feeding regime as a list of time and concentration tuples
        (e.g. [(50, 0), (80, 0), (100, 0), (120, 10)])
        """

        df = data.get_cell_params(clone_distribution=clone_distribution)
        params = df[df["cell_type"] == cell_type]

        self.reactor = reactor
        self.volume = 3  # L
        self.cell_type = cell_type
        self.time = time  # experiment time
        self.mu_max = params["mu_max"].iloc[0]

        self.K_lysis = params["K_lysis"].iloc[0]

        self.K_L, self.K_A, self.K_G, self.K_Q = params["K"].iloc[0]
        self.Y = params["Y"].iloc[0]
        self.m = params["m"].iloc[0]
        self.k_d_Q, self.k_d_max, self.k_mu = params["k"].iloc[0]

        self.A = params["A"].iloc[0]
        self.E_a = params["E_a"].iloc[0]
        self.pH_opt = params["pH_opt"].iloc[0]

        # P, X_T, X_V, X_D, G, Q, L, A = x
        self.initial_conditions = [0, 1e6, 0.8 * 1e6, 0, 210, 1, 9, 0]
        self.solution = None
        self.t = None

        self.T = T
        self.pH = pH

        self.feeding = feeding

        self.R = 8.314

    def temperature_effect(self):
        """
        Calculates a temperature-dependent factor for the growth rate.

        This function models the effect of temperature on the cell growth rate using a
        Gaussian-like function centered around an optimal temperature.

        Returns
        -------
        numpy.ndarray or float
            The temperature factor, clipped to a minimum value of 0.2.
        """
        x = self.T
        mu = self.E_a
        A = 1.2

        left_part = np.exp(-1 * ((x - mu) / 5) ** 2)
        right_part = np.exp(-1 * ((x - mu) / 5) ** 2)

        factor = A * np.where(x < mu, left_part, right_part)
        factor = np.clip(factor, 0.2, None)
        return factor

    # Function to calculate pH effect
    def pH_effect(self):
        """
        Calculates a pH-dependent factor for the growth rate.

        This function models the effect of pH on the cell growth rate
        using an asymmetric Gaussian-like function centered around an optimal pH.

        Returns
        -------
        float
            The pH factor, clipped to a minimum value of 0.2.
        """
        x = self.pH
        mu = self.pH_opt
        A = 1.2

        left_part = np.exp(-1 * ((x - mu) / 1.5) ** 2)
        right_part = np.exp(-1 * ((x - mu) / 0.8) ** 2)

        factor = A * np.where(x < mu, left_part, right_part)
        factor = np.clip(factor, 0.2, None)
        return factor

    def mu(self, G, Q, L, A, growth_inhibition=1):
        """
        Calculates the Growth Rate depending on:

            G - glucose concentration
            Q - glutamine concentration
            L - lactate concentration
            A - ammonia concentration

        Returns a growth rate value in h^(-1)
        """
        temperature_factor = self.temperature_effect()
        pH_factor = self.pH_effect()

        mu_max = self.mu_max
        K_G = self.K_G
        K_Q = self.K_Q
        K_L = self.K_L
        K_A = self.K_A

        G = max(G, 1e-6)
        Q = max(Q, 1e-6)
        L = max(L, 1e-6)
        A = max(A, 1e-6)

        mu = (
            mu_max
            * G
            / (K_G + G)
            * Q
            / (K_Q + Q)
            * K_L
            / (K_L + L)
            * K_A
            / (K_A + A)
            * temperature_factor
            * pH_factor
            * growth_inhibition
        )
        return mu

    def ODE(self, t, x, scale=None):
        """
        Calculates the rate of change for the system's components.

        This function defines the system of ordinary differential equations (ODEs) that
        describe the kinetics of the cell culture, including cell growth, death, and
        substrate/product metabolism.

        Parameters
        ----------
        t : float
            Current time.
        x : list
            A list of current concentrations/amounts: [P, X_T, X_V, X_D, G, Q, L, A],
            where:
            P: Product
            X_T: Total cell concentration
            X_V: Viable cell concentration
            X_D: Dead cell concentration
            G: Glucose concentration
            Q: Glutamine concentration
            L: Lactate concentration
            A: Ammonia concentration
        scale : str, optional
            The reactor scale ("MTP", "MBR", or "PILOT").
            If None, uses the instance's reactor attribute.

        Returns
        -------
        list
            A list of the gradients (rates of change) for each component.
        """

        if scale is None:
            scale = self.reactor

        P, X_T, X_V, X_D, G, Q, L, A = x

        if scale == "MTP":
            scale_factor = 5
            growth_inhibition = 0.95

        elif scale == "MBR":
            scale_factor = 4
            growth_inhibition = 0.99

        elif scale == "PILOT":
            scale_factor = 8
            growth_inhibition = 1.1

        mu = self.mu(G, Q, L, A, growth_inhibition)

        k_d = self.k_d_max * (self.k_mu / (mu + self.k_mu)) * scale_factor

        Y_X_G, Y_X_Q, Y_L_G, Y_A_Q, Y_P_X, Y_dot_P_X = self.Y
        m_G, m_Q = self.m

        dX_T_dt = mu * X_V - self.K_lysis * X_D
        dX_V_dt = (mu - k_d) * max(X_V, 1e-6) ** growth_inhibition
        dX_D_dt = k_d * X_V - self.K_lysis * X_D

        dP_dt = Y_P_X * X_T + Y_dot_P_X * (mu * G / (self.K_G + G)) * X_V

        dG_dt = X_V * (-mu / Y_X_G - m_G)
        dQ_dt = X_V * (-mu / Y_X_Q - m_Q) - self.k_d_Q * Q
        dL_dt = -X_V * Y_L_G * (-mu / Y_X_G - m_G)
        dA_dt = -X_V * Y_A_Q * (-mu / Y_X_Q - m_Q) + self.k_d_Q * Q

        gradients = [dP_dt, dX_T_dt, dX_V_dt, dX_D_dt, dG_dt, dQ_dt, dL_dt, dA_dt]

        return gradients

    # Integration
    def ODE_solver(self, model=None):
        """
        Solves the ODE system over time, handling feeding events dynamically.

        The method integrates the system of ODEs in segments, with each segment ending
        at a scheduled feeding time. At each feeding time, the glucose and glutamine
        concentrations are updated before proceeding to the next segment.

        Parameters
        ----------
        model : str, optional
            The reactor model to use for the ODEs.
            If None, the instance's `reactor` attribute is used.

        Returns
        -------
        numpy.ndarray
            A NumPy array of the solutions for all state variables
            over the entire time course.
        """

        if model is None:
            model = self.reactor

        t_eval_total = []
        y_total = []
        current_t = 0
        current_y = self.initial_conditions.copy()

        for event_time, new_G_value in self.feeding:
            t_span_segment = (current_t, event_time)
            t_eval_segment = np.linspace(current_t, event_time, 1000)

            solution = solve_ivp(
                fun=self.ODE,
                t_span=t_span_segment,
                y0=current_y,
                t_eval=t_eval_segment,
                method="RK45",
                args=(model,),
            )

            t_eval_total.extend(solution.t)
            y_total.append(solution.y)

            current_t = event_time
            current_y = solution.y[:, -1]
            new_Q_value = new_G_value * 0.1

            current_y[4] += new_G_value
            current_y[5] += new_Q_value

        t_span_segment = (current_t, self.time)
        t_eval_segment = np.linspace(current_t, self.time, 1000)

        solution = solve_ivp(
            fun=self.ODE,
            t_span=t_span_segment,
            y0=current_y,
            t_eval=t_eval_segment,
            method="RK45",
            args=(model,),
        )

        t_eval_total.extend(solution.t)
        y_total.append(solution.y)

        t_eval_total = np.array(t_eval_total)
        y_total = np.hstack(y_total)

        self.solution = y_total
        self.solution[0] = self.solution[0] / (self.volume * 1e3)
        self.t = t_eval_total
        return y_total

    def measurement(self, noise_level=None, quantity="P", rng=None):
        """
        Simulates a measurement with added noise.

        This method first solves the ODE system to get the true final value
        of a specified quantity and then adds a normally distributed noise to
        this value to simulate a real-world measurement.

        Parameters
        ----------
        noise_level : float, optional
            The standard deviation of the noise as a fraction of the true value.
            If None, no noise is added. Defaults to None.
        quantity : str, optional
            The name of the quantity to measure:
            ("P", "X_T", "X_V", "X_D", "G", "Q", "L", or "A").
            Defaults to "P".
        rng : numpy.random.Generator, optional
            A NumPy random number generator for reproducibility.
            If None, a default generator is used.

        Returns
        -------
        float
            The simulated noisy measurement value, clipped to a minimum of 0.0.
        """

        if noise_level is None:
            noise_level = 0

        self.ODE_solver()

        index = {"P": 0, "X_T": 1, "X_V": 2, "X_D": 3, "G": 4, "Q": 5, "L": 6, "A": 7}

        true_value = self.solution[index[quantity]][-1]

        if rng is None:
            rng = np.random.default_rng()

        noise_magnitude = max(noise_level * true_value, 1e-8)
        noise = rng.normal(0, noise_magnitude)
        noisy_value = true_value + noise

        return max(noisy_value, 0.0)


def conduct_experiment(
    X,
    initial_conditions=[0, 0.4 * 1e9, 0.4 * 1e6, 0, 10, 20, 0, 1.8],
    mbr_level=7,
    clone_distribution="alpha",
    rng=None,
):
    """
    Simulates a series of cell culture experiments and returns the final product values.

    This function processes a multidimensional input `X`, where each row represents
    a set of experimental conditions (e.g., temperature, pH, feeding regime).
    It initializes an `EXPERIMENT` object for each set of conditions and simulates
    the outcome, returning a list of final product measurements.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array where each row contains the process parameters
        for a single experiment. The number of columns determines the
        parameters to use (2, 5, 6, or 7).
    initial_conditions : list, optional
        Initial conditions for the cell culture variables.
        Defaults to [0, 0.4e9, 0.4e6, 0, 10, 20, 0, 1.8].
    mbr_level : int, optional
        A user-defined fidelity level for the MBR reactor scale. Defaults to 7.
    clone_distribution : str, optional
        The type of clone distribution to use ("alpha", "beta", or "test").
        Defaults to "alpha".
    rng : numpy.random.Generator, optional
        A NumPy random number generator for reproducibility.
        If None, a default generator is used.

    Returns
    -------
    list
        A list of the simulated final product values for each experiment,
        with added noise.

    Raises
    ------
    ValueError
        If the dimensionality of X is not 2, 5, 6, or 7.
    NameError
        If an invalid `clone_distribution` is specified.
    """

    reactor_list = {0: "MTP", mbr_level: "MBR", 10: "PILOT"}

    if clone_distribution == "alpha":
        process_parameters = data.process_parameters_alpha
    elif clone_distribution == "beta":
        process_parameters = data.process_parameters_beta
    elif clone_distribution == "test":
        process_parameters = data.process_parameters_test
    else:
        raise NameError(
            f"No clone distribution named {clone_distribution}. \
                Please specify 'alpha' or 'beta' as input"
        )

    result = []
    feeding = [(10, 0), (20, 0), (30, 0)]
    reactor = "PILOT"
    clone = "celltype_1"

    for row in X:
        if len(row) == 2:
            T, pH = row

        elif len(row) == 5:
            T, pH, F1, F2, F3 = row
            feeding = [(70, float(F1)), (100, float(F2)), (130, float(F3))]

        elif len(row) == 6:
            T, pH, F1, F2, F3, fidelity = row
            if int(np.round(fidelity)) == 0:
                feeding = [(70, 0), (100, float(F2)), (130, 0)]
            else:
                feeding = [(40, float(F1)), (100, float(F2)), (160, float(F3))]
            reactor = reactor_list[int(np.round(fidelity))]

        elif len(row) == 7:
            T, pH, F1, F2, F3, fidelity, clone = row
            clone = list(process_parameters.keys())[int(np.round(clone))]

            if int(np.round(fidelity)) == 0:
                feeding = [(70, 0), (140, float(F2)), (210, 0)]
            else:
                feeding = [(70, float(F1)), (140, float(F2)), (210, float(F3))]
            reactor = reactor_list[int(np.round(fidelity))]

        else:
            raise ValueError(
                f"Cannot handle the dimensionality of X. \
                n must be 2, 5, 6 or 7 but is {len(row)}"
            )

        cell = EXPERIMENT(
            T=T,
            pH=pH,
            time=320,
            feeding=feeding,
            reactor=reactor,
            cell_type=clone,
            clone_distribution=clone_distribution,
        )
        cell.initial_conditions = initial_conditions
        value = float(
            cell.measurement(
                quantity="P", noise_level=data.noise_level[reactor], rng=rng
            )
        )
        result.append(value)
    return result
