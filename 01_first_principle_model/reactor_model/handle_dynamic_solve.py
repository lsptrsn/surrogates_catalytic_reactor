"""
Solving Partial-Differential Equation 1D-PFTR for Methanantion
Author: Luisa Peterson
"""

###########################################################################
# Import packages
###########################################################################
import datetime
import diffrax as dx
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time

from balances import PDE_sys
from utils import format_time, get_center_temperature, replace_out_of_bounds_values
from parameter import data

# from functools import partial

jax.config.update("jax_enable_x64", True)
eps = np.finfo(float).eps

def run_reactor_model_dynamic(initial_values, simulation_name='', dynamic_plot=True):
    """
    Solve the reactor model using Diffrax.

    Args:
        initial_values (list): Initial state of the reactor.
        PDE_sys (function): The function defining the ODE system.
        data (dict): Dictionary containing model parameters and time points.
        dynamic_plot (bool): Whether to dynamically plot results (optional).

    Returns:
        dx.Solution: Solution object containing the ODE solution.
    """
    ###########################################################################
    # Solve localation-discretized PDE-system in each control volume
    ###########################################################################
    # intial composition values for t=0 for each component in control volume
    X_CO2_0 = initial_values[0]
    tau_0 = (initial_values[1]-data["T_scale_add"])/data["T_scale"]
    x_ini = np.array([X_CO2_0, tau_0])
    x_ini = np.ravel(x_ini)  # intial values as long row vector
    # Ensure initial values are on the GPU
    x_ini = jax.device_put(x_ini)

    # from functools import partial
    # @partial(jax.jit, static_argnums=(0,))
    # https://docs.kidger.site/diffrax/examples/stiff_ode/
    def solve_differential_equation(PDE_sys, data, x_ini):
        """
        Solve a differential equation using a specified solver.

        Args:
            PDE_sys (dx.ODESystem): The ODE system.
            data (dict): Data containing parameters and time points.
            x_ini (numpy.ndarray): Initial conditions.

        Returns:
            dx.Solution: The solution to the differential equation.
        """
        # Define the ODE term using dx.ODETerm
        term = dx.ODETerm(PDE_sys)
        # Set up a PID controller for step size control
        stepsize_controller = dx.PIDController(pcoeff=0.2, icoeff=0.3,
                                               rtol=1e-12, atol=1e-12,
                                               # dtmax=1e-3
                                               )
        # Define when to save the results during integration
        saveat = dx.SaveAt(ts=data["t_points"])
        # root finder
        root_finder = dx.VeryChord(rtol=1e-12, atol=1e-12)
        # Set up the differential equation solver with relevant parameters
        sol = dx.diffeqsolve(term,
                             solver=dx.Kvaerno5(root_finder=root_finder),
                             t0=0,
                             t1=data["t_end"],
                             dt0=None,
                             y0=x_ini,
                             max_steps=2**30,
                             stepsize_controller=stepsize_controller,
                             saveat=saveat,
                             adjoint=dx.RecursiveCheckpointAdjoint(),
                             args=data
                             )
        return sol

    def handle_error(data, x_ini):
        """
        Handles errors during differential equation solving.

        Args:
            data (dict): Data containing parameters and time points.
            x_ini (numpy.ndarray): Initial conditions.
        """
        print("Error occurred during differential equation solving.")
        sol_ys = np.zeros((data["Nt"], data["Nz"] * 2))
        # Create a timestamp for the error report file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save initial values to NPY file
        # np.save(f"error_{timestamp}_initial_values.npy", x_ini)
        # with open(f"error_{timestamp}_data.pkl", "wb") as f:
        #     pickle.dump(data, f)
        return sol_ys

    try:
        time_start = time.time()
        sol = solve_differential_equation(PDE_sys, data, x_ini)
        time_end = time.time()
        time_elapsed = time_end-time_start
        print(f"Time for solving: {time_elapsed:.2f} seconds")

        # print(sol.stats)
        sol_ys = sol.ys
    except:
        sol = None
        sol_ys = handle_error(data, x_ini)


    ###########################################################################
    # unpack output vector and add boundary conditions
    ###########################################################################
    # row vector for molar fraction of i at left boundary for each time step
    X_CO2_LB = np.append(np.ones((1, 1)) * data["X_in"],
                         np.ones((1, np.size(data["t_points"]) - 1))
                         * data["X_in"])
    X_CO2_LB_inlet_channel = np.tile(X_CO2_LB, (data["Nz_all"]+1-data["Nz"], 1))
    T_LB = np.append(np.ones((1, 1)) * data["T_gas_in"],
                     np.ones((1, np.size(data["t_points"]) - 1))
                     * data["T_gas_in"])
    T_LB_inlet_channel = np.tile(T_LB, (data["Nz_all"]+1-data["Nz"], 1))
    # big matrices for molar fractions in each control volume and for
    # each time step
    X_sol = sol_ys[:, :data["Nz"]].T
    X_CO2 = np.vstack([X_CO2_LB, X_sol])
    tau_sol = sol_ys[:, data["Nz"]:2*data["Nz"]].T
    T_sol = tau_sol*data["T_scale"]+data["T_scale_add"]
    T = np.vstack([T_LB, T_sol])

    ###########################################################################
    # Get center temperature
    ###########################################################################
    T_center = get_center_temperature(X_CO2, T, data)
    print(f"Maximal average hotspot temperature is {np.max(T):.2f} Kelvin")
    print(f"Maximal center hotspot temperature is {np.max(T_center):.2f} Kelvin")

    ###########################################################################
    # Inlet Channel
    ###########################################################################
    X_CO2 = np.vstack([X_CO2_LB_inlet_channel, X_sol])
    T = np.vstack([T_LB_inlet_channel, T_sol])
    T_center_inlet_channel = np.tile(T_center[0, :], (data["Nz_all"]-data["Nz"], 1))
    T_center = np.vstack([T_center_inlet_channel, T_center])

    ###########################################################################
    # Get correction for conversion
    ###########################################################################
    y_pred = -0.022302522914144696 - 0.0061261*data["F_in_Nl_min"] + 0.0018558*data["T_cool"]
    correction_factor = y_pred/(X_CO2[-1, :]+eps)
    X_CO2_corrected = X_CO2*correction_factor.reshape(1, -1)
    X_CO2_corrected = replace_out_of_bounds_values(X_CO2_corrected,
                                                   threshold_value=1,
                                                   is_lower_bound=False)

    ###########################################################################
    # Plot results
    ###########################################################################
    # z-coordinate in each control volume
    dz = data["D_zeta"][0]*data["L_R"]
    z = np.zeros((data["Nz_all"]+1, 1))  # intialize empty vector z
    z = z.at[0].set(0)  # z-coordinate: reactor inlet [m]
    z = z.at[1].set(dz / 2)  # z-coordinate: middle of first control volume [m]
    for i in range(2, data["Nz_all"]+1):
        z = z.at[i].set(z[i-1] + dz)  # z-coordiante: other control volumes [m]

    # get part of fractions
    n_i = (data["n_in"][:, None, None]
           + data["nue"][:, None, None]*X_CO2_corrected*data["n_in"][1])

    # mole fraction
    n = np.sum(n_i, axis=0)
    molar_frac = n_i/n

    j_modulus = 100
    if dynamic_plot is True:
        # dynamic plot of molar fractions over reactor length
        for j in range(0, np.size(data["t_points"])):
            if j % j_modulus == 0:  # Show only every xth figure
                plt.figure(1)  # plot in figure 1
                plt.xlim(left=0, right=data["L_R"])
                plt.ylim(bottom=0.95*np.min(molar_frac),
                         top=1.05*np.max(molar_frac))
                currentPlotH2, = plt.plot(z, molar_frac[0, :, j],
                                          '-', c='lightseagreen', linewidth=3)
                currentPlotCO2, = plt.plot(z, molar_frac[1, :, j],
                                           '-', c='dimgrey', linewidth=3)
                currentPlotCH4, = plt.plot(z, molar_frac[2, :, j],
                                           '-', c='maroon', linewidth=3)
                currentPlotH2O, = plt.plot(z, molar_frac[3, :, j],
                                           '-', c='teal', linewidth=3)
                plt.legend(['H2', 'CO2', 'CH4', 'H2O'])
                plt.title('Mole-fraction over reactor length at t = '
                          + str(format_time(data["t_points"][j])) + ' min')
                plt.xlabel('Reactor Length z in m')
                plt.ylabel('Mole Fraction')
            plt.show()  # show graphic

        # dynamic plot of temperature over reactor length
        for j in range(0, np.size(data["t_points"])):
            if j % j_modulus == 0:  # Show only every xth figure
                plt.figure(2)
                plt.xlim(left=0, right=data["L_R"])
                plt.ylim(bottom=0.95*np.min(T), top=1.05*np.max(T))
                currentPlotT, = plt.plot(z, T[:, j], '-', linewidth=3)
                plt.title('Temperature over reactor length at t = '
                          + str(format_time(data["t_points"][j])) + ' min')
                plt.ylabel('Temperature in K')
                plt.xlabel('Reactor Length z in m')
            plt.show()  # show graphic

        for j in range(0, np.size(data["t_points"])):
            if j % j_modulus == 0:  # Show only every xth figure
                plt.figure(2)
                plt.xlim(left=0, right=data["L_R"])
                plt.ylim(bottom=0.95*np.min(X_CO2_corrected),
                         top=1.05*np.max(X_CO2_corrected))
                currentPlotT, = plt.plot(z, X_CO2_corrected[:, j],
                                         '-', linewidth=3)
                plt.title('CO2 conversion over reactor length at t = '
                          + str(format_time(data["t_points"][j])) + ' min')
                plt.xlabel('Reactor Length z in m')
                plt.ylabel('CO2 conversion')
            plt.show()  # show graphic

    # plot for concentrations
    plt.figure(3)
    plt.xlim(left=0, right=data["L_R"])
    plt.plot(z, molar_frac[0, :, -1], linewidth=3)
    plt.plot(z, molar_frac[1, :, -1], linewidth=3)
    plt.plot(z, molar_frac[2, :, -1], linewidth=3)
    plt.plot(z, molar_frac[3, :, -1], linewidth=3)
    plt.grid()
    plt.legend(['H2', 'CO2', 'CH4', 'H2O'])
    plt.xlabel('Reactor Length z in m')
    plt.ylim((0.95*np.min(molar_frac), 1.05*np.max(molar_frac)))
    plt.ylabel('Mole Fraction')
    plt.title('Mole Fraction over Reactor Length')
    plt.show()

    # plot for temperature over reactor length
    plt.figure(4)
    plt.xlim(left=0, right=data["L_R"])
    plt.ylim(bottom=0.95*np.min(T[:, -1]), top=1.05*np.max(T[:, -1]))
    plt.plot(z, T[:, -1], linewidth=3)
    plt.grid()
    plt.xlabel('Reactor Length z in m')
    plt.ylabel('Temperature in K')
    plt.title('Temperature over Reactor Length')
    plt.show()

    # plot for conversion over reactor length
    plt.figure(5)
    plt.xlim(left=0, right=data["L_R"])
    plt.ylim(bottom=0.9*np.min(X_CO2_corrected),
             top=1.05*np.max(X_CO2_corrected))
    plt.plot(z, X_CO2_corrected[:, -1], linewidth=3)
    plt.grid()
    plt.xlabel('Reactor Length z in m')
    plt.ylabel('CO2 conversion')
    plt.title('CO2 conversion over Reactor Length')
    plt.show()

    # plot for outlet temperature over time
    plt.figure(6)
    plt.plot(data["t_points"][10:], T[-1, 10:], linewidth=3)
    plt.ylim(bottom=np.min(T[-1, 10:]-1), top=np.max(T[-1, 10:])+1)
    plt.grid()
    plt.xlabel('Simulation time in s')
    plt.ylabel('Outlet Temperature in K')
    plt.title('Outlet Temperature over Time')
    plt.show()

    # plot for hotspot temperature over time
    plt.figure(7)
    T_max = np.max(T[:, 10:], axis=0)
    plt.plot(data["t_points"][10:], T_max, linewidth=3)
    plt.ylim(bottom=0.95*np.min(T_max), top=1.05*np.max(T_max))
    plt.grid()
    plt.xlabel('Simulation Time in s')
    plt.ylabel('Hotspot Temperature in K')
    plt.title('Hotspot Temperature over Time')
    plt.show()

    # plot for outlet conversion over time
    plt.figure(8)
    plt.plot(data["t_points"][1:], X_CO2_corrected[-1, 1:], linewidth=3)
    plt.grid()
    plt.xlabel('Simulation time in s')
    plt.ylabel('CO2 conversion')
    plt.title('Outlet CO2 conversion over Time')
    plt.show()

    ###########################################################################
    # Output
    ###########################################################################
    # Specify the data folder path
    results_folder = os.path.join(os.getcwd(), "results")

    # Ensure the data folder exists, create it if not
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    text = '_' + str(simulation_name)

    np.save(os.path.join(results_folder, 'time' + text + '.npy'),
            data["t_points"])
    np.save(os.path.join(results_folder, 'z' + text + '.npy'),
            z.flatten())
    np.save(os.path.join(results_folder, 'conversion' + text + '.npy'),
            X_CO2)
    np.save(os.path.join(results_folder, 'conversion_corrected' + text + '.npy'),
            X_CO2_corrected)
    np.save(os.path.join(results_folder, 'temperature' + text + '.npy'),
            T)
    np.save(os.path.join(results_folder, 'center_temperature' + text + '.npy'),
            T_center)
    np.save(os.path.join(results_folder, 'wall_temperature' + text + '.npy'),
            data["T_cool"])
    np.save(os.path.join(results_folder, 'load' + text + '.npy'),
            data["F_in_Nl_min"])

    ###########################################################################
    # Derivatives
    ###########################################################################
    if data["derivatives"]:
        # Calculate derivatives and save to file
        derivatives = np.array([PDE_sys(t, x, data)
                                for t, x in zip(data["t_points"], sol_ys)]).T
        derivatives_conversion = derivatives[0*data["Nz"]:1*data["Nz"]]
        derivatives_temperature = derivatives[1*data["Nz"]:2*data["Nz"]]
        np.save(os.path.join(results_folder,
                             'derivatives_conversion' + text + '.npy'),
                derivatives_conversion)
        np.save(os.path.join(results_folder,
                             'derivatives_temperature' + text + '.npy'),
                derivatives_temperature)
    return sol
