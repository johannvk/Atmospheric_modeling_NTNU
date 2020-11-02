import numpy as np
import sympy as sy


# Base parameters. Sun incoming power, Stefan–Boltzmann constant.
P_sun, sigma = sy.symbols("P_sun, sigma", real=True, positive=True)

# Atmosphere variables:
T_A, r_sm, a_sw, a_lw, a_O3, eps_A, f_A = sy.symbols("T_A, r_sm, a_sw, a_lw, a_O3, eps_A, f_A",
                                                    real=True, positive=True)

# Cloud variables:
T_C, Cc, a_sc, r_sc, a_lc, r_lc = sy.symbols("T_C, Cc, a_sc, r_sc, a_lc, r_lc",
                                            real=True, positive=True)

# Earth variables:
T_E, a_se, r_se, a_le, r_le, eps_E = sy.symbols("T_E, a_se, r_se, a_le, r_le, eps_E",
                                    real=True, positive=True)

# Effective Black body emission power for atmosphere, clouds and the eart:
tau_A, tau_C, tau_E = eps_A*sigma*T_A**4, eps_A*sigma*T_C**4, sigma*T_E**4

# Useful variables for defining our model:
a_sa = sy.Integer(1) - (sy.Integer(1) - a_sw)*(sy.Integer(1) - a_O3)
a_la = a_sw
r_la = sy.Integer(0)

geometric_reflection = sy.Integer(1)/(sy.Integer(1) - r_sc*r_sm)

base_param_values = [(sigma, 5.670374e-8), (P_sun, 341.3)]

atmosphere_param_values = [(r_sm, 0.1065), (a_sw, 0.1451), (a_lw, 0.8258), (a_O3, 0.08), (eps_A, 0.875), (f_A, 0.618)]

cloud_param_values = [(Cc, 0.66), (a_sc, 0.1239), (r_sc, 0.22), (a_lc, 0.622), (r_lc, 0.195)]

earth_param_values = [(a_se, 1.0), (r_se, 0.17), (a_le, 1.0), (r_le, 0.0), (eps_E, 1.0)]

parameter_values = base_param_values + atmosphere_param_values + \
                cloud_param_values + earth_param_values
