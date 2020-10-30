import numpy as np
import sympy as sy
from sympy.solvers.solveset import solveset, solveset_real
from sympy.solvers import solve

from atmospheric_model import radiation_scattering


def simple_model():
    # Base parameters. Sun incoming power, Stefan–Boltzmann constant.
    P_sun, sigma = sy.symbols("P_sun, sigma", real=True, positive=True)

    # Atmosphere variables:
    T_A, r_sm, a_sw, a_lw, a_O3, eps_A, f_A = sy.symbols("T_A, r_sm, a_sw, a_lw, a_O3, eps_A, f_A", 
                                                        real=True, positive=True)

    # Cloud variables:
    T_C, Cc, a_sc, r_sc, a_lc, r_lc = sy.symbols("T_C, Cc, a_sc, r_sc, a_lc, r_lc",
                                                real=True, positive=True)

    # Earth variables:
    T_E, r_se, r_le, eps_E = sy.symbols("T_E, r_se, r_le, eps_E", 
                                        real=True, positive=True)

    # Black body emission power for atmosphere, clouds and the eart:
    P_A_BB, P_C_BB, P_E_BB = sigma*T_A**4, sigma*T_C**4, sigma*T_E**4

    base_param_values = [(sigma, 5.670374e-8), (P_sun, 341.3)]

    atmosphere_param_values = [(r_sm, 0.1065), (a_sw, 0.1451), (a_lw, 0.8258), (a_O3, 0.08), (eps_A, 0.875), (f_A, 0.618)]

    cloud_param_values = [(Cc, 0.66), (a_sc, 0.1239), (r_sc, 0.22), (a_lc, 0.622), (r_lc, 0.195)]

    earth_param_values = [(r_se, 0.17), (r_le, 0.0), (eps_E, 1.0)]

    parameter_values = base_param_values + atmosphere_param_values + \
                    cloud_param_values + earth_param_values
    
    P_A_refl, P_A_abs, P_A_trans = radiation_scattering(P_sun, a_sw, r_sm)
    res = sy.simplify(P_sun - (P_A_refl + P_A_abs + P_A_trans))
    
    atmosphere_equation = sy.Eq(P_A_abs, eps_A*P_A_BB).subs(parameter_values)
    earth_equation = sy.Eq(P_A_trans + f_A*eps_A*P_A_BB, P_E_BB + r_se*P_A_trans).subs(parameter_values)

    ans = solve([atmosphere_equation, earth_equation], [T_A, T_E])
    # t_e_num = ans[0][T_E].subs(parameter_values)
    print("Answer:", ans)
    # print("Testing!")

