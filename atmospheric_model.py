import numpy as np
import sympy as sy
from sympy.solvers.solveset import solveset, solveset_real
from sympy.solvers import solve

from scipy.optimize import fsolve
from sympy.utilities.lambdify import lambdify

# Import all the necessary parameters:
from parameters import *


def radiation_scattering(P_in, a, r):
    """
    Returns the amount of power 'P_in' reflected,
    absorbed and transmitted from a zero-dimensional
    interaction, with very simple dynamics.
    P_in: Incoming power
    a: Absorption coefficient
    r: Reflection coefficient
    """
    P_refl = r*P_in
    P_abs = (sy.Integer(1) - r)*a*P_in
    P_trans = (sy.Integer(1) - r)*(sy.Integer(1) - a)*P_in
    return P_refl, P_abs, P_trans


def scattering_mechanics_model(nudge=False, i=0):

    # Downward pass of short wave radiation:
    sw_S_to_A_refl, sw_S_to_A_abs, sw_S_to_A_trans = radiation_scattering(P_sun, a_sw, r_sm)

    # Cc * P_S_to_A_trans Incoming sw radiation to Clouds:
    sw_A_to_C_refl, sw_A_to_C_abs, sw_A_to_C_trans = radiation_scattering(Cc*sw_S_to_A_trans, a_sc, r_sc)

    # (1 - Cc) * P_S_to_A_trans Incoming sw radiation to Earth: 
    sw_A_to_E_refl, sw_A_to_E_abs, sw_A_to_E_trans = radiation_scattering((1 - Cc)*sw_S_to_A_trans, a_se, r_se)

    # Incoming sw radiation from Cloud to earth:
    sw_C_to_E_refl, sw_C_to_E_abs, sw_C_to_E_trans = radiation_scattering(sw_A_to_C_trans, a_se, r_se)
    
    # Total incoming sw radiation to earth:
    sw_E_refl_tot = sw_A_to_E_refl + sw_C_to_E_refl

    # Reflected sw radiation from the earth to clouds: Cc*sw_E_refl_tot
    sw_E_to_C_refl, sw_E_to_C_abs, sw_E_to_C_trans = radiation_scattering(Cc*sw_E_refl_tot, a_sc, r_sc)

    # TODO: One more sw reflecting + Long wave outgoing BB radiation and reflections of lw radiation.
    pass
    


def cloud_atmosphere_model(temperature_difference, nudge=False, i=0):
    if nudge:
        parameter_values[i] = (parameter_values[i][0], parameter_values[i][1]*1.01)

    def earth_equation():
        # 12CCτC+ [(1−CC) +CC(1−rLC)(1−aLC)]fAτA+CCrLCτE
        l_E = Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*f_A*tau_A + Cc*r_lc*tau_E
        s_E = P_sun*(1-r_sm)*(1-a_sa)*((1-Cc)*(1-r_se) + Cc*(1-r_sc)*(1-a_sc)*geometric_reflection*(1-r_se)/(1-r_se*r_sc))*(1-r_se)

        equation = sy.Eq(l_E + s_E, tau_E + sy.Float(temperature_difference*7))
        expr = l_E + s_E - (tau_E + (beta+Cc*alpha)*E_C_temp_diff + (1-Cc)*alpha*E_A_temp_diff)
        return equation, expr

    def cloud_equation():
        l_C = Cc*tau_E + Cc*f_A*tau_A
        # P0S(1−rSM)(1−αSA)CC1−rSC1−rSCrSMαSC(1 + (1−αSC)rSE1−rSC1−rSCrSE)
        s_C = P_sun*(1 - r_sm)*(1 - a_sa)*Cc*(1 - r_sc)*geometric_reflection*a_sc*(1 + (1 - a_sc)*r_se*(1 - r_sc)/(1-r_sc*r_se))

        equation = sy.Eq(l_C + s_C + sy.Float(temperature_difference*7), 2*tau_C)
        expr = l_C*a_lc + s_C + (beta+Cc*alpha)*E_C_temp_diff - 2*tau_C
        return equation, expr

    def atmosphere_equation():

        l_A = Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*tau_E + Cc*r_lc*f_A*tau_A

        s_A = P_sun*(1 - r_sm)*(a_sa + (1-a_sa)*Cc*r_sc*(1-r_sm)*geometric_reflection*a_sa + (1-a_sa)*(1-Cc)*r_se*a_sa)
        equation = sy.Eq(l_A + s_A, 2*tau_A)
        expr = l_A*a_lw + s_A + (1-Cc)*alpha*E_A_temp_diff - 2*tau_A
        return equation, expr


    Earth_eq, earth_expr = earth_equation()
    Cloud_eq, cloud_expr = cloud_equation()
    Atmosphere_eq, atm_expr = atmosphere_equation()

    # Equations and expressions:
    return (Atmosphere_eq, Cloud_eq, Earth_eq), (atm_expr, cloud_expr, earth_expr)


def solve_with_heat(tol = 0.001, nudge = False, i=0):
    diff = 15.6728951112
    for j in range (20):
        print('j=',j)
        ans = cloud_atmosphere_model(diff, nudge, i)[0]
        new_diff = ans[2]-ans[1]
        diff = diff + (new_diff - diff)/2
        #print('diff:', diff)
        if abs(diff - new_diff)<tol:
            break
    #print("Ans in Celsius:", [T-273.15 for T in ans])
    return ans


def solve_temperatures(expressions, nudge=False, i=0):
    atm_expr, cloud_expr, earth_expr = expressions

    if nudge:
        nudge_tup = (parameter_values[i][0], parameter_values[i][1]*1.01)
        subs_list = [nudge_tup] + parameter_values
    else:
        subs_list = parameter_values

    atm_func = lambdify([T_E, T_A, T_C], atm_expr.subs(subs_list), modules='numpy')
    cloud_func = lambdify([T_E, T_A, T_C], cloud_expr.subs(subs_list), modules='numpy')
    earth_func = lambdify([T_E, T_A, T_C], earth_expr.subs(subs_list), modules='numpy')

    def objective(p):
        t_a, t_c, t_e = p
        return np.array([atm_func(t_e, t_a, t_c), cloud_func(t_e, t_a, t_c), earth_func(t_e, t_a, t_c)])

    # Solutions: In order T_A, T_C, T_E:
    return fsolve(objective, x0=np.ones(3)*273.15)

def sensitivity_analysis(tol = 0.00001):
    param_names = ['sigma','psun','r_sm', 'a_sw', 'a_lw', 'a_03', 'eps_A', 'f_a',
                   'cc','a_sc', 'r_sc', 'a_lc', 'r_lc','r_se', 'r_le', 'eps_e', 'alpha','beta']

    model_eq, model_expr = cloud_atmosphere_model(temperature_difference=15, nudge=False)

    orig_ans = solve_temperatures(model_expr)

    print(f"Model temperatures in Celsius: \nT_A: {orig_ans[0]-273.15}, \
            \tT_C: {orig_ans[1]-273.15},\tT_E: {orig_ans[2]-273.15}.\n")

    for i in range(len(parameter_values)):

        print('i=', i, f"Nudging parameter: {param_names[i]}.")
        nudge_ans = solve_temperatures(model_expr, nudge=True, i=i)

        percent_change = (np.array(nudge_ans)-np.array(orig_ans))/np.array(orig_ans)*100
        print(f"Relative change in temperatures in %:\t {percent_change}\n")


def main():
    sensitivity_analysis()

if __name__ == "__main__":
    main()
