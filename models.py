import numpy as np
import sympy as sy
from sympy.solvers.solveset import solveset, solveset_real
from sympy.solvers import solve

from atmospheric_model import radiation_scattering

# Import variables:
from parameters import *


def simple_model():

    # Incoming radiation from the sun scattering in the atmosphere:
    sw_A_refl, sw_A_abs, sw_A_trans = radiation_scattering(P_sun, a_sa, r_sm)

    # Short-wave radiation hitting the earth:    
    sw_E_refl, sw_E_abs, sw_E_trans = radiation_scattering(sw_A_trans, a=sy.Integer(1), r=r_se)
    
    # Short-wave radiation hitting the atmosphere again:
    sw_E_to_A_refl, sw_E_to_A_abs, sw_E_to_A_trans = radiation_scattering(sw_E_refl, a=a_sa, r=r_sm)
    # Ignore further reflection of SW radiation down to earth again.

    # Long wave reflected radiation from the earth hitting the Atmosphere:
    # No ozone absorption for long-wave radiation. Only a_lw. 
    # Also assume the same molecular scattering constant r_sm for long waves.
    lw_A_refl, lw_A_abs, lw_A_trans = radiation_scattering(tau_E, a=a_lw, r=r_sm)
    
    # Incoming power = Outgoing power
    sw_in_total_A = sw_A_abs + sw_E_to_A_abs
    lw_in_total_A = lw_A_abs
    atmosphere_equation = sy.Eq(sw_in_total_A + lw_in_total_A, tau_A)

    # Long wave BB radiation from the atmosphere fully absorbed: r_le = 0.0 
    # Incoming power = Outgoing power
    sw_in_total_E = sw_E_abs
    lw_in_total_E = f_A*tau_A  # + lw_A_refl  # Might ignore reflected lw_A-radiation.
    earth_equation = sy.Eq(sw_in_total_E + lw_in_total_E, tau_E)
    
    # Solve symbolically for T_A and T_E: Can then take derivatives of parameters!
    symb_ans = solve([atmosphere_equation, earth_equation], [T_A**4, T_E**4], dict=True) 
    symb_T_E = symb_ans[0][T_E**4]**(1/4)
    symb_T_A = symb_ans[0][T_A**4]**(1/4)
    
    num_ans = solve([atmosphere_equation.subs(parameter_values), earth_equation.subs(parameter_values)], 
                [T_A, T_E], dict=True)

    t_e_Celsius = num_ans[0][T_E] - 273.15

    print("Earth temperature in Celsius:\t{:.3f}".format(t_e_Celsius))


def downward_pass_cloud_model():
    # Using fractional decomposing of Black Body radiation from Atmosphere and Clouds.
    # Only a single downward pass, do not account for reflections upward again.

    # Incoming radiation from the sun scattering in the atmosphere:
    sw_A_refl, sw_A_abs, sw_A_trans = radiation_scattering(P_sun, a_sa, r_sm)

    # Short wave scattering A -> C:
    sw_A_to_C_refl, sw_A_to_C_abs, sw_A_to_C_trans = radiation_scattering(Cc*sw_A_trans, a_sc, r_sc)

    # Short wave scattering A -> E:
    sw_A_to_E_refl, sw_A_to_E_abs, sw_A_to_E_trans = radiation_scattering((1 - Cc)*sw_A_trans, a_se, r_se)

    # Short wave scattering C -> E:
    sw_C_to_E_refl, sw_C_to_E_abs, sw_C_to_E_trans = radiation_scattering(sw_A_to_C_trans, a_se, r_se)

    # Downward facing Black Body power from the atmosphere:
    P_down_BB_A = f_A*tau_A

    # Black Body scattering A -> C:
    bb_A_to_C_refl, bb_A_to_C_abs, bb_A_to_C_trans = radiation_scattering(Cc*P_down_BB_A, a_lc, r_lc)

    # Black Body scattering A -> E:
    bb_A_to_E_refl, bb_A_to_E_abs, bb_A_to_E_trans = radiation_scattering((1 - Cc)*P_down_BB_A, a_le, r_le)

    # Black Body scattering C -> A:
    bb_C_to_A_refl, bb_C_to_A_abs, bb_C_to_A_trans = radiation_scattering(sy.Rational(1/2)*tau_C, a_la, r_la)

    # Black Body scattering C -> E:
    bb_C_to_E_refl, bb_C_to_E_abs, bb_C_to_E_trans = radiation_scattering(sy.Rational(1/2)*tau_C, a_le, r_le)
    
    # Black Body scattering E -> C:
    bb_E_to_C_refl, bb_E_to_C_abs, bb_E_to_C_trans = radiation_scattering(Cc*tau_E, a_lc, r_lc)

    # Black Body scattering E -> A:
    bb_E_to_A_refl, bb_E_to_A_abs, bb_E_to_A_trans = radiation_scattering((1 - Cc)*tau_E, a_la, r_la)

    # Long wave scattering C -> E:


    # Incoming Power = Outgoing Power:
    atmosphere_equation = sy.Eq(sw_A_abs + bb_C_to_A_abs + bb_E_to_A_abs, tau_A)

    cloud_equation = sy.Eq(sw_A_to_C_abs + bb_A_to_C_abs + bb_E_to_C_abs, tau_C)

    earth_equation = sy.Eq(sw_A_to_E_abs + sw_C_to_E_abs + bb_C_to_E_abs + bb_A_to_E_abs, tau_E)
    equations = [atmosphere_equation, cloud_equation, earth_equation]
    subs_equations = [eq.subs(parameter_values) for eq in equations]
    print(subs_equations)
    
    ans = solve(subs_equations, [T_A**4, T_C**4, T_E**4])
    
    if type(ans) is dict:
        ans["T_E_Celsius"] = ans[T_E**4]**(1/4) - 273.15
        ans["T_C_Celsius"] = ans[T_C**4]**(1/4) - 273.15
        ans["T_A_Celsius"] = ans[T_A**4]**(1/4) - 273.15

    print("Ans:\n", ans)


def cloud_atmosphere_model():
    # f_A = 2*f_A
    def earth_equation():
        # 12CCτC+ [(1−CC) +CC(1−rLC)(1−aLC)]fAτA+CCrLCτE
        l_E = sy.Rational(1, 2)*Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*f_A*tau_A + Cc*r_lc*tau_E
        s_E = P_sun*(1-r_sm)*(1-a_sa)*((1-Cc)*(1-r_se) + Cc*(1-r_sc)*(1-a_sc)*geometric_reflection*(1-r_se)/(1-r_se*r_sc))*(1-r_se)

        return sy.Eq(l_E + s_E, tau_E)

    def cloud_equation():
        l_C = Cc*tau_E + Cc*f_A*tau_A
        # P0S(1−rSM)(1−αSA)CC1−rSC1−rSCrSMαSC(1 + (1−αSC)rSE1−rSC1−rSCrSE)
        s_C = P_sun*(1 - r_sm)*(1 - a_sa)*Cc*(1 - r_sc)*geometric_reflection*a_sc*(1 + (1 - a_sc)*r_se*(1 - r_sc)/(1-r_sc*r_se))

        return sy.Eq(l_C + s_C, tau_C)

    def atmosphere_equation():

        l_A = sy.Rational(1, 2)*Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*tau_E + Cc*r_lc*f_A*tau_A

        s_A = P_sun*(1 - r_sm)*(a_sa + (1-a_sa)*Cc*r_sc*(1-r_sm)*geometric_reflection*a_sa + (1-a_sa)*(1-Cc)*r_se*a_sa)

        return sy.Eq(l_A + s_A, tau_A)


    Earth_eq = earth_equation().subs(parameter_values)
    Cloud_eq = cloud_equation().subs(parameter_values)
    Atmosphere_eq = atmosphere_equation().subs(parameter_values)

    ans = solve([Earth_eq, Cloud_eq, Atmosphere_eq], [T_A, T_C, T_E])
    print("Ans in Kelvin :", ans)
    print("Ans in Celsius:", [T-273.15 for T in ans[0]])
    print("(Atmosphere, cloud, earth respectively)")


def atmospheric_model():
    print(parameter_values)
    pass


if __name__ == "__main__":
    # simple_model()
    # atmospheric_model()
    downward_pass_cloud_model()
    # cloud_atmosphere_model()
    pass
