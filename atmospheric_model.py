import numpy as np
import sympy as sy
from sympy.solvers.solveset import solveset, solveset_real
from sympy.solvers import solve


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


def cloud_atmosphere_model():
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

    # Effective Black body emission power for atmosphere, clouds and the eart:
    tau_A, tau_C, tau_E = eps_A*sigma*T_A**4, eps_A*sigma*T_C**4, sigma*T_E**4

    # Useful variables for defining our model:
    a_sa = sy.Integer(1) - (sy.Integer(1) - a_sw)*(sy.Integer(1) - a_O3)
    geometric_reflection = sy.Integer(1)/(sy.Integer(1) - r_sc*r_sm)

    base_param_values = [(sigma, 5.670374e-8), (P_sun, 341.3)]

    atmosphere_param_values = [(r_sm, 0.1065), (a_sw, 0.1451), (a_lw, 0.8258), (a_O3, 0.08), (eps_A, 2*0.875), (f_A, 0.5)]

    cloud_param_values = [(Cc, 0.66), (a_sc, 0.1239), (r_sc, 0.22), (a_lc, 0.622), (r_lc, 0.195)]

    earth_param_values = [(r_se, 0.17), (r_le, 0.0), (eps_E, 1.0)]

    parameter_values = base_param_values + atmosphere_param_values + \
                    cloud_param_values + earth_param_values

    def earth_equation():
        # 12CCτC+ [(1−CC) +CC(1−rLC)(1−aLC)]fAτA+CCrLCτE
        l_E = sy.Rational(1, 2)*Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*f_A*tau_A + Cc*r_lc*tau_E
        s_E = P_sun*(1-r_sm)*(1-a_sa)*((1-Cc)*(1-r_se) + Cc*(1-r_sc)*(1-a_sc)*geometric_reflection*(1-r_se)/(1-r_se*r_sc))*(1-r_se)

        return sy.Eq(l_E + s_E, tau_E + sy.Integer(14*7))

    def cloud_equation():
        l_C = Cc*tau_E + Cc*f_A*tau_A
        # P0S(1−rSM)(1−αSA)CC1−rSC1−rSCrSMαSC(1 + (1−αSC)rSE1−rSC1−rSCrSE)
        s_C = P_sun*(1 - r_sm)*(1 - a_sa)*Cc*(1 - r_sc)*geometric_reflection*a_sc*(1 + (1 - a_sc)*r_se*(1 - r_sc)/(1-r_sc*r_se))

        return sy.Eq(l_C + s_C, tau_C)

    def atmosphere_equation():

        l_A = sy.Rational(1, 2)*Cc*tau_C + ((1 - Cc) + Cc*(1 - r_lc)*(1 - a_lc))*tau_E + Cc*r_lc*f_A*tau_A

        s_A = P_sun*(1 - r_sm)*(a_sa + (1-a_sa)*Cc*r_sc*(1-r_sm)*geometric_reflection*a_sa + (1-a_sa)*(1-Cc)*r_se*a_sa)

        return sy.Eq(l_A + s_A + sy.Integer(14*7), tau_A)


    Earth_eq = earth_equation().subs(parameter_values)
    Cloud_eq = cloud_equation().subs(parameter_values)
    Atmosphere_eq = atmosphere_equation().subs(parameter_values)

    print(earth_equation, atmosphere_equation,cloud_equation)

    ans = solve([Earth_eq, Cloud_eq, Atmosphere_eq], [T_A, T_C, T_E])
    print("Ans in Kelvin :", ans)
    print("Ans in Celsius:", [T-273.15 for T in ans[0]])
    print("(Atmosphere, cloud, earth respectively)")



def main():

    cloud_atmosphere_model()
    # from models import simple_model

    # simple_model()


if __name__ == "__main__":
    main()
