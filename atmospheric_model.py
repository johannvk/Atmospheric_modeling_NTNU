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


def main():

    # cloud_atmosphere_model()
    # from models import simple_model

    # simple_model()
    pass


if __name__ == "__main__":
    main()
