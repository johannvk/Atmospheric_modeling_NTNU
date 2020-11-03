import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt


def temperature_coefficient(k, Q_S_a, D, A_out, B_out):
    # Return coefficient of the 2*k'th Legendre polynomial 
    # for the 'Legendre' series of the temperature T(x).
    
    P_2k = Legendre.basis(2*k)
    integrand = lambda x: (Q_S_a(x) - A_out)*P_2k(x)

    I = quad(integrand, 0.0, 1.0)[0]

    return (4*k + 1.0)*I/(2*k*(2*k + 1)*D + B_out)


def test_temp_legendre_series(k_max=3):
    # Setup parameters:
    D = 2.0  # No clue what D should be!
    x_s = 0.22  # Ice cap seperator, x_s = sin(ϕₛ)
    Q = 1360/4.0  # Incoming Solar Flux [W m⁻²]
    
    A_out = 201.4  # [W m⁻²]
    B_out = 1.45   # [W m⁻² °C⁻¹]

    S_2 = -0.477
    S = lambda x: 1.0 + S_2*(3*x**2 - 1)/2.0 
    
    a_u, a_l = 0.38, 0.68
    
    def a(x):
        return a_u if x_s < x else a_l
    
    def Q_S_a(x):
        return Q*S(x)*a(x)

    # Find k_max + 1 coefficients:
    T_coefficients = np.zeros((2*k_max + 1))

    for k in range(k_max + 1):
        T_2k = temperature_coefficient(k, Q_S_a, D, A_out, B_out)
        T_coefficients[2*k] = T_2k

    T_func = Legendre(T_coefficients)

    # Plot solution on x ∈ [0, 1]:
    plot_xs = np.linspace(0, 1.0, int(1.0e2))
    T_vals = T_func(plot_xs)

    plt.plot(plot_xs, T_vals, label="T(x) Approximation.")
    plt.xlabel('x = sin(ϕ)', fontsize=18)
    plt.ylabel('T(x) = ∑ₙ₌₀ᴺ TₙPₙ(x)', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()
    pass

test_temp_legendre_series(k_max=5)