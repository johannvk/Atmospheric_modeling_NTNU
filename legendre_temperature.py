import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt


def temperature_coefficient(k, Q_S_a, D, A_out, B_out):
    # Return coefficient of the 2*k'th Legendre polynomial 
    # for the 'Legendre' series of the temperature T(x).
    
    P_2k = Legendre.basis(2*k)
    integrand = lambda x: (Q_S_a(x) - A_out)*P_2k(x)

    I, _ = quad(integrand, 0.0, 1.0)

    return (4*k + 1.0)*I/(2*k*(2*k + 1)*D + B_out)


def test_temp_legendre_series(k_max=3):
    # Setup parameters:
    R = 6.378e6  # [m]
    K_a = 0.026  # 2000 meters: 0.02276, Temp: 259.31 K, 
                   # source: https://en.wikipedia.org/wiki/List_of_thermal_conductivities
    D = K_a/R**2  # No clue what D should be!
    x_s = 0.5  # Ice cap seperator, x_s = sin(ϕₛ)
    Q = 1*1360/4.0  # Incoming Solar Flux [W m⁻²]
    
    A_out = 201.4  # [W m⁻²]
    B_out = 1.45   # [W m⁻² °C⁻¹]

    S_0 = 0.78539816  # Egen-utregnede coeffisienter i Legendre-approksimasjonen til S(x)
    S_2 = -0.49087385 # -0.477
    S = lambda x: S_0*1.0 + S_2*(3*x**2 - 1)/2.0 
    
    a_u, a_l = 0.38, 0.68
    
    def a_crack(x):
        return a_u if x_s < x else a_l
    a_crack = np.vectorize(a_crack)

    def H(x, beta):
        return np.arctan(x/beta)/np.pi + 0.5

    def a_smooth(x, beta=0.01):
        # Smoother approximation to the ice cap albedo function:
        return a_l*(1 - H(x - x_s, beta)) + a_u*H(x - x_s, beta)

    def Q_S_a(x):
        return Q*S(x)*a_smooth(x)

    # Find k_max + 1 coefficients:
    T_coefficients = np.zeros((2*k_max + 1))

    for k in range(0, k_max + 1):
        T_2k = temperature_coefficient(k, Q_S_a, D, A_out, B_out)
        T_coefficients[2*k] = T_2k

    T_func = Legendre(T_coefficients)

    # Plot solution on x ∈ [0, 1]:
    plot_xs = np.linspace(0, 1.0, int(1.0e3))

    plt.plot(plot_xs, a_crack(plot_xs), "r--", label="Crack a(x, x_s)")
    plt.plot(plot_xs, a_smooth(plot_xs, beta=0.01), label="Smooth a(x, x_s)")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    T_vals = T_func(plot_xs)

    plt.plot(plot_xs, T_vals, label="T(x) Approximation.")
    plt.xlabel('x = sin(ϕ)', fontsize=18)
    plt.ylabel('T(x) = ∑ₙ₌₀ᴺ TₙPₙ(x)', fontsize=18)
    plt.axvline(x=x_s, linewidth=2, color='b', label="x_s")
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()
    pass


def legendre_series(func, max_degree=6, a=-1.0, b=1.0):
    legendre_coefs = np.zeros((max_degree+1))
    
    for k in range(0, max_degree + 1):
        integrand = lambda x: func(x)*Legendre.basis(k)(x)
        legendre_coefs[k] = ((2*k + 1)/2.0)*quad(integrand, a=a, b=b)[0]

    return Legendre(legendre_coefs)


def geometric_tempt_distribution():

    func = lambda x: np.sqrt(1 - x**2)

    legendre_func = legendre_series(func, max_degree=4)
    plot_xs = np.linspace(-1.0, 1.0, 1000)
    plt.plot(plot_xs, func(plot_xs), label="Original")
    plt.plot(plot_xs, legendre_func(plot_xs), label="Legendre Series")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("What what!")


test_temp_legendre_series(k_max=8)
# geometric_tempt_distribution()