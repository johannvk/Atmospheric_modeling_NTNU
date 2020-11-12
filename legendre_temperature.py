import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20.0})

class IceCapTemperature():

    def __init__(self, x_s_init, Q_init, num_terms=4):
        
        # From paper, empirically fitted to measured T(x):
        self.D = 0.649  # [W m⁻²]
        self.x_s = x_s_init  # Ice cap seperator, x_s = sin(ϕₛ)
        self.Q = Q_init  # Incoming Solar Flux [W m⁻²]
        
        # From [North 1975]:
        self.A_out = 201.4  # [W m⁻²]
        self.B_out = 1.45   # [W m⁻² °C⁻¹]

        # From [North and Coakley, 1979]:
        # self.A_out = 203.3 # [W m⁻²] 
        # self.B_out = 2.09  # [W m⁻² °C⁻¹]

        # self.S_0 = 0.78539816  # Egen-utregnede coeffisienter i Legendre-approksimasjonen til S(x)
        # self.S_2 = -0.49087385 # -0.477
        self.S_0 = 1.0
        self.S_2 = -0.477
        self.S = lambda x: self.S_0*1.0 + self.S_2*(3*x**2 - 1)/2.0 
        
        # self.S = lambda x: (4/np.pi)*np.sqrt(1 - x**2)
        self.a_u, self.a_l = 0.38, 0.68
        self.num_terms = num_terms
        self.ice_cap_temperature = -10  # °C
    
    def a_crack(self, x, x_s=None):
        if x_s is None: x_s = self.x_s
        return self.a_u if x_s < x else self.a_l

    def H(self, x, beta=0.01):
        return np.arctan(x/beta)/np.pi + 0.5

    def a_smooth(self, x, x_s=None, beta=0.01):
        if x_s is None: x_s = self.x_s
        # Smoother approximation to the ice cap albedo function:
        return self.a_l*(1 - self.H(x - x_s, beta)) + self.a_u*self.H(x - x_s, beta)

    def S_a(self, x, x_s=None, beta=0.001):
        if x_s is None: x_s = self.x_s
        return self.S(x)*self.a_smooth(x, x_s, beta)

    def legendre_series_T(self, Q=None, x_s=None, Q_derivative=False):
        if Q is None: Q = self.Q
        if x_s is None: x_s = self.x_s

        k_max = self.num_terms
        
        # Find k_max + 1 coefficients:
        T_coefficients = np.zeros((2*k_max + 1))

        for k in range(0, k_max + 1):
            # k, S_a, D, A_out, B_out, Q, x_s
            T_2k = temperature_coefficient(k, self.S_a, self.D, self.A_out, self.B_out, 
                                           Q=Q, x_s=x_s, Q_derivative=Q_derivative)
            T_coefficients[2*k] = T_2k

        return Legendre(T_coefficients)

    def T(self, x, Q, x_s=None, Q_derivative=False):
        if x_s is None: x_s = self.x_s
        return self.legendre_series_T(Q=Q, x_s=x_s, Q_derivative=Q_derivative)(x)    

    def ice_cap_Q(self, x_s, atol=1.0e-3, max_iter=50):
        
        # Newton method for root-finding:
        # Q_0 = self.Q
        # Initialize to low value of incoming power:
        Q_0 = 100
        T_x_s = self.T(x=x_s, Q=Q_0, x_s=x_s)

        iteration = 0
        while abs(T_x_s - self.ice_cap_temperature) > atol and iteration < max_iter:
            iteration += 1

            Q_0 -= (self.T(x=x_s, Q=Q_0, x_s=x_s) - self.ice_cap_temperature)/self.T(x=x_s, Q=Q_0, x_s=x_s, Q_derivative=True)
            T_x_s = self.T(x=x_s, Q=Q_0, x_s=x_s)
        
        if iteration >= max_iter:
            raise Exception(f"Max iterations reached in Newton-Iteration for T(x_s|Q) = 0, \nwith T_value {T_x_s}.")
 
        return Q_0
    
    def find_Q_xs(self, x_s_array: np.ndarray, atol=1.0e-3):
        Q_array = np.zeros(x_s_array.shape)

        for i, x_s in enumerate(x_s_array):
            if (i % 10) == 0:
                print(f"At step {i}...")
            Q_array[i] = self.ice_cap_Q(x_s, atol=atol)
        
        return Q_array

    def display_Q_xs(self, exact=False):
        x_s_array = np.linspace(0.0, 1.0, num=int(2.0e2))
        
        if exact:
            Q_array = np.zeros(x_s_array.shape)
            P_2 = Legendre.basis(2)

            for i, x_s in enumerate(x_s_array):
                numerator = (self.A_out + self.ice_cap_temperature*self.B_out)
                H_0 = quad(lambda x: self.S_a(x, x_s), a=0.0, b=1.0)[0]
                H_2 = (2*2.0 + 1)*quad(lambda x: self.S_a(x, x_s)*P_2(x), 0.0, 1.0)[0]
                Q_array[i] = numerator/(H_0 + (self.B_out*H_2/(6*self.D +  self.B_out))*P_2(x_s))

        else:
            Q_array = self.find_Q_xs(x_s_array)
        fig, ax = plt.subplots(figsize=(14, 12))

        if exact:
            fig.suptitle(r"Two term analytic approx. to $Q$ required for $T(x_s|x_s, Q) = T_{Ice\,Cap}$")
        else:
            fig.suptitle(f"{self.num_terms} term "+ r"num. approx. to $Q$ required for $T(x_s|x_s, Q) = T_{Ice\,Cap}$")

        ax.plot(x_s_array, Q_array, label=r"$Q(x_s)$")

        ax.axhline(y=self.Q, xmin=0.0, xmax=1.0, label=r"Initial $Q$")
        ax.axvline(x=self.x_s, linewidth=2, color='b', label=r"Initial $x_s$")

        ax.set_xlabel(r"$x_s$")
        ax.set_ylabel(r"$Q(x_s)$")

        plt.legend(loc=(0.5, 0.5))
        plt.grid()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.92)
        plt.show()

    def display_temperature(self, x_s=None, Q=None):
        if x_s is None: x_s = self.x_s
        if Q is None: Q = self.Q

        # Plot solution on x ∈ [0, 1]:
        plot_xs = np.linspace(0, 1.0, int(1.0e3))
        T = self.legendre_series_T(Q=Q, x_s=x_s)
        T_vals = T(plot_xs)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"{self.num_terms} term " + r"Legendre Polynomail approximation to $T(x | x_s, Q)$")
        ax.plot(plot_xs, T_vals, "r--",label="T(x) Approximation.")
        ax.set_xlabel('x = sin(ϕ)')
        ax.set_ylabel(r'$T(x | x_s, Q) \approx \sum_{n=0}^{N} T_nP_n(x)$,' + " [°C]")

        ax.text(x=0.1, y=0.165, s=f"Incoming solar Power Q: {self.Q} [W m⁻²]")
        ax.axhline(y=self.ice_cap_temperature, xmin=0.0, xmax=1.0, label="Ice cap Temp.")
        ax.axvline(x=x_s, linewidth=2, color='b', label="x_s")
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.99)
        plt.legend()
        plt.grid()
        # plt.tight_layout()
        plt.show()
        


def temperature_coefficient(k, S_a, D, A_out, B_out, Q, x_s, Q_derivative=False):
    # Return coefficient of the 2*k'th Legendre polynomial 
    # for the 'Legendre' series of the temperature T(x).
    
    P_2k = Legendre.basis(2*k)

    integrand = lambda x: S_a(x, x_s)*P_2k(x)
    
    H_2k = (4*k + 1.0)*quad(integrand, 0.0, 1.0)[0]
    L_2k = (2*k*(2*k + 1)*D + B_out)
    
    if Q_derivative:
        return H_2k/L_2k
    elif k == 0:
        return Q*H_2k/L_2k - A_out/B_out
    else:
        return Q*H_2k/L_2k


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

    def S_a(x, x_s):
        return S(x)*a_smooth(x)

    # Find k_max + 1 coefficients:
    T_coefficients = np.zeros((2*k_max + 1))

    for k in range(0, k_max + 1):
        T_2k = temperature_coefficient(k, S_a, D, A_out, B_out, Q=Q, x_s=x_s)
        T_coefficients[2*k] = T_2k

    T_func = Legendre(T_coefficients)

    # Plot solution on x ∈ [0, 1]:
    plot_xs = np.linspace(0, 1.0, int(1.0e3))

    # Plot a(x, x_s):
    # plt.plot(plot_xs, a_crack(plot_xs), "r--", label="Crack a(x, x_s)")
    # plt.plot(plot_xs, a_smooth(plot_xs, beta=0.01), label="Smooth a(x, x_s)")
    # plt.ylim(0, 1.0)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

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



if __name__ == "__main__":
    x_s = 0.95  # Ice cap seperator, x_s = sin(ϕₛ)
    Q = 1*1360/4.0  # Incoming Solar Flux [W m⁻²]
    a = IceCapTemperature(x_s_init=x_s, Q_init=Q, num_terms=4)

    # Free boundary condition at x=0: P'_2n(x=0) = 0.
    # Jevne Legendre-polynom har derivert lik null i x = 0.
    a.display_temperature(x_s=x_s, Q=Q)
    a.display_Q_xs(exact=False)
    a.display_Q_xs()
   

