# Python module containing solutions for the growth of 2D plane-strain 
# fluid-driven shear ruptures in a planar fault with uniform and constant
# properties. 
# The friction coefficient is  constant
# The hydraulic properties (permeability and storage) are constant
# There is shear-induced dilatancy.
# References:
#  Dunham EM. 2024 Fluid driven aseismic fault slip with permeability enhancement and dilatancy.

# notations:
#       T: fault stress parameter defined as (f\sigma'_o - \tau_o) / (f Dp)
#       lam: ratio crack half-length / diffusion length (the later being sqrt(4\alpha t) )
#

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root

class FluidDrivenAseismicSlipDilatancy2D:

    lambda_min = 0
    lambda_max = None

    @staticmethod
    def evalG2(xi, lambda_, epsilon):
        xi = np.abs(xi)  # enforce symmetry
        # this is dimensionless pressure solution
        a = np.sqrt(np.pi) * lambda_ * np.exp(lambda_**2)
        g = (1 + a * (erf(lambda_) - (1 + epsilon) * erf(lambda_ * xi))) / (1 + a * erf(lambda_))
        return g * (xi <= 1)

    @staticmethod
    def evalT2(lambda_, epsilon):
        # Compute parameter T for fixed lambda and epsilon
        func = lambda z: FluidDrivenAseismicSlipDilatancy2D.evalG2(np.cos(z), lambda_, epsilon)
        return 2 / np.pi * quad(func, 0, np.pi / 2)[0]

    @staticmethod
    def Tfunction2D(epsilon):
        # returns parameter T as a FUNCTION of lambda for fixed epsilon
        lambda_ = np.logspace(-4, 1, 10000)
        # Compute parameter T for each lambda
        results = np.array([FluidDrivenAseismicSlipDilatancy2D.evalT2(l, epsilon) for l in lambda_])
        # Find the index of the first negative result
        if all(x > 0 for x in results):
            negative_index = len(results)
        else:
            negative_index = np.where(results < 0)[0][0]

        # Use the positive values before the first negative value for interpolation
        lambda_positive = lambda_[:negative_index]
        results_positive = results[:negative_index]

        # Set class attributes for bounds
        FluidDrivenAseismicSlipDilatancy2D.lambda_min = lambda_positive[0] if lambda_positive.size > 0 else None  # set explicitly
        FluidDrivenAseismicSlipDilatancy2D.lambda_max = lambda_positive[-1] if lambda_positive.size > 0 else None

        # Ensure there are enough points for interpolation
        if len(lambda_positive) > 1:
            #interpolation_function = interp1d(lambda_positive, results_positive, kind='cubic')
            interpolation_function = interp1d(lambda_positive, results_positive, kind='linear')
            return interpolation_function
        else:
            # Handle cases where interpolation is not possible
            return None

    #@staticmethod
    #def evalLambda2(Tparameter, epsilon):
    #    # Compute parameter lambda for fixed T and epsilon
    #    # Get the interpolation function from Tfunction2D
    #    interpolation_function = FluidDrivenAseismicSlipDilatancy2D.Tfunction2D(epsilon)
    #    # Initial guess near the middle of the lambda range used for interpolation
    #    initial_guess = 0.5 * (FluidDrivenAseismicSlipDilatancy2D.lambda_min + FluidDrivenAseismicSlipDilatancy2D.lambda_max)
    #    # Define the equation to solve with bounds
    #    def equation_to_solve(lambda_):
    #        if lambda_ < FluidDrivenAseismicSlipDilatancy2D.lambda_min or lambda_ > FluidDrivenAseismicSlipDilatancy2D.lambda_max:
    #            return np.inf  # Repelling fsolve from out-of-bounds values
    #        return interpolation_function(lambda_) - Tparameter
    #    # Solve for lambda that makes the equation zero
    #    lambda_result = fsolve(equation_to_solve, initial_guess)[0]
    #    return lambda_result    

    @staticmethod
    def evalLambda2(Tparameter, epsilon):
        # Compute parameter lambda for fixed T and epsilon
        # Get the interpolation function from Tfunction2D
        interpolation_function = FluidDrivenAseismicSlipDilatancy2D.Tfunction2D(epsilon)
    
        from scipy.optimize import root_scalar

        def equation_to_solve(lambda_):
            return interpolation_function(lambda_) - Tparameter

        sol = root_scalar(
           equation_to_solve,
          method='brentq',
          bracket=[FluidDrivenAseismicSlipDilatancy2D.lambda_min, FluidDrivenAseismicSlipDilatancy2D.lambda_max]
        )

        lambda_result = sol.root
        return lambda_result

    @staticmethod
    def G_small_lambda(xi, lambda_, epsilon):
        """ asymptotic solution for small lambda  - See Dunham 2024 - eq3.6 
        note overpuresse is Dp g(xi)
        xi = x/ a(t)
        """
        g = 1-2*(1+epsilon)*lambda_**2*(xi) 
        return g * (xi <= 1)
    
    @staticmethod
    def T_small_lambda(lambda_, epsilon):
        """ asymptotic solution for small lambda  - See Dunham 2024 - eq3.9
          error in O(lambda_^4)        
        """
        return 1-(4/np.pi)*(1+epsilon)*lambda_**2
    
from scipy.special import expi, erf

class FluidDrivenAseismicSlipDilatancy3D:

    lambda_min = 0
    lambda_max = None

    # 3D functions
    @staticmethod
    def evalG3(xi, lambda_, epsilon):
        # this is dimensionless pressure solution
        # Evaluates the G function for the 3D problem
        term1 = -expi(-lambda_**2 * xi**2)
        term2 = -expi(-lambda_**2)
        term3 = np.exp(-lambda_**2) / lambda_**2
        g = term1 - term2 + term3 - epsilon
        return g * (xi <= 1)

    @staticmethod
    def evalT3(lambda_, epsilon):
        # Compute parameter T for fixed lambda and epsilon in 3D
        func = lambda z: FluidDrivenAseismicSlipDilatancy3D.evalG3(np.cos(z), lambda_, epsilon) * np.cos(z)
        return quad(func, 0, np.pi / 2)[0]

    @staticmethod
    def evalG3smallLambda(xi, lambda_, epsilon):
        g = 1 / lambda_**2 - 1 - np.log(xi**2) - epsilon
        return g * (xi <= 1)

    @staticmethod
    def Tfunction3D(epsilon):
        # Returns parameter T as a FUNCTION of lambda for fixed epsilon in 3D
        lambda_ = np.logspace(-4, 1, 5000)
        # Compute parameter T for each lambda
        results = np.array([FluidDrivenAseismicSlipDilatancy3D.evalT3(l, epsilon) for l in lambda_])
        if all(x > 0 for x in results):
            negative_index = len(results)
        else:
            negative_index = np.where(results < 0)[0][0]
        
        # Use the positive values before the first negative value for interpolation
        lambda_positive = lambda_[:negative_index]
        results_positive = results[:negative_index]

        # Set class attributes for bounds
        FluidDrivenAseismicSlipDilatancy3D.lambda_min = lambda_positive[0] if lambda_positive.size > 0 else None
        FluidDrivenAseismicSlipDilatancy3D.lambda_max = lambda_positive[-1] if lambda_positive.size > 0 else None

        if len(lambda_positive) > 1:
            interpolation_function = interp1d(lambda_positive, results_positive, kind='cubic')
            return interpolation_function
        else:
            return None

    @staticmethod
    def evalLambda3(Tparameter, epsilon):
        # Compute parameter lambda for fixed T and epsilon in 3D
        interpolation_function = FluidDrivenAseismicSlipDilatancy3D.Tfunction3D(epsilon)
        if interpolation_function is None:
            raise ValueError("Interpolation function could not be created; check the range of lambda or data.")

        # Initial guess near the middle of the lambda range used for interpolation
        initial_guess = 0.5 * (FluidDrivenAseismicSlipDilatancy3D.lambda_min + FluidDrivenAseismicSlipDilatancy3D.lambda_max)

        # Define the equation to solve with bounds
        def equation_to_solve(lambda_):
            if lambda_ < FluidDrivenAseismicSlipDilatancy3D.lambda_min or lambda_ > FluidDrivenAseismicSlipDilatancy3D.lambda_max:
                return np.inf  # Repelling fsolve from out-of-bounds values
            return interpolation_function(lambda_) - Tparameter

        lambda_result = fsolve(equation_to_solve, initial_guess)[0]
        return lambda_result

import matplotlib.pyplot as plt
import numpy as np

def plot_T_vs_lambda_2D():
    epsilon_values = [0., 1., 10., 100.]  # Different epsilon values for demonstration
    lambda_range = np.logspace(-2, 1, 100)  # Lambda range

    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        T_values = [FluidDrivenAseismicSlipDilatancy2D.evalT2(l, epsilon) for l in lambda_range]
        plt.plot(lambda_range, T_values, label=f'Epsilon = {epsilon}')

    plt.xlabel('Lambda')
    plt.ylabel('T')
    plt.title('T vs Lambda for Different Epsilon Values in 2D')
    plt.legend()
    plt.xscale('log')
    #plt.yscale('log')
    plt.yscale('linear')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def plot_T_vs_lambda_3D():
    epsilon_values = [0., 1., 10., 100.]  # Different epsilon values for demonstration
    lambda_range = np.logspace(-2, 1, 100)  # Lambda range

    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        T_values = [FluidDrivenAseismicSlipDilatancy3D.evalT3(l, epsilon) for l in lambda_range]
        plt.plot(lambda_range, T_values, label=f'Epsilon = {epsilon}')

    plt.xlabel('Lambda')
    plt.ylabel('T')
    plt.title('T vs Lambda for Different Epsilon Values in 3D')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    #plt.yscale('linear')
    plt.ylim(0.1, 100)
    plt.xlim(0.01, 2.5)
    plt.grid(True)
    plt.show()

# Example
plot_T_vs_lambda_2D()
#plot_T_vs_lambda_3D()

# Pressure profile
#epsilon = 50
#Tparam = 20
#lam = FluidDrivenAseismicSlipDilatancy3D.evalLambda3(Tparam, epsilon)
#r = np.linspace(0, 10., 200)
#time = 10
#alpha = 1
#xi = r / (lam * np.sqrt(4.* alpha * time))
#overpressure = FluidDrivenAseismicSlipDilatancy3D.evalG3(xi, lam, epsilon=10)
#plt.figure(figsize=(10, 6))
#plt.plot(xi, overpressure, label=f'Epsilon = {epsilon}')
#plt.xlabel('Dimensionless coordinate xi = r/a(t)')
#plt.title('Overpressure as a function of coordinate at fixed time')
#plt.ylabel('Overpressure profile')
#plt.ylim(0, 2)
#plt.xlim(0, 2)
#plt.legend()
#plt.grid(True)
#plt.show()