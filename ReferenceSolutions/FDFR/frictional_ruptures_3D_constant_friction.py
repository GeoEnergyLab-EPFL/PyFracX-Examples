import mpmath
import numpy as np
import scipy.special
# using mpmath library to solve Eq. 21 (scipy hyp not supported for 2f2 function)
# https://mpmath.org/doc/current/functions/hypergeometric.html
# Also install gmpy2 for better mpmath performance https://pypi.org/project/gmpy2/
# pip install gmpy2

# Function to find the analytical solution of the amplification factor lambda 
# From Saez & Lecampion 2023 

def lambda_analytical(T):
    
    if T > 2:
        lam = 0.5 * np.exp((2.0 - float(mpmath.euler) - T) * 0.5)
        # Marginally pressurized case, Eq. 23 from Saez & Lecampion 2023
        
    elif T < 0.3:
        lam = 1.0 / np.sqrt(2.0 * T)
        #Critically stressed case, Eq. 22 from Saez & Lecampion 2023
    else:
        lam = ((0.5 * np.exp((2.0 - float(mpmath.euler) - T) * 0.5)) 
               + (1.0 / np.sqrt(2.0 * T))) * 0.5
    
    # Eq 21
    eq = (
        lambda lam2: 2
        - mpmath.euler
        + (2 / 3) * lam2 * mpmath.hyp2f2(1, 1, 2, 2.5, -lam2)
        - mpmath.log(4 * lam2)
        - T
    )
    return float(mpmath.sqrt(mpmath.findroot(eq, lam ** 2).real))

#Critically stressed case, Eq. 22 from Saez & Lecampion 2023
def lambda_approx_cs(T):
    return 1.0 / np.sqrt(2.0 * T)

# Marginally pressurized case, Eq. 23 from Saez & Lecampion 2023
def lambda_approx_mp(T):
    return 0.5 * np.exp((2.0 - float(mpmath.euler) - T) * 0.5)



def complete_slip_profile_cs(G, f, dp_star, alpha, T, t, lambda_analytical, distances):
    #complete analytical solution for fluid driven circular frictional ruptures in critically stressed regime
    #Given by Viesca 2024
    # G - shear modulus
    # f - friction coefficient
    # dp_star - dimensionless injection rate - Q/(4*pi*hydraulic_cond*wh)
    # alpha - hyd diffusivity
    # T - stress intensity parameter
    # t - time
    # lambda_analytical - analytical solution for lambda - function of stress injection parameter T
    # distances - distances from the injection point
    
    alpha_new = 4*alpha
    lam = lambda_analytical(T)
    rho = distances / (lam * np.sqrt(alpha_new*t))
    arg = lam * rho
    
    first_term =  2 * np.sqrt(np.pi) * np.exp(-0.5 * arg**2) *\
                  ( (1 + arg**2) * scipy.special.i0(0.5 * arg**2) + arg**2 * scipy.special.i1(0.5 * arg**2)) - 4 * arg
    
    sec_term = (2 / (lam * np.pi)) * (np.arccos(rho)/rho - np.sqrt(1 - rho**2)) - 1/arg
    
    third_term = (0.25 / (lam**3 * np.pi)) * (np.arccos(rho)/rho**3 - np.sqrt(1 - rho**2) * (2 - 1/rho**2)) - 1/((2*arg)**3) 
    fourth_term = (3 / (16 * lam**5 * np.pi)) * (np.arccos(rho)/rho**5 - np.sqrt(1 - rho**2) *  
                                                 (8/3 - 2/(3 * rho**2) - 1/rho**4)) - 3/((2*arg)**5)
    
    slip_profile = dp_star * f * np.sqrt(alpha_new*t) * (1/G) * (first_term + sec_term + third_term + fourth_term)
    
    return slip_profile

def complete_slip_profile_mp(G, f, dp_star, alpha, T, t, lambda_analytical, distances):
    #complete analytical solution for fluid driven circular frictional ruptures in marginally pressurised regime
    #Given by Viesca 2024
    # G - shear modulus
    # f - friction coefficient
    # dp_star - dimensionless injection rate - Q/(4*pi*hydraulic_cond*wh)
    # alpha - hyd diffusivity
    # T - stress intensity parameter
    # t - time
    # lambda_analytical - analytical solution for lambda - function of stress injection parameter T
    # distances - distances from the injection point
    
    alpha_new = 4*alpha
    lam = lambda_analytical(T)
    rho = distances / (lam * np.sqrt(alpha_new*t))
    arg = lam * rho
    
    first_term = (8 / np.pi) * (np.sqrt(1 - rho**2) - rho * np.arccos(rho))
    
    sec_term = ((lam**2 * 16) / (9 * np.pi)) * (1 - rho**2)**(3/2)
    
    third_term = lam**4 * (32 / (225 * np.pi)) * (1 - rho**2)**(3/2) * (3 + 2 * rho**2)
    
    slip_profile = dp_star * f * lam * np.sqrt(alpha_new*t) * (1/G) * (first_term + sec_term + third_term)
    
    return slip_profile
