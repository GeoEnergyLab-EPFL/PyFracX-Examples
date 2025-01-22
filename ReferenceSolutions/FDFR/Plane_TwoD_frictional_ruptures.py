# Python module containing solutions for the growth of 2D plane-strain 
# fluid-driven shear ruptures in a planar fault with uniform and constant
# properties. 
# The friction coefficient is  constant
# The hydraulic properties (permeability and storage) are constant
# There is no shear-induced dilatancy.
# References:
#   R. C. Viesca. Self-similar fault slip in response to fluid injection. Journal of Fluid Mechanics, 928, 2021.
#   P. Bhattacharya and R. C. Viesca. Fluid-induced aseismic fault slip outpaces pore-fluid migration. Science, 364(6439):464–468, 2019.

# notations:
#       T: fault stress parameter defined as (f\sigma'_o - \tau_o) / (f Dp)
#       lam: ratio crack half-length / diffusion length (the later being sqrt(4\alpha t) )
#

# TODO : adds documentation of functions, check and complement some.
import numpy as np

#--------------------- Critically Stressed case --------------------------------
# Solution for Critically stressed conditions - T<< 1 , lam >> 1
def criticallyStressed_lambda(T):
    """Solution for the amplification factor \lambda = a(t) / L_diffusion(t)
    in the critically stress regime (T<<1, lambda >>1) up to O(1/lambda^3). 
    L_diffusion(t)=sqrt(4 alpha t) is the location of the pore-pressure diffusion front.
    a(t) is the half-length of the shear rupture.
    Args:
        T (double): The fault-injection stress parameter (f\sigma'_o - \tau_o) / (f Dp)
    where f is the friction coef., \sigma'_o and \tau_o the initial
    effective normal and shear stresses acting on the fault plane.
    Dp is the injection over-pressure at the injection point.
    Returns:
       double :  lambda (the amplification factor)
    """
    # the \lambda - T relation
    return 2./(np.pi**(3/2)) * 1/T  # O(1/\lambda^3)

# outer solution (coordinates scaled by half-crack length)
def criticallyStressed_slip_Outer(xb):
    """Outer solution for the shear slip as function of xb=x/a(t) in the
    critically-stress regime (T<<1, lambda>>1) 

    Args:
        xb (double or array of double): x/ a(t) = x/ (lam * sqrt(4*alpha*t)

    Returns:
        (double or array of double): dimensionless outer solution of shear slip (valid for xb>0.3)
    """
    "xb = x/ a(t) = x/ (lam * sqrt(4*alpha*t) "
    slip = (2./np.pi**(3/2))*(np.arctanh(np.sqrt(1.-xb**2))-np.sqrt(1.-xb**2))
    return slip

# inner solution ....
# The function f(x) = \int_0^x (1/pi) \int_{-\infty}^{+\infty} Erfc[|s|]/(x-s) ds dx
# tabulated f values from Viesca 2021 Supplemental
abs_xhat=np.array([0.,0.024548622,0.049126850,
                   0.073764432,0.098491403, 0.12333824,0.14833599,0.17351646,0.19891237,0.22455751,0.25048696,0.27673727,0.30334668,
                   0.33035538,0.35780572,0.38574257,0.41421356,0.44326951,0.47296478,0.50335770,0.53451114,0.56649300,0.59937693,0.63324302,
                   0.66817864,0.70427946,0.74165054,0.78040766,0.82067879,0.86260593,0.90634717,0.95207915,1.,1.0503328,
                   1.1033300,1.1592779,1.2185035,1.2813816,1.3483439,1.4198909,1.4966058,1.5791726,1.6683992,1.7652469,
                   1.8708684,1.9866588,2.1143224,2.2559639,2.4142136,2.5924025,2.7948128,3.0270432,3.2965582,3.6135357,3.9922238,
                   4.4532022,5.0273395,5.7631420,6.7414524,8.1077858,10.153170,13.556669,20.355468,40.735483])
f_xhat=np.array([0.,0.0010645142,0.0036608823,0.0074561611,0.012278829,0.018013946,0.024577096,0.031903213,0.039940676,
    0.048647765,0.057990359,0.067940342,0.078474446,0.089573374,0.10122112,0.11340443,0.12611233,0.13933575,0.15306718,0.16730037,
    0.18203004,0.19725163,0.21296104,0.22915436,0.24582765,0.26297665,0.28059656,0.29868178,0.31722565,0.33622021,0.35565599,0.37552179,0.39580461,
    0.41648953,0.43755981,0.45899714,0.48078204,0.50289466,0.52531586,0.54802878,0.57102094,0.59428697,0.61783187,0.64167499,0.66585435,0.69043124,
    0.71549463,0.74116509,0.76759781,0.79498501,0.82355843,0.85359380,0.88542014,0.91943717,0.95614458,0.99618834,1.0404337,1.0900859,1.1469030,
    1.2136044,1.2947397,1.3988316,1.5450076,1.7942933])
from scipy.interpolate import interp1d
f_func_tab=interp1d(abs_xhat,f_xhat, kind='cubic')

def f_func_small(xhat):
    " xhat=x/sqrt(4\alpha t)"
    #
    # valid for xhat < 0.2 (less than 0.5 percent relative difference)
    f_small=(2./np.pi**(3/2))*(xhat**2)*(np.log(1./np.abs(xhat))-np.euler_gamma/2.+3./2.) # O(x^4 \log (x))
    return f_small

def f_func_large(xhat):
    " xhat=x/sqrt(4\alpha t)"
    #
    # valid for xhat > 10. (less than 0.1 percent relative difference)
    f_large=(2./np.pi**(3/2))*(np.log(np.abs(xhat))+np.euler_gamma/2.+1)  # # O(1/x^2)
    return f_large

def f_func_raw(xhat):
    " xhat=x/sqrt(4\alpha t)"
    # the function f(x) = \int_0^x (1/pi) \int_{-\infty}^{+\infty} Erfc[|s|]/(x-s) ds dx
    if ( (np.abs(xhat)>0.2) & (np.abs(xhat)<10.)):
        return f_func_tab(np.abs(xhat))
    else :
        if np.abs(xhat)<=0.2:
            if xhat==0.:
                return 0.
            else:
                return f_func_small(xhat)
        else:
            return f_func_large(xhat)

from numpy import vectorize
f_func = vectorize(f_func_raw)


def criticallyStressed_slip0(lam):
    # slip at center in critically stressed case
    # scaled by a(t) f Dp/mu'
    slip0 =(1/lam)* (2. / (np.pi ** (3 / 2))) * (np.log(2. * lam) + np.euler_gamma / 2. - (1. / (4. * (lam ** 2))))
    return slip0

def criticallyStressed_slip_Inner(xhat, lam):
    " Inner solution - valid for lam >~ 2-4 "
    # slip is scaled as :sqrt(4\alpha t) f Dp/mu'
    # xhat=x/sqrt(4\alpha t)
    # accumulated slip at the center.
    slip0 = (2./(np.pi**(3/2)))*(np.log(2.*lam) + np.euler_gamma/2.-(1./(4.*(lam**2)))) # O(1/lam^4)
#
    slip = slip0-f_func(xhat)
    return slip

# this does not work.....
def criticallyStressed_slip_Matched(xb,lam):
    "xb = x/ a(t) = x/ (lam * sqrt(4*alpha*t) "
    # valid for lam > 5  (less than 1 percent error for lam =5)
    #
    slip_in =criticallyStressed_slip_Inner(xb*lam,lam)
    slip_out=criticallyStressed_slip_Outer(xb)
    slip_overlap=(2./(np.pi)**(3./2.))*(np.log(2./np.abs(xb))-1.+(xb**2)/4.)+(1/(lam**2))*(1./(3.*(np.pi**(3./2.))))*(1.0/(xb**2)-3./2.)
    return slip_in+slip_out-slip_overlap

#
def center_slip_app(lam):
    " slip at center as per eq. (3.16) of Viesca (2021)"
    # slip is scaled as :sqrt(4\alpha t) f Dp/mu'
    # within 5% for any value of lam
    return (2./(np.pi)**(3./2.))*lam*lam/(1+(lam**2)/(np.log(6.+2*lam)+np.euler_gamma/2.))

#
def criticallyStressed_slip_Inner_dim(x,T,alpha,time,f,dp,muprime):
#
    lam=criticallyStressed_lambda(T)
    slip_factor=np.sqrt(4*alpha*time)*f*dp/muprime
    xhat=x/(np.sqrt(4*alpha*time))
    return slip_factor*criticallyStressed_slip_Inner(xhat,lam)

def criticallyStressed_slip_Outer_dim(x,T,alpha,time,f,dp,muprime):
    lam=criticallyStressed_lambda(T)
    slip_factor=np.sqrt(4*alpha*time)*f*dp/muprime
    xb=x/(lam*np.sqrt(4*alpha*time))
    return slip_factor*criticallyStressed_slip_Outer(xb)

#------------------------- Marginally pressurized solution
# Marginally pressurized , T>>1, lam <<1
def marginallyPressurized_lambda(T):
    """Solution for the amplification factor \lambda = a(t) / L_diffusion(t)
    in the marginally pressurized regime (T~1, lambda <<1). 
    L_diffusion(t)=sqrt(4 alpha t) is the location of the pore-pressure diffusion front.
    a(t) is the half-length of the shear rupture.
    Args:
        T (double): The fault-injection stress parameter (f\sigma'_o - \tau_o) / (f Dp)
    where f is the friction coef., \sigma'_o and \tau_o the initial
    effective normal and shear stresses acting on the fault plane.
    Dp is the injection over-pressure at the injection point.
    Returns:
       double :  lambda (the amplification factor)
    """
    return (1-T)*(np.pi**(3./2.))/4


def marginallyPressurized_slip(xb,lam):
    """Outer solution for the shear slip as function of xb=x/a(t) in the
    marginally pressurized regime (T~1, lambda<<1) 

    Args:
        xb (double or array of double):   xb = x/ a(t) = x/ (lam * sqrt(4*alpha*t) 

    Returns:
        (double or array of double): dimensionless outer solution of shear slip 
    """
    slip = (2.*(lam**2)/(np.pi**(3/2)))*(np.sqrt(1.-xb**2)-(xb**2)*np.arctanh(np.sqrt(1-xb**2)))   # + O(lam^4)
    return slip

# dimensional
def marginallyStressed_slip_dimensional(x,T, alpha,time,f,dp,muprime):
    """Outer solution for the shear slip as function of xb=x/a(t) in the
    marginally pressurized regime (T~1, lambda<<1) 


    Args:
        x (double or array of double):  dimensional position at which to compute the slip
        T (double): fault-injection stress parameter
        alpha : fault hydraulic diffusivity
        time : time at which to compute the slip
        f : friction coef
        dp: overpressure
        muprime: the plane-strain young's modulus(mode II) or shear modulus (mode III)

    Returns:
        (double or array of double):  dimensinoal  solution of shear slip 
    """
    lam=marginallyPressurized_lambda(T)
    xb=x/(lam*np.sqrt(4*alpha*time))
    slip_factor = np.sqrt(4 * alpha * time) * f * dp / muprime
    return marginallyPressurized_slip(xb,lam)*slip_factor
