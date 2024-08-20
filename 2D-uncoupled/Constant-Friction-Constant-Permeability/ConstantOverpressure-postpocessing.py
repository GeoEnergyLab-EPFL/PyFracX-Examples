#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a 2D planar frictional fault (plane-strain problem) due to a constant over-pressure
# Constant friction reference results from Viesca (2021)
# All properties are constant
#
#%% imports
import numpy as np
import matplotlib.pyplot as plt

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
import os
directory_path = os.path.dirname(os.path.abspath(__file__))
basefolder = directory_path + '/2D-ctP-ctFriction-20-08-2024-09-03-17/'
basename="2D-ctP-ctFriction"
# we load n_steps - please adjst as per your simulation 
res = []
n_steps =40 
for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str(i+1)))
    res.append(tt)

param = json_read(basefolder+'Parameters')

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
T = param["T parameter"]
f_p = param["Friction"]
sigmap_o = param["Initial stress"][0]
tau_o = param["Initial stress"][1]
YoungM = param["Elasticity"]["Young"]
nu = param["Elasticity"]["Nu"]

shear_prime=YoungM/(2*(1+nu)*2*(1-nu))
dpcenter=(1.-tau_o/(f_p*sigmap_o))*sigmap_o/T

mm =  json_read(basefolder+'Mesh')
Nelts = mm["Nelts"] 
coor1D = np.array(mm["Coordinates"])[:,0]

#%% plots

from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *

# Select time-step to plot profiles later
jj=len(res)-2
t = res[jj].time

# numerical results
tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi/2.*(coor1D[1]-coor1D[0]) # only for uniform mesh

# analytical solution
if T<0.4:
    lam = 2 / (np.pi ** (3 / 2)) * 1 / T
    print("Critically stressed case lambda=",lam)
else:
    lam=(np.pi**(3/2))/4.*(1-T)
    print("Marginally pressurized case lambda=",lam)

y_ = lam * np.sqrt(4. * alpha_hyd * tts) # crack front from analytical solution

#%%  Plot: crack half length VS time
fig, ax = plt.subplots()
ax.loglog(tts,y_,'r')
ax.loglog(tts,cr_front,'.')
plt.xlabel("Time (s)")
plt.ylabel("Crack half-length (m)")
ax.legend(["analytical","numerics"])
plt.show()

#%% Plot: Slip profile (Dimensionless)
if T > 0.4:
    xx = np.linspace(-1, 1, Nelts)
    #analytical_slip = marginallyStressed_slip(xx)
    analytical_slip = marginallyPressurized_slip(xx)
    slip_scale_MP =lam**2*np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    numerical_slip = -np.array(res[jj].DDs)[0::2] / slip_scale_MP
    fig, ax = plt.subplots()
    ax.plot(xx, analytical_slip,'r.')
    ax.plot((coor1D[1:] + coor1D[0:-1]) / (2. * cr_front[jj]), numerical_slip,'b-')
    plt.xlabel("x (-)")
    plt.ylabel("Non-dimensional Slip profile (-))")
    ax.legend(['Analytical','Numerical'])
    ax.set_xlim([-1., 1.])  # range for x-axis
    plt.show()
else:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    xx_new = xx[(xx < cr_front[jj]) & (xx > - cr_front[jj])]

    slip_scale_CS =np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    numerical_slip = -np.array(res[jj].DDs)[0::2] / slip_scale_CS

    fig, ax = plt.subplots()
    ax.plot( xx / cr_front[jj], numerical_slip, 'b.')

    x_ = xx_new[(xx_new > cr_front[jj] / 2.) | (xx_new < -cr_front[jj] / 2)] / cr_front[jj]
    analytical_slip_Out = criticallyStressed_slip_Outer(x_)
    ax.plot(x_, analytical_slip_Out, 'r--')

    x__ = xx[(xx < cr_front[jj] / 2.) & (xx > -cr_front[jj] / 2)]
    xhat__ = x__ / np.sqrt(4 * alpha_hyd * t)
    analytical_slip_In = criticallyStressed_slip_Inner(xhat__ , lam)
    ax.plot(x__ / cr_front[jj], analytical_slip_In, 'g--')

    ax.legend(['Numerical','Analytical outer','Analytical inner'])
    ax.set_xlim([-1., 1.])  # range for x-axis
    plt.xlabel("x (-)")
    plt.ylabel("Non-dimensional Slip profile (-))")
    plt.show()

#%% Plot: Slip profile (Dimensional)

if T > 0.4:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    analytical_slip = marginallyStressed_slip_dimensional( xx,T, alpha_hyd,t,f_p,dpcenter,shear_prime)
    numerical_slip = -np.array(res[jj].DDs)[0::2]
    fig, ax = plt.subplots()
    ax.plot(xx, analytical_slip,'r-')
    ax.plot( xx , numerical_slip,'b.')
    plt.xlabel("x (m)")
    plt.ylabel("Dimensional Slip profile (m))")
    ax.legend(['Analytical','Numerical'])
    ax.set_xlim([-cr_front[jj], cr_front[jj]])  # range for x-axis
    plt.show()
else:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    xx_new = xx[(xx < cr_front[jj]) & (xx > -cr_front[jj])]

    slip_scale_CS =np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    numerical_slip = -np.array(res[jj].DDs)[0::2]

    fig, ax = plt.subplots()
    ax.plot( xx , numerical_slip)

    x_ = xx_new[(xx_new > cr_front[jj] / 2.) | (xx_new < -cr_front[jj] / 2)] 
    analytical_slip_Out = slip_scale_CS * criticallyStressed_slip_Outer(x_ / cr_front[jj])
    ax.plot(x_, analytical_slip_Out, 'r--')

    x__ = xx[(xx < y_[jj] / 2.) & (xx > -y_[jj] / 2)]
    analytical_slip_In = slip_scale_CS * criticallyStressed_slip_Inner(x__ / np.sqrt(4. * alpha_hyd * t), lam)
    ax.plot(x__, analytical_slip_In, 'g--')

    ax.legend(['Numerical','Analytical outer','Analytical inner'])
    ax.set_xlim([-cr_front[jj], cr_front[jj]])  # range for x-axis
    plt.xlabel("x (m)")
    plt.ylabel("Dimensional Slip profile (m))")
    plt.show()


#%% time - stepping

dts = np.array([res[i].timestep for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog( tts ,dts ,'.')
plt.xlabel("Time (s)")
plt.ylabel("time-step")
# ax.legend(["analytical solution","numerics"])
plt.show()
#  PLOTTING slip profile ---- 

ltes = np.array([res[i].lte for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog( tts ,ltes ,'.')
plt.xlabel("Time (s)")
plt.ylabel("LTE estimate")
# ax.legend(["analytical solution","numerics"])
plt.show()


#%%
jj=-1
fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-np.array(res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("Plastic slip profile (m) at time last step")
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(np.array(res[jj].DDs)[0::2]-np.array(res[jj].DDs_plastic)[0::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic slip profile (m) at time last step")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2., res[jj].DDs_plastic[1::2])
plt.xlabel("x (m)")
plt.ylabel("Plastic opening profile (m) at time last step")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(np.array(res[jj].DDs)[1::2]-np.array(res[jj].DDs_plastic)[1::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic Opg (m) at time last step")
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[1::2],'.-')
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa) at time last step")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[0::2],'.-')
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa) at time last step")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].yieldF)
plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa) at time last step")
plt.show()


# %%
