#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a planar frictional fault (plane-strain problem) due to a constant over-pressure
# Linear weakening friction reference results from Germanovich & Garagash (2012) 
# parameters taken from Ciardo et al IJNME 2021
#
#%% imports
import numpy as np
import matplotlib.pyplot as plt

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution

#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder ='./res_data/2D-ctP-LinearWeakening-MarginallyPressurized-10-02-2025-14-46-39/'
basename="2D-ctP-LinearWeakening-MarginallyPressurized"

# we load n_steps - please adjst as per your simulation 
res = []
n_steps =125 
for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str(i+1)))
    res.append(tt)

param = json_read(basefolder+'Parameters')

alpha_hyd = param["Flow"]["Hydraulic diffusivity"]
T = param["T parameter"]

mm =  json_read(basefolder+'Mesh')
coor1D=np.array(mm["Coordinates"])[:,0]

title_string =' $T_p$ ='+str(np.round(T,2)) + ", $\Delta P / \sigma'_o =0.5$, $f_r/f_p=0.6$, $\\tau_o / (f_p \sigma'_o)=0.55$"

#%% load reference results from G&G 2012 for that parameter sets
solu=json_read("../../ReferenceSolutions/FDFR/LW-2D-GG12-MP-reference-1")

ref_length=np.array(solu["Scaled Crack Length"])
ref_peak_slip=np.array(solu["Scaled slip at center"])

#%% plots

from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *

lam=marginallyPressurized_lambda(T) #for the early peak friction solution

tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi/2.*(coor1D[1]-coor1D[0])

#  Crack half length

t_ = np.linspace(0.001, tts.max(), 1000)
y_ = lam*np.sqrt(4.*alpha_hyd*t_)
yr_= 1*np.sqrt(4.*alpha_hyd*t_)
aw = 0.5
fig, ax = plt.subplots()
ax.plot(np.sqrt(4*alpha_hyd*t_)/aw,y_/.5,'r')
ax.plot(np.sqrt(4*alpha_hyd*t_)/aw,np.sqrt(4.*alpha_hyd*t_)/aw,'b--')
ax.plot(ref_length[:,0],ref_length[:,1],'-k')
ax.plot(np.sqrt(4*alpha_hyd*tts)/aw,cr_front/aw,'.')
plt.title(title_string)
plt.xlabel("$\sqrt{4 \\alpha t}   / a_w$")
plt.ylabel("Crack half-length $a / a_w$")
ax.legend(["Peak friction analytical solution",
           "Scaled Diffusion front $\sqrt{4\\alpha t}/a_w$","G&G 2012 reference solution","Numerical results"])
plt.show()


#%% peak slip at center

# we extract the shear slip at the central point (because there is 2 dofs per elt)
Nelts=mm["Nelts"] 
slip_0 = np.abs(np.array([res[i].DDs_plastic[Nelts] for i in range(len(res))]))
fig, ax = plt.subplots()

ax.plot(ref_peak_slip[:,0],ref_peak_slip[:,1],'-k')
ax.plot( np.sqrt(4*alpha_hyd*tts)/aw,slip_0 ,'.')
plt.xlabel(" $\sqrt{4\\alpha t}/a_w$")
plt.ylabel("Scaled slip at center $\delta(0)/\delta_w$")
plt.title(title_string)
ax.legend(["G&G 2012 reference solution","Numerical results"])
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
