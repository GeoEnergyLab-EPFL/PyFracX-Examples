# %% SCRIPT FOR THE SIMULATION OF THE BASEL-1 well Stimulation
import time
import json

import numpy as np
import matplotlib.pyplot as plt

from mesh.usmesh import usmesh

from utils.json_dict_dataclass_utils import *
from hm.HMFsolver import HMFSolution


#%% loading numerical results 
# here you need to change basefolder to the folder name of your simulation
basefolder ='./Axisymmetric-Basel-NonLinearStiffness _14_10_2024___17_02_00/'
basename="Axisymmetric-Basel-NonLinearStiffness "

# we load n_steps - please adjst as per your simulation 
res = []
n_steps =400 
off_set = 5 
for i in range(n_steps):
    tt = from_dict_to_dataclass(HMFSolution,json_read(basefolder+basename+'-'+str((i+1)*off_set)))
    res.append(tt)

param = json_read(basefolder+'Parameters')

mm =  json_read(basefolder+'Mesh')
coor1D=np.array(mm["Coordinates"])[:,0]



# %% Postprocess, compare with analytical solution
# compute the radial coordinates of the mesh nodes.
x=coor1D

tt=np.array([res[j].time for j in range(len(res))])
# # find index of coordinate close to 1. meter
# ind=np.argmin(np.absolute(x-1.))

####################################################################
# plots and checks


####################################################################
# plots and checks
timestep=-2
num_pressure=res[timestep].pressure
#true_sol=pressure(x,tt[timestep],Qinj)

import matplotlib.pyplot as plt

####################################################################
# %% pressure well
pw_t=np.array([res[j].pressure[0] for j in range(len(res))])

file = open('./SurfacePressure_Mpa.json') #/local_dev/Basel
pressureData = np.array(json.load(file))
file.close()

#true_sol=pressure(x[1],tt,Qinj)
fig, ax = plt.subplots()
ax.plot(tt[:], pw_t[:],'.b')
#x.plot(tt, true_sol,'.r')
plt.show()
fig, ax = plt.subplots()
ax.plot(pressureData[:,0], pressureData[:,1],'.r')
ax.plot(pressureData[:,0], pressureData[:,1],'.r')

ax.plot(tt[:], 1e-6*pw_t[:],'.b')
plt.xlabel("Time (s)")
plt.ylabel("Well Pressure MPa")
#plt.xlim([0,30000])
plt.show()
#%% ##### rupture radius

file = open('./SeismicRadius.json')
seismicRadius = np.array(json.load(file))
file.close()

nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi*(coor1D[1]-coor1D[0])
fig, ax = plt.subplots()
ax.plot( tt ,cr_front ,'.y')
ax.plot(seismicRadius[:90,0], seismicRadius[:90,1],'.r')

plt.xlabel("Time (s)")
plt.ylabel("rupture radius(m)")
plt.xlim([0.,800000.])
# ax.legend(["analytical solution","numerics"])
plt.show()

#%%
#### slip at origin

slip_0 = np.abs(np.array([res[i].DDs[0] for i in range(len(res))]))
slip_p_0 = np.abs(np.array([res[i].DDs_plastic[0] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,slip_0 ,'.r')
ax.plot( tt ,slip_p_0 ,'.r')
plt.xlabel("Time (s)")
plt.ylabel("slip @ injection point (m)")
plt.show()

# %% width at origin
width_0 = np.abs(np.array([res[i].DDs[1] for i in range(len(res))]))
width_p_0 = np.abs(np.array([res[i].DDs_plastic[1] for i in range(len(res))]))

fig, ax = plt.subplots()
ax.plot( tt ,width_0 ,'.k')
ax.plot( tt ,width_p_0 ,'.b')
plt.xlabel("Time (s)")
plt.ylabel("width @ injection point (m)")
# ax.legend(["analytical solution","numerics"])
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot( slip_0 ,width_0 ,'.k')
ax.plot( slip_p_0 ,width_p_0 ,'.b')
plt.xlabel("slip @ injection point (m)")
plt.ylabel("width @ injection point (m)")
# ax.legend(["analytical solution","numerics"])
plt.show()

#%%
### profiles
timestep=-50
sol_to_p= res[timestep]
x=coor1D
num_pressure=sol_to_p.pressure
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(x[1:400], num_pressure[1:400],'.b')
plt.xlabel("  r (m)")
plt.ylabel("fluid pressure (Pa)")
plt.show()

#%%

wh_o=1.
# width profile
w_profile=np.array(sol_to_p.DDs[1::2])
w_profile_p=np.array(sol_to_p.DDs_plastic[1::2])
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., w_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., w_profile_p[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel(" width (m)")
plt.show()

fig, ax = plt.subplots()
Tav = 2.e-15  
ax.plot((x[1:]+x[0::-2])/2., 1+(w_profile[:]**3)/(12*Tav),'.k')
plt.xlabel("  r (m)")
plt.ylabel(" hydr. transmissibility increase (-)")
plt.show()

#%%
# slip profile
slip_profile=sol_to_p.DDs[0::2]
slip_profile_p=sol_to_p.DDs_plastic[0::2]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., -slip_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., -slip_profile_p[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel("slip (m)")
plt.show()

# dilatancy profile
fig, ax = plt.subplots()
app_dila_c = -sol_to_p.DDs_rate[1::2]/sol_to_p.DDs_rate[0::2]
ax.plot((x[1:]+x[0::-2])/2.,app_dila_c,'.b')
plt.xlabel("  r (m)")
plt.ylabel(" w/slip (-)")
plt.show()

#%%
# effective tracitons profile
tau =sol_to_p.effective_tractions[0::2]
sig_n =sol_to_p.effective_tractions[1::2]

fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., -tau[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., -sig_n[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel("effective tractions (Pa)")
plt.show()

from src.mechanics.evolutionLaws import *

fric_c= [linearEvolution(slip_profile_p[i],d_c,f_p,f_r) for i in range(Nelts)]
dila_c = [linearEvolution(slip_profile_p[i],d_c,psi_p,0.) for i in range(Nelts)]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., fric_c,'.b')
ax.plot((x[1:]+x[0::-2])/2., dila_c,'.b')

plt.xlabel("  r (m)")
plt.ylabel("friction coef. & dilatancy coef.")
plt.show()

###
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2.,np.abs(tau)+fric_c*sig_n,'.b')
#ax.plot((x[1:]+x[0::-2])/2., dila_c,'.b')

plt.xlabel("  r (m)")
plt.ylabel(" F_mc")
plt.show()
 
# %%

# POST-Process stats

newton_its = np.array([ res[i].stats['iteration'] for i in range(len(res))])
elap_time = np.array([ res[i].stats['elapsed_time'] for i in range(len(res))])
step_time = np.diff(elap_time)
step_time=np.insert(step_time,0,step_time[0])
n_yield_step = np.array([ res[i].Nyielded for i in range(len(res))])

#%%
nmvt_step = 0.*newton_its 

for i in range(len(res)):
    for k in range(newton_its[i]-1):
        nmvt_step[i]+=res[i].stats['jacobian_stat_list'][k]['Total number of A11 matvect: ']

# %%
fig, ax = plt.subplots()
ax.plot(n_yield_step,nmvt_step,'.')
# %%
# %%
fig, ax = plt.subplots()
ax.plot(newton_its,nmvt_step,'.')

# %%
fig, ax = plt.subplots()
ax.plot(nmvt_step,step_time,'.')
plt.ylim([0, 200])
# %%
# %%
fig, ax = plt.subplots()
ax.plot(newton_its,step_time,'.')
plt.ylim([0, 250])

# %%
ig, ax = plt.subplots()
ax.plot(nmvt_step/newton_its,step_time/newton_its,'.')
plt.ylim([0, 50])
# %%
