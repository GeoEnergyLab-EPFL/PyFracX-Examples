# %% SCRIPT FOR THE SIMULATION OF THE BASEL-1 well Stimulation
import time
import json

import os,sys 
home = os.environ["HOME"]
dir_path = os.path.dirname(os.path.realpath(__file__))
#
#sys.path.append("/home/fakhretd/BigWham/build/interfaces/python/")
#sys.path.append("/home/fakhretd/PyFracX/src/")
#sys.path.append(dir_path)

# %%  Import required modules
import numpy as np
#import qtconsole.styles

from mesh.usmesh import usmesh
from utils.App import TimeIntegrationApp
from hm.HMFsolver import HMFSolution,hmf_coupled_step

from mechanics.H_Elasticity import Elasticity
from mechanics.mech_utils import MechanicalModel,QuasiDynamicsOperator
from mechanics.MixModeW2D import MixModeW2D,normalStiffness_BB
from mechanics.evolutionLaws import linearEvolution

from flow.flow_utils import *
from flow.FlowConstitutiveLaws import ShearZoneFlowCubicLawNewtonian
from loads.Injection import *

from utils.helper_utils import uniform_interface_properties_BB

import matplotlib.pyplot as plt

# %% Simple 1D mesh
r_w=0.
Rend = 600.0
h_x = 2.0 #0.25
Nelts= int(Rend / h_x)
coor1D=np.linspace(r_w,Rend,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me=usmesh(2,coor,conn,1)

# %%  Model and parameters.

kernel = "Axi3DS0-H"       # axisymmetric simulation 
# Elastic properties of the bulk matrix 
YoungM= 40.e9  #65e9   # [Pa] Young's modulus
nu= 0.        # [-] Poisson's ratio
shear_g= YoungM/(2*(1+nu))
Kk = 2/3 * shear_g + 2*shear_g*nu/(1-2*nu)
c_s = np.sqrt(shear_g/2700.)        # shear wave speed
c_p = c_s * np.sqrt(2*(1-nu)/(1.-2*nu)) # P -wave speed
eta_s = 0.5 *shear_g / c_s   # Quasi-dynamic s coef
eta_p = 0.5 *shear_g / c_p   # Quasi-dynamic p coef

# in-situ effective stress  #
# # uniform initial effective tractions in Pa
tauo=22.e6         # [Pa]
sigmap_o=37.5e6    # [Pa] 
in_situ_effective_tractions =np.full((Nelts,2),[-tauo,-sigmap_o])
# initial displacement discontinuities
DDs_initial=0.*in_situ_effective_tractions

## Properties of the pre-existing fracture/fault #
# Assume that matrix is impermeable and all the fluid went into one fracture.

# hydraulic properties from Pre-Stimulation test - cubic law model #
Tav = 2.e-15       # average transmibillity w^3/12    [m^3] 
#S_e_oedo = 2.e-8   # VALUE THAT WORKED REALLY WELL! Compressibility of fracture together with the shear zone    [Pa^{-1}] 
S_e_oedo = 8.e-9     # Compressibility of fracture together with the shear zone    [Pa^{-1}] 
# alpha_hydroSZ = 0.0023  # hydraulic diffusivity of the shear zone from pre-stim test
h_thickness = 1. # [m] shear zone thickness, just a guess!

fluid_visc = 2.5e-4         # [Pa*s] fluid dynamic viscosity (value from Andres)
# S_e_oedo = Tav / (fluid_visc * alpha_hydroSZ * h_thickness)
alpha_hydroSZ = Tav / (fluid_visc * S_e_oedo * h_thickness)
print('S_e_oedometric = : %g ' %(S_e_oedo))
print('Shear zone hydraulic diffusivity is = : %g ' %(alpha_hydroSZ))
perm = alpha_hydroSZ * S_e_oedo * fluid_visc
wh_o = 1.   # [m] hydraulic width of the fault, here it is a shear model, so we do not need wh_o, but I keep it for postprocessing without changes

cond_hyd = Tav / fluid_visc # [m^3/(Pa*s)]

C_wd =6.5*1.e-8 # [m^3/Pa] wellbore storage from Andres

#  interface mechanical properties  #
# Barton-Bandis stiffness, linear slip weakening interface
### stiffnesses - 
ks = YoungM*100 

#%%
#v_m=0.00048544393247491896
#kni=1.28553e10
kni=3.02693e10
v_m=0.000439377
kn_at_o=normalStiffness_BB(-(sigmap_o), kni, v_m)# at sigmap_o
kn_at_opg=normalStiffness_BB(-(0e6), kni, v_m)
wc = v_m / (1 + v_m * kni / (sigmap_o)) #mechanical aperture at which Tn'=0
print('kn(s_o)/YoungM : %g' %(kn_at_o/YoungM))
print('kn(T =0)/YoungM : %g' %(kn_at_opg/YoungM))
print('w(at T=0)/wh_o: %g ' %(wc/wh_o) )

#  friction properties 
chi = 3.2 # approximate slope of the seismic radius
f_p=0.66      # peak friction 
f_r = 2*np.pi * chi**2 * S_e_oedo * h_thickness * tauo /(2*np.pi * chi**2 * S_e_oedo * h_thickness * sigmap_o - 1)     # residual friction
f_r=0.64
print('Residual friction is : %g ' %(f_r))

psi_p = 0.1  # peak dilatancy
d_c = 0.0025   # critical slip distances
c_p=0.          # zero cohesion

sig_c = 0.
alpha =0.
beta = 0.

# Important dimensionless numbers 
S_number = tauo/(f_p*sigmap_o)
F_number = f_r/f_p
P_number = 10e6/sigmap_o

dil_max= psi_p*d_c/2.  # max dilation 

print('Stress criticality ratio : %g ' %(S_number))
print('ratio between residual and peak friction: %g' %(F_number))
print('Max transmissibility increase : %g ' %((1+(dil_max/wh_o))**3))

# %% Properties map creation  & interface model 
mat_id = np.zeros(me.nelts,dtype=int)

# flow in the SHEAR zone. We must set S*h and 
stor_c = PropertyMap(mat_id,np.array([S_e_oedo]))
thickness = PropertyMap(np.zeros(me.nelts,dtype=int),np.array([h_thickness]))  #shear zone thickness
kh_o = PropertyMap(np.zeros(me.nelts,dtype=int),np.array([Tav]))  #[m^3]   (k_o h)   Initial hydraulic transmissibility

shear_flow_properties={"Initial hydraulic transmissibility":kh_o, #property map
        "Shear zone thickness":thickness,#property map
                 "Storage coefficient":stor_c, # property map
"Fluid viscosity":fluid_visc } # float

shear_zone_flow=ShearZoneFlowCubicLawNewtonian(shear_flow_properties,Nelts)  # instantiate the permeability model

#mechanics
mat_properties = uniform_interface_properties_BB(
    me.nelts, ks, kni, v_m, sig_c, c_p, f_p, f_r, psi_p, wc, d_c, alpha, beta)
interface_model_BB_LW = MixModeW2D(mat_properties,me.nelts,
    linearEvolution,yield_atol=1.0e-5 * (f_p * sigmap_o - tauo),
    tol=1e-6)
# %% Elasticity model
elas_properties=np.array([YoungM, nu])
elastic_m = Elasticity(kernel, elas_properties, max_leaf_size=32, eta=3., eps_aca=1.0e-6,n_openMP_threads=16)
# Hmat creation
hmat = elastic_m.constructHmatrix(me)
# Testing matvec operation
zt = time.process_time()
for i in range(120):
    hmat @ np.ones(2 * Nelts)
elapsed = (time.process_time() - zt) / 120
print("number of active threads for MatVec", hmat.get_omp_threads())
print("elapsed time", elapsed)

#%% Flow model
#  Import injection rate history /local_dev/Basel
file = open('./Stimulation_Q_VS_time_Version_3.json')
QinjTimeData = np.array(json.load(file))
file.close()
list_times=QinjTimeData[::,0]
# injection with a given variation of injection rate
the_inj=Injection(np.array([0.,0.]),QinjTimeData,"Rate",volumetric_compressibility=C_wd)
#
vol_dt = np.array([the_inj.volume(list_times[i],list_times[i+1]) for i in range(len(list_times)-1)])
vol=np.add.accumulate(vol_dt)
vol = np.hstack(([0.],vol))
Q_dt = [the_inj.volume(list_times[i],list_times[i+1])/(list_times[i+1]-list_times[i]) for i in range(len(list_times)-1)]

#-----
fig, ax = plt.subplots()
ax.plot(list_times[5:13] ,vol[5:13] ,'.-b')

myt_t = np.hstack((list_times[:6],list_times[13:]))

#myt_t = np.array([ 4*3600.*i for i in range(25)])

vol_dt = [the_inj.volume(myt_t[i],myt_t[i+1]) for i in range(len(myt_t)-1)]
volc=np.add.accumulate(vol_dt)
volc = np.hstack(([0.],volc))
Q_dt = [the_inj.volume(myt_t[i],myt_t[i+1])/(myt_t[i+1]-myt_t[i]) for i in range(len(myt_t)-1)]

fig, ax = plt.subplots()
ax.plot( myt_t[1:10] ,volc[1:10] ,'-k')
ax.plot(list_times[1:16] ,vol[1:16] ,'.-.b')


fig, ax = plt.subplots()
ax.plot( myt_t[0:-1] ,Q_dt[0:] ,'.-k')
ax.plot(QinjTimeData[0:,0] ,QinjTimeData[0:,1] ,'.-.b')

 # use a smoother injection schedulte
newQinjTimeData=np.array([np.array(myt_t[:-1]), Q_dt])
the_inj=Injection(np.array([0.,0.]),newQinjTimeData.T,"Rate",volumetric_compressibility=C_wd)


# %% models 

# flow model
scale_r_f = 1.0 # scale for flow residuals

flow_model=FlowModelFractureSegments_axis(me, shear_zone_flow, the_inj,scalingR=scale_r_f,scalingX=1.)
# mechanical model
qd=QuasiDynamicsOperator(Nelts,2,eta_s,eta_p)
# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
scale_r_m = 1.0 # scale for mech residuals
mech_model = MechanicalModel(hmat, me.nelts,interface_model_BB_LW,precType="Jacobi",QDoperator=qd,scalingX=1.,scalingR=scale_r_m) #"Jacobi"

# %% reate the simulation object

#
model_config = {  # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh": me,
    "Elasticity": elastic_m,
    "Flow properties":shear_flow_properties,
    "Injection":the_inj,
    "Friction model": interface_model_BB_LW,
    "Material properties": mat_properties,
}

model_parameters = {
    "Elasticity": {"Young": YoungM, "Nu": nu},
    "Flow":{"who": wh_o,"Viscosity":fluid_visc,
            "Compressibility of the shear zone":S_e_oedo,"rock diffusivity":alpha_hydroSZ,"Transmissibility":Tav},
    "Injection history":{"Constant Rate":QinjTimeData,"Wellbore storage":C_wd},
    "Initial stress": [sigmap_o, tauo],
    "Friction coefficient":{"peak":f_p,"residual":f_r,"d_c":d_c,"peak dilatancy":psi_p,"form":'linear'},
    "Interface stiffness":{"ks":ks,"kni":kni,"vm":v_m} 
    }

# %% initial solution

from datetime import datetime
# dd/mm/YY H:M:S
Simul_description="Basel 1 simulation Coupled - Axisymmetric - Variable rate - Variable Perm - BartonBandis stiffness"
now=datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
basename="Axisymmetric-Basel-NonLinearStiffness "

basefolder="./res_data/"+basename+"_"+dt_string+"/" #/local_dev/Basel


sol0=HMFSolution(time=0.,effective_tractions=in_situ_effective_tractions.flatten(), pressure=0.*np.zeros(me.nnodes, dtype=float),
                 DDs=0.*in_situ_effective_tractions.flatten(),
                 DDs_plastic=0.*in_situ_effective_tractions.flatten(),Internal_variables=np.zeros(2*me.nelts),
                 pressure_rate=0.*np.zeros(me.nnodes, dtype=float),
                 DDs_rate=0.*in_situ_effective_tractions.flatten(),
                 res_mech=0.*in_situ_effective_tractions.flatten(),
                 res_flow=0.*np.zeros(me.nnodes, dtype=float))
#%% restart case 
#/home/fakhretd/PyFracX/Basel_scripts/Scripts/res_data/Axisymmetric-Basel-NonLinearStiffness _04_03_2024___23_19_42
restart = False
if restart : 
    basename="Axisymmetric-Basel-NonLinearStiffness "
    basefolder="./res_data/"+basename+"_04_03_2024___23_19_42/"
    from utils.json_dict_dataclass_utils import *
    data_sol=json_read(basefolder+"Axisymmetric-Basel-NonLinearStiffness -23375")
    s0=23375
    # switch to numpy array
    for k,v  in data_sol.items():
        #print(k,"->",v)
        if type(v) == list :
            data_sol[k] = np.array(v)
    sol0 = from_dict_to_dataclass(HMFSolution, data_sol)
else :
    s0=0

# %% Solver  options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
# newton solve options
res_atol =1e-4*scale_r_m* max(np.linalg.norm(in_situ_effective_tractions.flatten()), 1e3)
print("res_atol: %g" %(res_atol))

newton_solver_options=NonLinearSolve_options(max_iterations=15,residuals_atol=res_atol,
                                             residuals_rtol=np.inf,dx_atol=np.inf,dx_rtol=1e-3,
                                             line_search=True,line_search_type="cheap",verbose=True)

# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=300,
    restart_iterations=150,
    absolute_tolerance=0.,
    relative_tolerance=1e-5,
    preconditioner_side="Left",
    schur_ilu_fill_factor=50,
    schur_ilu_drop_tol=1e-3,
    mech_rtol=1e-5,
    mech_atol=0.,
    mech_max_iterations=int(me.nelts/2)
)
# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="BICGSTAB",
                                          jacobian_solver_opts=jac_solve_options,
                                          non_linear_start_factor=0.0,
                                          non_linear_solver_opts=newton_solver_options)

# %% Preparing the time-stepper
def onewayStepWrapper(solF,dt):
    solFNew = hmf_coupled_step(solF, dt, mech_model, flow_model, step_solve_options)
    return solFNew

# - initial time step from pure diffusion 
dt_ini=(1.*h_x)**2 / alpha_hydroSZ   # setting the initial time-step to have something "moving"
#dt_ini=1.
tend=5.4e5
maxSteps=2000  #500000000
min_time_step = 6. 

# options of the time inegration !  note that we also pass the step_solve_options for consistency
ts_options=TimeIntegration_options(max_attempts=10,dt_reduction_factor=2.0,max_dt_increase_factor=1.035,lte_goal=0.001,
                                   acceptance_a_tol=1.0*res_atol,minimum_dt=min_time_step,maximum_dt=600.,
                                   stepper_opts=step_solve_options)

from ts.Event import EnforceTimes
# we try here a simulation without enforcing events
#events=EnforceTimes(list_times)
#events.setTimeStepAfterEvent(4.*min_time_step) # do a small time step after an event
events=None

my_simul=TimeIntegrationApp(basename, model_config, model_parameters, ts_options,onewayStepWrapper,events,description=Simul_description, basefolder=basefolder)
my_simul.setupSimulation(sol0,tend,dt=dt_ini,maxSteps=maxSteps,saveEveryNSteps=5,log_level='INFO',start_step_number=s0) #
if not(restart):
    my_simul.saveParametersToJSon()
    me.saveToJson(basefolder+"Mesh.json")

# %% now we are ready to run the simulation
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("elapsed time",elapsed)

# %% Postprocess, compare with analytical solution
# compute the radial coordinates of the mesh nodes.
x=me.coor[:,0]

Qinj=QinjTimeData[0][1]
tt=np.array([res[j].time for j in range(len(res))])
# # find index of coordinate close to 1. meter
# ind=np.argmin(np.absolute(x-1.))

####################################################################
# plots and checks
# analytical solution for pressure at collocation points (in Pa) - constant pressure case
import scipy.special as sc
def pressure(r,t,Qinj):
    return fluid_visc/Tav * (-Qinj/(4*np.pi))*sc.expi(-r*r/(4*alpha*t))

import matplotlib.pyplot as plt

####################################################################
# plots and checks
timestep=-2
num_pressure=res[timestep].pressure
#true_sol=pressure(x,tt[timestep],Qinj)


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
timestep=-1
sol_to_p= res[timestep]
x=me.coor[:,0]
num_pressure=sol_to_p.pressure
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot(x[1:400], num_pressure[1:400],'.b')
plt.xlabel("  r (m)")
plt.ylabel("fluid pressure (Pa)")
plt.show()

#%%
# width profile
w_profile=sol_to_p.DDs[1::2]
w_profile_p=sol_to_p.DDs_plastic[1::2]
fig, ax = plt.subplots()
# ax.plot(x[1:],true_sol[1:],'r')
ax.plot((x[1:]+x[0::-2])/2., w_profile[:],'.k')
ax.plot((x[1:]+x[0::-2])/2., w_profile_p[:],'.b')
plt.xlabel("  r (m)")
plt.ylabel(" width (m)")
plt.show()

fig, ax = plt.subplots()

ax.plot((x[1:]+x[0::-2])/2., (1+w_profile[:]/wh_o)**3,'.k')
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

newton_its = np.array([ res[i].stats.iteration for i in range(len(res))])
elap_time = np.array([ res[i].stats.elapsed_time for i in range(len(res))])
step_time = np.diff(elap_time)
step_time=np.insert(step_time,0,step_time[0])
n_yield_step = np.array([ res[i].Nyielded for i in range(len(res))])

#%%
nmvt_step = 0.*newton_its 

for i in range(len(res)):
    for k in range(newton_its[i]-1):
        nmvt_step[i]+=res[i].stats.jacobian_stat_list[k]['Total number of A11 matvect: ']

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
plt.ylim([0, 50])
# %%
# %%
fig, ax = plt.subplots()
ax.plot(newton_its,step_time,'.')
plt.ylim([0, 50])

# %%
ig, ax = plt.subplots()
ax.plot(nmvt_step/newton_its,step_time,'.')
plt.ylim([0, 50])
# %%
