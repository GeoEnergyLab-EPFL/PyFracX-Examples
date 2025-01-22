#
# This file is part of PyFracX.
#
# Created by Brice Lecampion on 9.12.2021
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#
#
#%%  2D  HF fracture re-opening

import matplotlib.pyplot as plt
import time

from mesh.usmesh import usmesh
from mechanics.H_Elasticity import *
from mechanics.friction2D import FrictionCt2D
from mechanics.mech_utils import *

from flow.FlowConstitutiveLaws import *
from flow.flow_utils import *
from loads.Injection import *
from hm.HMFsolver import HMFSolution,hmf_coupled_step

from ReferenceSolutions.HF.KGDHFsolutions import *

# Parameters
YoungM=35.e9
nu=0.4
shear_prime = YoungM / (2 * (1 + nu) * 2 * (1 - nu))
E_prime = YoungM/(1-nu*nu)
# flow
wh_o =1.e-5    #(1.e-12)**(1./3)
fluid_visc = 1.0/12
cond_hyd=(wh_o**3/(12*fluid_visc) ) #   m^3/ (Pa.s)
c_f =5.0e-10  # 1./ Pa
# hydraulic diffusivity
k_frac=wh_o**2/12
alpha_h = (k_frac/(fluid_visc*c_f) )

# fault friction
f_p=0.6
f_d=0.0
# injection rate
Qinj=5.e-6 #  m^2/s injected fluid rate in the total fracture

kgd_m=KGD_Mvertex(Ep=E_prime,Qo=Qinj,mu=fluid_visc)

# far-field effective tractions
tau_o=6e6*f_p*0
sig_o_p=15e6
p_f_o=0.

# ratio of shear patch pressurization scale over opening patch pressurization scale
Sc = (tau_o/(f_p*sig_o_p))
One_minus_S=1.0-Sc
One_minus_S**2      # ratio of timescale for shear activation over opening activation...

L_s = (f_p*sig_o_p-tau_o)*k_frac*(np.sqrt(4*np.pi)) / (f_p*Qinj*fluid_visc/wh_o)
t_s = (L_s**2/(4.*alpha_h))   # time of shear activation
#
L_op = L_s/One_minus_S
t_op =(L_op**2 /4./alpha_h)  # time of opening activation

# t_lambda
t_lambda=E_prime**2 * (12*fluid_visc)/(sig_o_p**3)

# simple 1D mesh uniform
Nelts=2000
Lc=50
coor1D=np.linspace(-Lc,Lc,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me=usmesh(2,coor,conn,0)
colPts=(coor1D[1:]+coor1D[0:-1])/2.# collocation points for P0
h_x=coor[1,0]-coor[0,0]

# initial in-situ conditions ----
in_situ_tractions =np.full((Nelts,2),[-tau_o,-sig_o_p])  # positive stress in traction !

# initial DDs.
DDs_initial=0.*in_situ_tractions

#%% Elasticity model
kernel="2DS0-H"

elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel,elas_properties,max_leaf_size=32,eta=3.,eps_aca=1.e-5)
# hmat creation
h1=elastic_m.constructHmatrix(me)

zt= time.process_time()
for i in range(120) :
    h1.dot(np.ones(2*Nelts))
elapsed=(time.process_time()-zt)/120
print("elapsed time for matvect",elapsed)

#%% effective normal stress
effective_tractions_0=in_situ_tractions

#-------- # initial pore pressure
# pore pressure at nodes. - zero everywhere except in the fracture
P_f_nodes=np.zeros(me.nnodes,dtype=float)

# setting frictional properties
friction_c=PropertyMap(np.zeros(Nelts,dtype=int),np.array([f_p]))           # friction coefficient
dilatant_c=PropertyMap(np.zeros(Nelts,dtype=int),np.array([f_d]))           # dilatancy coefficient
k_sn=PropertyMap(np.zeros(Nelts,dtype=int),np.array([[100*YoungM/(2*(1+nu)),100*YoungM]])) # springs shear, normal
mat_properties={"Friction coefficient":friction_c,"Dilatancy coefficient":dilatant_c,"Spring Cij":k_sn,"Elastic parameters":{"Young":YoungM,"Poisson":nu}}
frictionModel=FrictionCt2D(mat_properties,Nelts)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech_model = MechanicalModel(h1, me.nelts, frictionModel,precType="Jacobi")

# injection under constant rate
the_inj=Injection(np.array([0.,0.]),np.array([[0.,Qinj]]),"Rate")
###  Creation of the flow model :: flow cubic law
hyd_width=PropertyMap(np.zeros(me.nelts,dtype=int),np.array([wh_o]))  #uniform hydraulic width
comp_c=PropertyMap(np.zeros(me.nelts,dtype=int),np.array([ c_f]))  #
flow_properties={"Fluid viscosity":fluid_visc,"Initial hydraulic width": hyd_width,"Compressibility":comp_c}
cubicModel=CubicLawNewtonian(flow_properties,me.nelts)   # instantiate a cubic law model
flow_model=FlowModelFractureSegments(me, cubicModel, the_inj)

# initial solution object
sol0=HMFSolution(time=0.0,effective_tractions=effective_tractions_0.flatten(),pressure=P_f_nodes,DDs=DDs_initial.flatten(),DDs_plastic=DDs_initial.flatten(),
                 pressure_rate=0.*P_f_nodes,DDs_rate=0.*effective_tractions_0.flatten(),Internal_variables=np.zeros(0),
                 res_mech=0.*effective_tractions_0.flatten(),res_flow=0.*P_f_nodes)

# options for the time-stepper solver
#stepper_options={"Verbose":True,"Time integration":{"LTE goal": 0.005,"Acceptance res tolerance":100.,"Number of attempts":6,"Step decrease factor":1.5,"Maximum step factor":2.},
#    "Newton":{"Norm order":2,"Max iterations":22,"Residuals rtol":np.inf,"Residuals tol":100,"Dx rtol":1.e-3,"Dx tol":np.inf,"Line search":True},
#                 "Tangent solver":{"Mechanics":{"GMRES":{"Max iterations":150,"Tolerance":1.e-8,"Absolute tolerance":1e-2,"Restart":150,"Preconditioner side":"Left"},"Initial guess factor":0.1}}}
# stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_atol = 1e-3* max(np.linalg.norm(effective_tractions_0.flatten()), 1e3)
print("res_atol: %g" %(res_atol))
newton_solver_options=NonLinearSolve_options(max_iterations=20,residuals_atol=res_atol,residuals_rtol=np.inf,dx_atol=np.inf,dx_rtol=1e-2,line_search=True,line_search_type='cheap')
# options for the jacobian solver
# options for the jacobian solver
jac_solve_options = IterativeLinearSolve_options(
    max_iterations=50,
    restart_iterations=50,
    absolute_tolerance=np.inf,
    relative_tolerance=1e-4,
    preconditioner_side="Left",
    schur_ilu_fill_factor=50,
    schur_ilu_drop_tol=1e-4,
    mech_rtol=1e-4,
    mech_atol=1,
    mech_max_iterations=me.nelts/2
)
# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="GMRES",
                                          jacobian_solver_opts=jac_solve_options,
                                          non_linear_start_factor=0.0,
                                          non_linear_solver_opts=newton_solver_options)

#Create the simulation object
from datetime import datetime
# dd/mm/YY H:M:S
Simul_description="2D plane-strain - constant rate - with zero shear stress"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
basename="2D-ConstantRate-HF-reopening"
basefolder="./"+basename+"-"+dt_string+"/"
model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":me,
    "Elasticity":elastic_m,
    "Friction model": frictionModel,
    "Material properties": mat_properties,
    "Flow properties":flow_properties,
    "Injection":the_inj
}
# in this dict, we store parameters and description...
model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu,"Kernel":kernel},
    "Flow":{"who": wh_o,"Viscosity":fluid_visc,"Compressibility":c_f},
    "Injection":{"Constant Rate":Qinj},
    "Initial stress":[tau_o,sig_o_p],
    "Friction coef":f_p,
    "Stress criticality":Sc,
    "Lop ": L_op,"Ls":L_s
}

def StepWrapper(solN,dt):
    solNew = hmf_coupled_step(solN, dt, mech_model, flow_model, step_solve_options)
    return solNew

#%%
from utils.App import TimeIntegrationApp

dt_ini=0.1
maxSteps=300 #800
tend=1.5e4
# options of the time inegration !  note that we also pass the step_solve_options
ts_options=TimeIntegration_options(max_attempts=4,dt_reduction_factor=1.5,max_dt_increase_factor=1.2,lte_goal=0.01,
                                   acceptance_a_tol=res_atol,minimum_dt=dt_ini/10.0,stepper_opts=step_solve_options)

my_simul=TimeIntegrationApp(basename, model_config, model_parameters, ts_options,StepWrapper,description=Simul_description, basefolder=basefolder)
my_simul.setAdditionalStoppingCriteria(lambda sol : sol.Nyielded==me.nelts)

my_simul.setupSimulation(sol0,tend,dt=dt_ini,maxSteps=maxSteps,saveEveryNSteps=1,log_level='INFO')
my_simul.saveConfigToBinary()
my_simul.saveParametersToJSon()
me.saveToJson(basefolder+"Mesh.json")

#%% now we are ready to run the simulation
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("elapsed time",elapsed)

#my_simul.archiveSimulation()  # zip all files


#%%

n_k=[]
n_k_nit=[]
for i0 in range(len(res)):
    k=0
    nit=len(res[i0].stats.jacobian_stat_list)
    for i in range(nit):
        k=k+res[i0].stats.jacobian_stat_list[i]['Total number of A11 matvect: ']+nit
    n_k.append(k)
    n_k_nit.append(k/nit)


#%%---------------------------------------------
# post-process to estimate crack front position
# sol
tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi/2.*(coor1D[1]-coor1D[0])

# analytical sol  M vertex
t_ = np.linspace(0.01, tts.max(), 1000)
y_ = kgd_m.half_length(t_)

fig, ax = plt.subplots()
ax.loglog(t_,y_,'r')
ax.loglog(tts,cr_front,'.')
plt.xlabel("Time (s)")
plt.ylabel("Crack half-length (m)")
ax.legend(["analytical solution","numerics"])
plt.show()

half_length_true=kgd_m.half_length(tts)
rel_error=np.abs(cr_front-half_length_true)/half_length_true

fig, ax = plt.subplots()
ax.loglog(tts,rel_error,'.')
plt.xlabel("Time (s)")
plt.ylabel(" Rel. Error fracture length")
plt.show()
#

half_length_true=kgd_m.half_length(tts)
lamb_=np.abs(cr_front/half_length_true)

fig, ax = plt.subplots()
ax.plot(tts,lamb_,'.')
ax.plot(tts,tts*0+1.236,'.')

plt.xlabel("Time (s)")
plt.ylabel(" \ell_s / \ell")
plt.show()
#

pp_flow= (Qinj/wh_o)*np.sqrt(4.*alpha_h*tts) / (np.sqrt(4*np.pi)*k_frac)
ppi=np.array([res[i].pressure[the_inj.locate_in_mesh(me)] for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog(tts,ppi*1e-6,'.')
#ax.loglog(tts,pp_flow*1e-6,'.')
plt.xlabel("Time (s)")
plt.ylabel(" fluid pressure at inj. (MPa)")
plt.show()


#---- PLOTTING PROFILES ----
jj=len(res)-1   # choosing the step to profile

xi_list=np.linspace(-0.9999,0.9999,60)
net_p_m=kgd_m.pressure(res[jj].time,np.abs(xi_list))
width_m=kgd_m.width(res[jj].time,np.abs(xi_list))
x_list=(kgd_m.half_length(res[jj].time))* xi_list

fig, ax = plt.subplots()
#ax.plot(colPts,solp)
ax.plot(coor1D[:],(res[jj].pressure)[:] ,'.') # 900:1100
ax.plot(x_list,net_p_m+sig_o_p,'.r-')  # add the initial sigma_o
plt.xlabel("x (M)")
plt.ylabel("Fluid pressure (Pa)")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+ coor1D[0:-1]) / 2., res[jj].DDs[1::2])
ax.plot(x_list,width_m,'.r-')
plt.xlabel("x (m)")
plt.ylabel(" fracture opening (m)")
plt.show()

#
# ax.plot(coor1D[:],(res[jj].res_flow)[:] ,'.-')
# plt.xlabel("x (M)")
# plt.ylabel(" res flow")
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].res_mech[1::2])
# plt.xlabel("x (m)")
# plt.ylabel("res mech ")
# plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("plastic slip  (m)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs_plastic[0::2]-res[jj-1].DDs_plastic[0::2])/res[jj].timestep)
plt.xlabel("x (m)")
plt.ylabel("plastic slip rate (m)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs[0::2]-res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic slip  (m)")
plt.show()

# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs[0::2]-res[jj].DDs_plastic[0::2]))
# plt.xlabel("x (m)")
# plt.ylabel("elastic slip (m)")
# plt.show()

# fig, ax = plt.subplots()
# ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-res[jj].DDs[0::2])
# plt.xlabel("x (m)")
# plt.ylabel("Total slip (m)")
# plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,(res[jj].DDs[1::2]-res[jj].DDs_plastic[1::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic Opg (m)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2., (res[jj].DDs[1::2]+wh_o)**3 )
ax.set_yscale('log')
plt.xlabel("x (m)")
plt.ylabel("Current flow transmissivity")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[1::2],'.')
plt.xlabel("x (m)")
plt.ylabel("Effective normal stress (Pa)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[0::2],'.')
plt.xlabel("x (m)")
plt.ylabel("Shear stress (Pa)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].yieldF)
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[1::2],'.')
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,res[jj].effective_tractions[0::2],'.')

plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa)")
plt.show()

