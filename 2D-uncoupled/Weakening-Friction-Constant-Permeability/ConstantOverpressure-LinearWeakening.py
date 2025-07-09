#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a planar frictional fault (plane-strain problem) due to a constant over-pressure
# Linear weakening friction reference results from Germanovich & Garagash (2012) 
# parameters taken from Ciardo et al IJNME 2021 - do not change if you want to compare with the ref solution
#

#%% imports
import matplotlib.pyplot as plt
from scipy import special
import time,sys

from mesh.usmesh import usmesh
from mechanics.H_Elasticity import *
from MaterialProperties import PropertyMap
from mechanics.mech_utils import *
from mechanics.friction2D import *
from mechanics.evolutionLaws import *
from hm.HMFsolver import HMFSolution,hmf_one_way_step
from src.ts.TimeStepper import *
from utils.App import TimeIntegrationApp

#%% Material parameters, in-sity stress & over-pressure
# material elasticity
YoungM=1.0
nu=0.0
shear_prime=YoungM/(2*(1+nu)*2*(1-nu))
shear_g = YoungM/(2*(1+nu))
c_s = np.sqrt(shear_g/(1.))
c_p = c_s * np.sqrt(2*(1-nu)/(1.-2*nu))
# fault friction
f_p=1.
psi_p=0. 
f_r=0.6*f_p
d_c=0.4
cohesion=0.
###
# initial effective tractions
sigmap_o =1.0
tau_o=0.55
# injection overpressure
dpcenter=0.5
alpha_hyd=10.      #hydraulic diffusivity of the fault

# corresponding parameters dimensionless parameters
tau_o-f_p*(sigmap_o+dpcenter)
T=(1-tau_o/(f_p*sigmap_o))*sigmap_o/dpcenter
tau_o/(f_p*sigmap_o)
T_r=(1-tau_o/(f_r*sigmap_o))*sigmap_o/dpcenter

#%% mesh and model creation 
# simple 1D mesh
Nelts=2000
coor1D=np.linspace(-10.,10.,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
me=usmesh(2,coor,conn,0)
colPts=(coor1D[1:]+coor1D[0:-1])/2.# collocation points for P0

# analytical solution for pressure at collocation points for a constant over-pressure at the center
pressure = lambda x,t,Dpcenter: Dpcenter * special.erfc(np.abs(x)/((4.*alpha_hyd*t)**0.5))
pressureAtColPts = lambda t,Dpcenter: Dpcenter * special.erfc(np.abs(colPts)/((4.*alpha_hyd*t)**0.5))

# Elasticity discretization via boundary element - plane-strain piece-wise constant displacement discontinuity element
kernel="2DS0-H"
elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel,elas_properties,max_leaf_size=80,eta=3,eps_aca=1.e-4)
# BE hierarchical matrix creation 
h1=elastic_m.constructHmatrix(me)

#### Populate properties
# initial in-situ conditions  over the mesh
in_situ_tractions =np.full((Nelts,2),[-tau_o,-sigmap_o])  # positive stress in traction !

# setting frictional properties for the linear weakening elasto-plastic interface model
friction_p=PropertyMap(np.zeros(Nelts,dtype=int),np.array([f_p]))           # friction coefficient
friction_r=PropertyMap(np.zeros(Nelts,dtype=int),np.array([f_r]))           # friction coefficient
dilatant_p=PropertyMap(np.zeros(Nelts,dtype=int),np.array([psi_p]))           # dilatancy coefficient
cohesion_p=PropertyMap(np.zeros(Nelts,dtype=int),np.array([0.]))
slip_dc = PropertyMap(np.zeros(Nelts,dtype=int),np.array([d_c]))
k_sn=PropertyMap(np.zeros(Nelts,dtype=int),np.array([[100*YoungM/(2*(1+nu)),100*YoungM]])) # springs shear, normal

mat_properties={"Peak friction":friction_p,"Residual friction":friction_r,"Peak dilatancy coefficient":dilatant_p,"Peak cohesion":cohesion_p,
                "Critical slip distance":slip_dc,"Spring Cij":k_sn,
                "Elastic parameters":{"Young":YoungM,"Poisson":nu}}

frictionModel_LW=FrictionVar2D(mat_properties,Nelts,linearEvolution,tol=1.e-6,yield_atol=1e-6*sigmap_o)  # linear frictional weakening model

# quasi-dynamics term (put here to a small value)
eta_s = 1.e-3*0.5 *shear_g / c_s
eta_p = 1.e-3*0.5 *shear_g / c_p
qd=QuasiDynamicsOperator(Nelts,2,eta_s,eta_p)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech = MechanicalModel(h1, me.nelts, frictionModel_LW,precType="Jacobi",QDoperator=qd)

# initial solution
sol0=HMFSolution(time=0.,effective_tractions=in_situ_tractions.flatten(),pressure=np.zeros(me.nnodes),DDs=0.*in_situ_tractions.flatten(),DDs_plastic=0.*in_situ_tractions.flatten(),
                 pressure_rate=np.zeros(me.nnodes),DDs_rate=0.*in_situ_tractions.flatten(),Internal_variables=np.zeros(2*me.nelts),
                 res_mech=0.*in_situ_tractions.flatten())

# stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_a_tol = 1.e-5*max(sigmap_o*me.nelts,1e3) # absolute convergence tolerance on mechanical residualts.
newton_solver_options=NonLinearSolve_options(max_iterations=20,residuals_atol=res_a_tol,
                                             residuals_rtol=np.inf,dx_atol=np.inf,dx_rtol=1e-3,
                                             line_search=True,
                                             line_search_type="cheap")
# options for the jacobian solver 
jac_solve_options=IterativeLinearSolve_options(max_iterations=200,absolute_tolerance=1.e-12,
                                               relative_tolerance=1.e-6,restart_iterations=150)
# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="GMRES",
                                          jacobian_solver_opts=jac_solve_options,non_linear_start_factor=0.,
                                          non_linear_solver_opts=newton_solver_options)

## function wrapping the one-way / uncoupled solver for this case
Dtinf=np.zeros(2 * me.nelts)
def onewayStepWrapper(solN,dt):
    time_s = solN.time + dt 
    # compute pressure increment at nodes form the analytical solution 
    dp = pressure(coor1D, time_s,dpcenter) - solN.pressure
    # at collocation points
    if solN.time==0.:
        dpcol = pressureAtColPts(time_s, dpcenter)
    else:
        dpcol = pressureAtColPts(time_s, dpcenter) - pressureAtColPts(solN.time, dpcenter)
    solTnew = hmf_one_way_step(solN, dt, mech, dpcol, Dtinf, step_solve_options)
    solTnew.pressure = solTnew.pressure + dp
    return solTnew


#  Preparing folders and base dict for savings
from datetime import datetime
Simul_description="2D plane-strain - ct pressure - linear weak. friction - one way simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="2D-ctP-LinearWeakening-MarginallyPressurized"
basefolder="./res_data/"+basename+"-"+dt_string+"/"
model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":me,
    "Elasticity":elastic_m,
    "Friction model": frictionModel_LW,
    "Material properties": mat_properties,
}
model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu},
    "Flow":{"Hydraulic diffusivity":alpha_hyd},
    "Injection":{"P center":dpcenter},
    "Initial stress":[sigmap_o,tau_o],
    "Friction peak":f_p,"Friction residual":f_r,"Dilatancy peak":psi_p,"Dc":d_c,
    "T parameter":T
}

# prepare the time-stepper simulations
new_dt=1.e-4      # initial time-step
maxSteps=160      #120 max number of stpes of the simulation
tend=1e6          # max time to simulate

# options of the time inegration !  note that we also pass the step_solve_options
ts_options=TimeIntegration_options(max_attempts=4,dt_reduction_factor=2.,max_dt_increase_factor=1.25,
                                   lte_goal=0.005,
                                   acceptance_a_tol=res_a_tol,
                                   minimum_dt=new_dt/100.0,stepper_opts=step_solve_options)

my_simul=TimeIntegrationApp(basename, model_config, model_parameters, ts_options,onewayStepWrapper,
                            description=Simul_description, basefolder=basefolder)
my_simul.setAdditionalStoppingCriteria(lambda sol : sol.Nyielded==me.nelts)

my_simul.setupSimulation(sol0,tend,dt=new_dt,maxSteps=maxSteps,saveEveryNSteps=1)
my_simul.saveParametersToJSon()
#my_simul.saveConfigToBinary()
me.saveToJson(basefolder+"Mesh.json")
# if (os.path.exists(simul_definition["Simulation folder"])==False):
#     os.mkdir(simul_definition["Simulation folder"])


#%% now we are ready to run the simulation
zt= time.process_time()
res,status_ts=my_simul.run()
elapsed=time.process_time()-zt
print("elapsed time",elapsed)
#%%
#---------------------------------------------
# post-process to estimate crack front position
# sol
from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *

lam=marginallyPressurized_lambda(T)
lam_r=criticallyStressed_lambda(T_r)
# xb=np.linspace(-1.,1.,num=50,endpoint=True)
# slip_in=criticallyStressed_slip_Inner(xb*lam,lam)
# slip_out=criticallyStressed_slip_Outer(xb)

tts=np.array([res[i].time for i in range(len(res))])
nyi=np.array([res[i].Nyielded for i in range(len(res))])
cr_front = nyi/2.*(coor1D[1]-coor1D[0])

#  solution for crack half length

t_ = np.linspace(0.001, tts.max(), 1000)
y_ = lam*np.sqrt(4.*alpha_hyd*t_)
yr_= lam_r*np.sqrt(4.*alpha_hyd*t_)
aw = 0.5
fig, ax = plt.subplots()
ax.plot(np.sqrt(4*alpha_hyd*t_)/.5,y_/.5,'r')
ax.plot(np.sqrt(4*alpha_hyd*t_)/.5,np.sqrt(4.*alpha_hyd*t_)/.5,'b')
ax.plot(np.sqrt(4*alpha_hyd*tts)/.5,cr_front/.5,'.')
#ax.plot(np.sqrt(4*alpha_hyd*(t_-0.1))/.5,yr_/.5,'.')
plt.xlabel("$\sqrt{4 \\alpha t}   / a_w$")
plt.ylabel("Crack half-length $a / a_w$")
ax.legend(["Early time analytical solution",
           "Diffusion front","numerics"])
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
plt.ylabel("LTEstimate")
# ax.legend(["analytical solution","numerics"])
plt.show()

#%%
slip_0 = np.abs(np.array([res[i].DDs_plastic[Nelts] for i in range(len(res))]))
fig, ax = plt.subplots()
ax.loglog( tts ,slip_0 ,'.')
plt.xlabel("Time (s)")
plt.ylabel("slip @ injection point (m)")
# ax.legend(["analytical solution","numerics"])
plt.show()


#%%
all_its=np.array([res[i].stats.iteration for i in range(len(res))])

#  PLOTTING slip profile ----
#%
# 1 check similarity - plot all scaled slip profiles for every steps ...
fig, ax = plt.subplots()
for jj in range(len(res)-1):
    t = res[jj].time
    a = cr_front[jj]
    slip_fact=np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    xb=colPts/a
    scaled_slip=-res[jj].DDs_plastic[0::2]/slip_fact
    ax.plot(xb,scaled_slip,'-')

plt.xlim([-2,2])
plt.ylim([-0.1,1.8])
plt.xlabel("x/a(t) ")
plt.ylabel("scaled slip")
plt.show()



#%%

#---- PLOTTING PROFILES ----
jj=len(res)-12
fig, ax = plt.subplots()
solp=pressureAtColPts(res[jj].time,dpcenter)
ax.plot(colPts,solp)
ax.plot(coor1D,res[jj].pressure,'.')
plt.xlabel("x (M)")
plt.ylabel("Over-pressure (Pa)")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-res[jj].DDs_plastic[0::2])
plt.xlabel("x (m)")
plt.ylabel("plastic slip (m)")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs[0::2]-res[jj].DDs_plastic[0::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic slip (m)")
plt.show()


fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-res[jj].DDs[0::2])
plt.xlabel("x (m)")
plt.ylabel("Total slip (m)")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:] + coor1D[0:-1]) / 2., res[jj].DDs_plastic[1::2])
plt.xlabel("x (m)")
plt.ylabel("plastic opg (m)")
plt.show()

fig, ax = plt.subplots()
ax.plot((coor1D[1:]+coor1D[0:-1])/2.,-(res[jj].DDs[1::2]-res[jj].DDs_plastic[1::2]))
plt.xlabel("x (m)")
plt.ylabel("elastic Opg (m)")
plt.show()

#%%
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
plt.xlabel("x (m)")
plt.ylabel("Yield function (Pa)")
plt.show()

