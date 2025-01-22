#
# This file is part of PyFracX-Examples
#

#
# UnCoupled fluid injection into a 2D planar frictional fault (plane-strain problem) due to a constant over-pressure
# Constant friction reference results from Viesca (2021)
# All properties are constant

#%% imports
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import time

from mesh.usmesh import usmesh
from mechanics.H_Elasticity import *
from MaterialProperties import PropertyMap
from mechanics.friction2D import *
from mechanics.mech_utils import *
from hm.HMFsolver import HMFSolution, hmf_one_way_step
from ts.TimeStepper import *
from utils.App import TimeIntegrationApp


#%% Material parameters, in-sity stress & over-pressure
# material elasticity
YoungM=48.e9
nu=0.2
shear_prime=YoungM/(2*(1+nu)*2*(1-nu))

# fault friction
f_p=0.6
# dilatancy
psi_p=0
# initial effective tractions
sigmap_o =90.e6
tau_o=50.e6
# injection overpressure
T=0.1
dpcenter=(1.-tau_o/(f_p*sigmap_o))*sigmap_o/T
alpha_hyd=10.e-2 

# corresponding parameters dimensionless parameters
#T=(1-tau_o/(f_p*sigmap_o))*sigmap_o/dpcenter
print("T=", T)

#%% mesh and model creation 
# simple 1D mesh
Nelts=4000
coor1D=np.linspace(-10.,10.,Nelts+1)
coor = np.transpose(np.array([coor1D,coor1D*0.]))
conn=np.fromfunction(lambda i, j: i + j, (Nelts, 2), dtype=int)
colPts=(coor1D[1:]+coor1D[0:-1])/2. # collocation points for P0
# BE hierarchical matrix creation 
me=usmesh(2,coor,conn,0)

# analytical solution for pressure at collocation points for a constant over-pressure at the center
pressure = lambda x,t,Dpcenter: Dpcenter * special.erfc(np.abs(x)/((4.*alpha_hyd*t)**0.5))
pressureAtColPts = lambda t,Dpcenter: Dpcenter * special.erfc(np.abs(colPts)/((4.*alpha_hyd*t)**0.5))

# Elasticity discretization via boundary element - plane-strain piece-wise constant displacement discontinuity element
kernel="2DS0-H"
elas_properties=np.array([YoungM, nu])
elastic_m=Elasticity(kernel,elas_properties,max_leaf_size=16,eta=10.,eps_aca=1.e-6,n_openMP_threads=16)
# BE hierarchical matrix creation 
h=elastic_m.constructHmatrix(me)

#### Populate properties
# initial in-situ conditions  over the mesh
in_situ_tractions =np.full((Nelts,2),[-tau_o,-sigmap_o])  # positive stress in traction !

# setting frictional properties for the constant friction elasto-plastic interface model
friction_c=PropertyMap(np.zeros(Nelts,dtype=int),np.array([f_p]))           # friction coefficient
dilatant_c=PropertyMap(np.zeros(Nelts,dtype=int),np.array([psi_p]))           # dilatancy coefficient
k_sn=PropertyMap(np.zeros(Nelts,dtype=int),np.array([[100*YoungM/(2*(1+nu)),100*YoungM]])) # springs shear, normal

mat_properties={"Friction coefficient":friction_c,"Dilatancy coefficient":dilatant_c,"Spring Cij":k_sn,"Elastic parameters":{"Young":YoungM,"Poisson":nu}}

frictionModel=FrictionCt2D(mat_properties,Nelts,yield_atol=1.e-6*sigmap_o)

# creating the mechanical model: hmat, preconditioner, number of collocation points, constitutive model
mech = MechanicalModel(h,me.nelts,frictionModel,precType="ILUT")#"Jacobi+")
mech.always_update_pc_=True

# initial solution
sol0=HMFSolution(time=0.,effective_tractions=in_situ_tractions.flatten(),pressure=np.zeros(me.nnodes),DDs=0.*in_situ_tractions.flatten(),DDs_plastic=0.*in_situ_tractions.flatten(),
                 pressure_rate=np.zeros(me.nnodes),DDs_rate=0.*in_situ_tractions.flatten(),Internal_variables=np.zeros(2*me.nelts),
                 res_mech=0.*in_situ_tractions.flatten(),res_flow=np.zeros(me.nnodes))

# stepper options
from utils.options_utils import TimeIntegration_options,NonLinear_step_options,NonLinearSolve_options,IterativeLinearSolve_options
# newton solve options
res_a_tol = 1.e-3*max(sigmap_o*me.nelts,1e3) # absolute convergence tolerance on mechanical residualts.
newton_solver_options=NonLinearSolve_options(max_iterations=20,residuals_atol=res_a_tol,
                                             residuals_rtol=np.inf,dx_atol=np.inf,dx_rtol=1e-4,
                                             line_search=True,
                                             line_search_type="cheap")
# options for the jacobian solver 
jac_solve_options=IterativeLinearSolve_options(max_iterations=200,absolute_tolerance=1e-6,
                                               relative_tolerance=1e-6,restart_iterations=150)
# combining the 2 as option for the non-linear time-step
step_solve_options=NonLinear_step_options(jacobian_solver_type="Bicgstab",
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
Simul_description="2D plane-strain - ct pressure - ct friction - one way simulation"
now=datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
basename="2D-ctP-ctFriction"
basefolder="./"+basename+"-"+dt_string+"/"
model_config={ # in this dict, we store object etc. (but not the hmat that is not storable for now)
    "Mesh":me,
    "Elasticity":elastic_m,
    "Friction model": frictionModel,
    "Material properties": mat_properties,
}
model_parameters={
    "Elasticity":{"Young":YoungM,"Nu":nu},
    "Flow":{"Hydraulic diffusivity":alpha_hyd},
    "Injection":{"P center":dpcenter},
    "Initial stress":[sigmap_o,tau_o],
    "Friction":f_p,"Dilatancy":psi_p,
    "T parameter":T
}

# prepare the time-stepper simulations
new_dt=1e-1      # initial time-step
maxSteps=90      #120 max number of stpes of the simulation
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
me.saveToJson(basefolder+"Mesh.json")

#%% now we are ready to run the simulation
zt= time.time() #time.process_time()
res,status_ts=my_simul.run()
elapsed=time.time() -zt
print("elapsed time",elapsed)

#%%
#---------------------------------------------
# post-process to estimate crack front position
# sol
from ReferenceSolutions.FDFR.Plane_TwoD_frictional_ruptures import *

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

#%%  Plot: rel_err VS time
rel_err_fr=np.abs(y_-cr_front)/y_

fig, ax = plt.subplots()
ax.loglog(nyi,rel_err_fr,'.')
plt.xlabel("Number of yielded elements")
plt.ylabel("Relative error (-)")
plt.show()

#%% time - stepping
dts = np.array([res[i].timestep for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog( tts ,dts ,'.')
plt.xlabel("Time (s)")
plt.ylabel("time-step")
plt.show()

#%% Plot: LTE VS time
ltes = np.array([res[i].lte for i in range(len(res))])
fig, ax = plt.subplots()
ax.loglog( tts ,ltes ,'.')
plt.xlabel("Time (s)")
plt.ylabel("LTE estimate")
plt.show()

#%% Info about number of iterations per Newton's step
all_its=np.array([res[i].stats.iteration for i in range(len(res))])

#%%---- PLOTTING slip profile ----
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

#%% Plot: Overpressure
jj=len(res)-2
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
plt.ylabel("elastic opg (m)")
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

print("Slip at zero in critically stressed limit is = ", criticallyStressed_slip0(lam))

#%% Plot: Slip profile (Dimensionless)
if T > 0.4:
    xx = np.linspace(-1, 1, Nelts)
    #analytical_slip = marginallyStressed_slip(xx)
    analytical_slip = marginallyPressurized_slip(xx)
    slip_scale_MP =lam**2*np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    numerical_slip = -res[jj].DDs[0::2] / slip_scale_MP
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
    numerical_slip = -res[jj].DDs[0::2] / slip_scale_CS

    fig, ax = plt.subplots()
    ax.plot( xx / cr_front[jj], numerical_slip)

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
    numerical_slip = -res[jj].DDs[0::2]
    fig, ax = plt.subplots()
    ax.plot(xx, analytical_slip,'r.')
    ax.plot( xx , numerical_slip,'b-')
    plt.xlabel("x (m)")
    plt.ylabel("Dimensional Slip profile (m))")
    ax.legend(['Analytical','Numerical'])
    ax.set_xlim([-cr_front[jj], cr_front[jj]])  # range for x-axis
    plt.show()
else:
    xx = (coor1D[1:] + coor1D[0:-1]) / 2
    xx_new = xx[(xx < cr_front[jj]) & (xx > -cr_front[jj])]

    slip_scale_CS =np.sqrt(4*alpha_hyd*t)*f_p*dpcenter/shear_prime
    numerical_slip = -res[jj].DDs[0::2]

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

# %%
