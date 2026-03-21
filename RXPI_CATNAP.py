# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:01:27 2026

@author: Maximilian Slavik


################################


RXPI CATNAP

Coupled Adiabatic Tank and Nozzle Analysis Program


⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡤⠶⠒⠛⠉⠙⠛⠒⠶⢤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠁⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠈⠙⢦⡀⠀⠀⠀
⠀⠀⠀⠀⣀⣠⠤⠤⡴⠻⢓⣶⠦⠤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀
⣀⡠⠴⠊⠁⠀⠀⠀⠀⠀⠒⠽⠀⠀⠀⠉⢙⠒⢢⡄⠀⠀⠀⠀⠀⠀⠀⠈⢷⡀
⠘⢆⠉⠑⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢆⠁⣠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣇
⠀⠈⠓⠦⠁⢀⣀⠀⠀⠀⠀⠀⠀⣀⣀⢸⡊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻
⠀⠀⠀⡸⠀⠉⠀⠙⠆⠀⠀⠀⠏⠀⠈⠀⢇⠀⠀⠀⠀⠀⠀⠀⢀⠔⠀⠀⠀⣼
⠀⠀⠀⠹⣄⠀⠠⣤⡡⠪⡭⠃⡤⢤⡄⡰⠋⠀⠀⠀⣀⡠⠴⠊⠁⠀⠀⠀⢠⠇
⠀⠀⢠⢿⣠⠟⠓⠛⠉⠛⡟⠛⠛⠛⠛⠒⠒⠚⠉⠉⠁⠀⠀⠀⠀⠀⢀⡴⠋⠀
⠀⠀⠈⠛⢯⣀⣀⣀⡤⠤⠤⠤⢤⣤⣀⣀⣀⣀⣀⣀⣀⣤⠤⠴⠒⠋⠁⠀⠀⠀

#################################

"""

import numpy as np
import scipy.interpolate
import CoolProp.CoolProp as CP
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq
from scipy.optimize import root

from numba import njit

#### CATNAP functions
from RXPI_CATNAP_Fluids import mdot_spi_hem_nhne, mdot_vapor_orifice, nozzle, Injector_obj

from RXPI_CATNAP_Combustion import SolvePC, CombustionPerformance, ChamberTransport, TPRhoStag, Props_obj, Transport_obj

from RXPI_CATNAP_Regen import DittusB, Gneilinski, Etafin, bartz_hg, Resistances, DeltaP, RegenGeom, Regen_obj

from rocketcea.cea_obj_w_units import CEA_Obj

plt.style.use('dark_background')

######### FEED SYSTEM INPUTS ############

Ethoxide = Props_obj('N2O','ETHANOL','N2O','ETHANOL')

Pintle = Injector_obj(0.65,0.55,0.65,25,20,20,2.533e-3,1.240e-3,0.480e-3,Ethoxide)

mdotcoolinput = 1.81



simtime = 25 #seconds

numsteps = 150

dt = simtime/numsteps

timevec = np.linspace(0,simtime,numsteps)

m1 = 53.3438 #kg (approx)

dP_piston_psi = 15 

T_init = 290 #K

massratio_target = 3.2

mdot_target = 3.40194 #kg/s

###########################################

######## PROPULSION INPUTS ################

C = CEA_Obj(oxName='N2O', fuelName='ETHANOL')

plotchannel = True




coolant = Ethoxide.fuel

Engine_name = 'Altair'

Tc_init = 170 #K

numpts = 20000


Dt = 52.77 #throat diameter in mm
De = 111.04 #exit diameter in mm

eps = (De**2)/(Dt**2)


### THESE DIMENSIONS ARE IN INCHES 

_in2m = 1 / 39.3700787

Lnozzle = 4.19*_in2m
Lcon1 = 0.367*_in2m
Lcon2 = 2.912*_in2m
Lcham = 4.841*_in2m

Le = Lnozzle + Lcon1 + Lcon2 + Lcham

expansionangle = 15 #degrees
    
Rc1 = 0.589*_in2m #radius of con1 arc
Rc2 = 4.678*_in2m #radius of con2 arc
    
Rcham = 2.17*_in2m #combustion chamber radius
    
Rexit = 2.13*_in2m

#### GEOM ARRAY

geominches = [Lnozzle,Lcon1,Lcon2,Lcham,Rexit,Rc1,Rc2,Rcham]

geom = np.array(geominches)

#### RPA/CEA

R_spec = 0.3597*1000 #J/kg*K

gamma = 1.226

Tc = 2641.5 #K

A_t = 0.25*pi*((52.01/1000)**2) #throat area in m^2

A_e = 0.25*pi*((108.29/1000)**2)

P_amb = 14.7*6895*0.8 # Ambient pressure

Rthroat = Dt/(2*1000) #throat radius in m


def R(z):
    '''
    Defines 2 arcs for the converging section, and a conical diverging section.
    Radius of curvature at throat is a separate parameter for the Bartz input

    Parameters
    ----------
    z : Axial coordinate (inches to integrate w/ NX CAD).

    Returns
    -------
    R : Thrust chamber contour radius in inches.

    '''
   
    
    if z >=0 and z < Lnozzle:
        R = Rexit - tan((pi*expansionangle/180))*z
    elif z >= Lnozzle and z < Lnozzle + Lcon1:
        
        z = z - Lnozzle
        
        R = (Rexit - tan((pi*expansionangle/180))*Lnozzle +
             + Rc1 - np.sqrt(Rc1**2 - z**2))
    elif z >= Lnozzle + Lcon1 and z <= Lnozzle + Lcon1 + Lcon2:
        
        z = z - (Lnozzle + Lcon1)
        
        zi = Lcon1
        
        R = (Rexit - tan((pi*expansionangle/180))*Lnozzle +
             + Rc1 - np.sqrt(Rc1**2 - zi**2)
             + np.sqrt(Rc2**2 - (z - Lcon2)**2)
             - np.sqrt(Rc2**2 - Lcon2**2))
    elif z>= Lnozzle + Lcon1 + Lcon2 and z < Lnozzle + Lcon1 + Lcon2 + Lcham + 1e-2:
        z = Lcon2
        zi = Lcon1
        Rcham = (Rexit - tan((pi*expansionangle/180))*Lnozzle +
             + Rc1 - np.sqrt(Rc1**2 - zi**2)
             + np.sqrt(Rc2**2 - (z - Lcon2)**2)
             - np.sqrt(Rc2**2 - Lcon2**2))
        R = Rcham
    else:
        
        raise ValueError('Z value given is outside nozzle range!')
        
    return R


Altair = Regen_obj(1.5e-3,1e-3,1.5e-3,R,Rthroat,0.0254,90,Ethoxide.fuel,237,15e-6,80,Le,35)

######################################


z = np.linspace(0,Le,numpts)

zout = np.zeros([numpts])

for i in range(numpts):
    zinput = z[i]

    zout[i] = R(zinput)
    


# Contour plot


plt.figure(figsize=(8,4))

nozzlecolor = 'steelblue'

contournumber = 7

gridnumber = 15

plt.plot(z,zout,color=nozzlecolor)
plt.plot(z,-zout,color=nozzlecolor)

plt.plot([0,0],[zout[0],-zout[0]],color=nozzlecolor)
plt.plot([Le,Le],[zout[numpts-2],-zout[numpts-2]],color=nozzlecolor)

for i in range(contournumber):
    
    a = i/contournumber
    plt.plot(z,zout*a,color=nozzlecolor)
    plt.plot(z,-zout*a,color=nozzlecolor)
    
for i in range(gridnumber):
    grdx = round((i/gridnumber)*numpts)
    grdz = (i/gridnumber)*Le
    plt.plot([grdz,grdz],[zout[grdx],-zout[grdx]],color=nozzlecolor)




plt.axis('equal')
plt.title(f'{Engine_name} Nozzle Contour (inches)')

plt.show()




#### REDEFINING z ARRAY for metric

z = z/39.3700787


Channelwise_Engine_Length = 0.3
numpts_axial = 5000  # number of interpolation points
dz = Channelwise_Engine_Length / numpts_axial
epsilon = 1e-3
plot = True


# THRUST CHAMBER MATERIAL PROPERTIES
Chamber_k = 350  # Chamber material thermal conductivity in W/(m*K)



# # Gas properties throughout TCA

# T_known = np.array([1086.7499, 1975.7020, 2224.1329, 2226.3701])
# cp_known = np.array([2.9881e3, 2.7191e3, 2.8082e3, 2.8090e3])
# rho_known = np.array([0.1681, 0.9754, 2.6424, 2.6480])
# mu_known = np.array([4.354e-5, 6.587e-5, 7.162e-5, 7.167e-5])
# k_known = np.array([0.201, 0.3214, 0.3568, 0.3841])
# Pr_known = np.array([0.5153, 0.5390, 0.5242, 0.5241])
# Mach_known = np.array([2.8585,1.0,0.0898,0.000])

# RPApts = np.array([0.0,Lnozzle/39.3700787,(Lnozzle + Lcon1 + Lcon2)/39.3700787,(Lnozzle + Lcon1 + Lcon2 + Lcham)/39.3700787])

# Tgas = PchipInterpolator(RPApts,T_known)
# cpgas = PchipInterpolator(RPApts,cp_known)
# rhogas = PchipInterpolator(RPApts,rho_known)
# mugas = PchipInterpolator(RPApts,mu_known)
# kgas = PchipInterpolator(RPApts,k_known)
# Prgas = PchipInterpolator(RPApts,Pr_known)
# Machgas = PchipInterpolator(RPApts,Mach_known)


# CP_gas_array = cpgas(z)  # Effective specific heat in J/(kg*K)
# Gamma_gas = 1.249  # Exhaust specific heat ratio
# Temperature_gas_array = Tgas(z)  # Temperature in K
# Thermal_conductivity_gas_array = kgas(z)  # Exhaust effective thermal conductivity in W/(m*k)
# Viscosity_gas_array = mugas(z)  # Viscosity in kg/(m*s) aka Pa*s
# Prantl_gas_array = Prgas(z) # Exhaust effective Prantl number
# Density_gas_array = rhogas(z)  # exhaust density in kg/m^3
# Mach_gas_array = Machgas(z) # Mach number, dimensionless


# plt.figure(figsize=(10, 6))
# plt.plot(z, CP_gas_array, label='Cp (J/kg*K)', color='blue', linewidth=2)
# plt.plot(z, Temperature_gas_array, label='Temperature (K)', color='red', linewidth=2)
# ### Cp and temp plots
# plt.xlabel('Axial position (m)', fontsize=14)
# plt.ylabel('Cp / Temperature', fontsize=14)
# plt.title('Specific Heat and Temperature Along Engine Axis', fontsize=16)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=12)
# plt.tight_layout()
# plt.show()

# wid2 = 3
# # Other property plots
# plt.figure(figsize=(10, 6))
# plt.plot(z, Thermal_conductivity_gas_array, label='Thermal Conductivity (W/m*K)', color='green', linewidth=wid2)
# plt.plot(z, Viscosity_gas_array, label='Viscosity (Pa*s)', color='orange', linewidth=wid2)
# plt.plot(z, Prantl_gas_array, label='Prandtl Number', color='purple', linewidth=wid2)
# plt.plot(z, Density_gas_array, label='Density (kg/m³)', color='royalblue', linewidth=wid2)
# plt.plot(z, Mach_gas_array, label='Mach Number', color='salmon', linewidth=wid2)


# plt.xlabel('Axial position (m)', fontsize=14)
# plt.ylabel('Gas Properties', fontsize=14)
# plt.title('Gas Properties Along Engine Axis', fontsize=16)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='best', fontsize=12)
# plt.tight_layout()
# plt.show()




Pc_arr       = []
P2_arr       = []
T2_arr       = []
x2_arr       = []
F_arr        = []
Isp_arr      = []
mdot_arr     = []







def timestep(T1,mdotox,mdotf,x1,dt,m1,v1,Props_obj):
    '''
    Function that evolves the thermodynamic state of an ADIABATIC tank over a time dt.
    Includes piston work, enthalpy loss by mass flow, and specific volume change.

    Parameters
    ----------
    T1 : Saturation temperature at time t (K).
    mdotox : Pressurant mass flow rate at time t (kg/s).
    mdotf : Fuel mass flow rate at time t (kg/s).
    x1 : Pressurant vapor mass fraction at time t.
    dt : Interval timestep.
    m1 : Total mass of pressurant within tank at time t (kg).
    v1 : Specific volume of pressurant at time t (m^3/kg).

    Returns
    -------
    T2 : Saturation temperature at time t + dt (K).
    x2 : Pressurant vapor mass fraction at time t + dt.
    m2 : Total mass of pressurant within tank at time t + dt (kg).
    v2 : Specific volume of pressurant within tank at time t + dt (m^3/kg).
    
    phase : STRING. Denotes the phase of the pressurant within the tank.
    -phase = 'saturated' during phase transition, phase = 'vapor' after boiling completes.

    '''
    
    Pressurant = Props_obj.ox

    Fuel = Props_obj.fuel

    if 0.0 <= x1 and x1 <= 0.999:
        
        print('Phase: Saturated')
        
        phase = 'saturated'

        uf1 = CP.PropsSI('U','T',T1,'Q',0,Pressurant)
        ug1 = CP.PropsSI('U','T',T1,'Q',1,Pressurant)
        u1 = uf1 + x1*(ug1 - uf1)
        
        hf1 = CP.PropsSI('H','T',T1,'Q',0,Pressurant)
        
        vf1 = 1/(CP.PropsSI('D','T',T1,'Q',0,Pressurant))
        vg1 = 1/CP.PropsSI('D','T',T1,'Q',1,Pressurant)
        v1 = vf1 + x1*(vg1 - vf1)
        
        P1 = CP.PropsSI('P','T',T1,'Q',0,Pressurant)
        
        vfuel = 1/(CP.PropsSI('D','T',T1,'P',P1 - dP_piston_psi*6894.75729,Fuel))
        Wp = -P1*mdotf*vfuel*dt #work done by piston during dt
        
        u2 = (Wp - mdotox*dt*hf1 + m1*u1)/(m1 - mdotox*dt) # First law / control volume balance
        
        v2 = (v1*m1 + mdotf*vfuel*dt)/(m1 - mdotox*dt)
        
        # x = np.linspace(200,CP.PropsSI('Tcrit',Pressurant) - 2,200)
        # plt.figure()
        # plt.plot(x,rootT2(x,v2,u2))
        # plt.show()
        
        T2 = brentq(rootT2, T1 - 20, CP.PropsSI('Tcrit',Pressurant) - 2, args=(v2, u2, Pressurant))
        
        
    
        vf2 = 1/(CP.PropsSI('D','T',T2,'Q',0,Pressurant))
        vg2 = 1/(CP.PropsSI('D','T',T2,'Q',1,Pressurant))
        
        x2 = (v2 - vf2)/(vg2 - vf2)
        
        m2 = m1 - mdotox*dt
        print('------')
        print()
        print('T2:',T2,'(K)')
        print('Tank Pressure:',P1/6894.75729,'psi')
        print('Vapor Mass Fraction',x2*100,'%')
        print()
        
    elif x1 >= 0.999:
        print('Phase: Vapor')
        
        phase = 'vapor'
        
        x2 = 1.0
    
        
        D1 = 1/v1
        
        u1 = CP.PropsSI('U','T',T1,'D',D1,Pressurant)
        
        hg1 = CP.PropsSI('H','T',T1,'D',D1,Pressurant)
        
        v1 = 1/(CP.PropsSI('D','T',T1,'D',D1,Pressurant))
        
        P1 = CP.PropsSI('P','T',T1,'D',D1,Pressurant)
        
        vfuel = 1/(CP.PropsSI('D','T',T1,'P',P1 - dP_piston_psi*6894.75729,Fuel))
        
        Wp = -P1*mdotf*vfuel*dt # work done by piston during dt
        
        u2 = (Wp - mdotox*dt*hg1 + m1*u1)/(m1 - mdotox*dt) # First law / control volume balance
        
        v2 = (v1*m1 + mdotf*vfuel*dt)/(m1 - mdotox*dt)
        
        m2 = m1 - mdotox*dt
        
        D2 = 1/v2
        
        T2 = CP.PropsSI('T','U',u2,'D',D2,Pressurant)
        
        D2 = 1/v2
        
        P2 = CP.PropsSI('P','T',T2,'D',D2,Pressurant)
        
        
    
    return T2,x2,m2,v2,phase
    

    
    
    
def rootT2(T2,v2,u2,Pressurant):
    '''
    Residual to solve for specific internal energy within 
    the tank after the thermodynamic process completes over 
    interval dt

    '''

    uf = CP.PropsSI('U','T',T2,'Q',0,Pressurant)
    ug = CP.PropsSI('U','T',T2,'Q',1,Pressurant)
    vf = 1/(CP.PropsSI('D','T',T2,'Q',0,Pressurant))
    vg = 1/(CP.PropsSI('D','T',T2,'Q',1,Pressurant))
    
    x2 = (v2 - vf)/(vg - vf)
    
    return u2 - uf - x2*(ug - uf)







######################################


Pc_init = 340*6895 #psi

def CATNAP(Tinit, mdotox, mdotf, mdotfilm, x1, dt, m1,
           Regen_obj, Injector_obj, Props_obj,
           regen_times=None,         
           Tcool_init=290,
           regen=True, plot=True):
    
    mdot_coolant = mdotf + mdotfilm

    Pressurant = Props_obj.ox

    Fuel = Props_obj.fuel
    
    T1 = Tinit
    
    mdot_total = mdotox + mdotf + mdotfilm
    
    MR = mdotox/(mdotf + mdotfilm)

    Rthroat = Regen_obj.Rthroat

    At = pi*(Rthroat**2)
    
    Pc = SolvePC(mdot_total, MR, At, Pc_init, Props_obj)

    vginit = 1/(CP.PropsSI('D','Q',1,'T',T1,Pressurant))
    
    vfinit = 1/(CP.PropsSI('D','Q',0,'T',T1,Pressurant))
    
    v1 = vfinit + x1*(vginit - vfinit)

    Pc_arr = []
    P2_arr = []
    T2_arr = []
    x2_arr = []
    F_arr  = []
    Isp_arr = []
    mdot_arr = []
    massratio_arr = []


    regen_index_map = {}
    if regen_times is not None:
        for t in regen_times:
            idx = int(round(t / dt))
            if 0 <= idx < numsteps:
                regen_index_map[idx] = t

    numz     = Regen_obj.numpts_z
    n_snap   = len(regen_index_map)
    regen_snap_times = []
    Tcool_3d = np.zeros((n_snap, numz))
    Pcool_3d = np.zeros((n_snap, numz))
    hg_3d    = np.zeros((n_snap, numz))
    Twall_3d = np.zeros((n_snap, numz))
    Qflux_3d = np.zeros((n_snap, numz))
    snap_count = 0

    transport_3d = []

    tempsC_3d = []
    
    for i in range(numsteps):
            
            time = (i/numsteps)*simtime
            print(f'Iteration Results (t = {time:.4f}):')
            print(f'Iteration # {i}')
            print()
            
            #######################
            
            Vapinit = CP.PropsSI('P','T',T_init,'Q',0,Injector_obj.ox)

            D_fuel_init = CP.PropsSI('D','T',T_init,'P',Vapinit,Injector_obj.fuel)


            T2,x2,m2,v2,phase = timestep(T1,mdotox,mdotf,x1,dt,m1,v1,Props_obj)
        
            if phase == 'saturated':
                
                P2 = CP.PropsSI('P','T',T2,'Q',0,Pressurant)
                
                mdotox = Injector_obj.mdot_ox_nhne(P2,T2,Pc)
                
            elif phase == 'vapor':
                
                D2 = 1/v2
                
                P2 = CP.PropsSI('P','T',T2,'D',D2,Pressurant)
                
                mdotox = Injector_obj.mdot_vapor_orifice(P2,T2,Pc)
            
            
            P2f = P2 - (dP_piston_psi*6894.75729)

            dPchannels = Regen_obj.dP_channel_Approx(Tcool_init,P2f,mdot_coolant)

            P2inj = P2f - dPchannels

            mdotf = Injector_obj.mdot_fuel(P2inj,T2,Pc)

            mdotfilm = Injector_obj.mdot_film(P2inj,T2,Pc)

            if P2inj < Pc*1.1:
                raise Warning('Injector Stiffness Below 10%! Instability Probable')
            
            mdot_total = mdotox + mdotf + mdotfilm
            
            massratio = mdotox/(mdotf + mdotfilm)

            radthroat = Regen_obj.Rthroat

            Athroat = pi*(radthroat**2)
            
            Pc = SolvePC(mdot_total, massratio, Athroat, Pc, Props_obj)

            Cf, F, Isp, cstar = CombustionPerformance(mdot_total,massratio,Athroat,
                                                    Pc,P_amb,eps,Props_obj)               


            if i in regen_index_map and mdot_coolant is not None:
                Trn = Transport_obj(mdot_total, massratio, Athroat, Props_obj, geom, eps, Pc)

                Pcool_init = P2f

                Tc_arr, Pc_snap, hg_arr, Tw_arr, Qf_arr = Regen_obj.SOLVE_REGEN(mdot_coolant, Tcool_init, Pcool_init, Trn)
                
                
                Trn = Transport_obj(mdot_total, massratio, Athroat, Props_obj, geom, eps, Pc)
                
                
                Tc_arr, Pc_snap, hg_arr, Tw_arr, Qf_arr = \
                    Regen_obj.SOLVE_REGEN(mdot_coolant, Tcool_init, Pcool_init, Trn)
                
                
                z_arr = Regen_obj.z_array
                Cp_t, mu_t, k_t, Pr_t, gamma_t = Trn.Chambertransport()
                MachF = lambda z: Trn.Mach(z, Regen_obj.R)
                trans_snap = np.array([[float(Cp_t(z)), float(mu_t(z)), float(k_t(z)),
                                        float(Pr_t(z)), float(gamma_t(z)), float(MachF(z))]
                                    for z in z_arr])  
                transport_3d.append(trans_snap.tolist())

                TempsC, _, _, _ = Trn.TPRhostag()
                temps_snap = np.array([[float(Tr), float(T0)]
                                    for Tr, T0 in [TempsC(z, MachF) for z in z_arr]])
                tempsC_3d.append(temps_snap.tolist())
    


                Tcool_3d[snap_count] = Tc_arr
                Pcool_3d[snap_count] = Pc_snap
                hg_3d[snap_count]    = hg_arr
                Twall_3d[snap_count] = Tw_arr
                Qflux_3d[snap_count] = Qf_arr
                regen_snap_times.append(regen_index_map[i])
                snap_count += 1
            

            Trn = Transport_obj(mdot_total,massratio,Athroat,Props_obj,geom,eps,Pc_init)

            
            
            Pc_arr.append(Pc)
            P2_arr.append(P2)
            T2_arr.append(T2)
            x2_arr.append(x2)
            F_arr.append(F)
            Isp_arr.append(Isp)
            mdot_arr.append(mdot_total)
            massratio_arr.append(massratio)
            
            
            
            T1 = T2
            x1 = x2
            m1 = m2
            v1 = v2
            
            print('------')
            print()
            print(f'Thrust:           {F/1000:.3f} kN')
            print(f'O/F Ratio:        {massratio:.3f}')
            print(f'Chamber Pressure: {Pc/6894.75729:.2f} psi')
            print(f'Isp:              {Isp:.2f} s')
            print(f'Tank Pressure:    {P2/6894.75729:.2f} psi')
            print(f'Mass Flow Rate:   {mdot_total:.2f} kg/s')
            print('------')

    ####### Mass Flow Rate and Thrust Data

    mdot_oxarr = np.zeros(numsteps)

    mdot_fuelarr = np.zeros(numsteps)

    for i in range(numsteps):
        factor = massratio_arr[i]/(1 + massratio_arr[i])
        mdot_oxarr[i] = factor*mdot_arr[i]
        mdot_fuelarr[i] = (1-factor)*mdot_arr[i]
        
        
    Imp = 0    
        
    for i in range(numsteps - 1):
        trap = 0.5*(F_arr[i] + F_arr[i+1])*dt
        
        Imp += trap


        


    print('===== SIMULATION RESULTS=====')

    print('Total Impulse (N*s):',Imp)



    Pc_arr        = np.asarray(Pc_arr)
    P2_arr        = np.asarray(P2_arr)
    T2_arr        = np.asarray(T2_arr)
    x2_arr        = np.asarray(x2_arr)
    F_arr         = np.asarray(F_arr)
    Isp_arr       = np.asarray(Isp_arr)
    mdot_arr      = np.asarray(mdot_arr)
    massratio_arr = np.asarray(massratio_arr)

    if plot==True:
        
        ############# PLOTS #################

        #################################### TEMPS AND PRESSURES


        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title('Feed System Temperatures and Pressures', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        ax1.plot(timevec, P2_arr/6894.75729, label='Tank Pressure (psi)', color='mediumblue', linewidth=2)
        ax1.plot(timevec, Pc_arr/6894.75729, label='Chamber Pressure (psi)', color='mediumpurple', linewidth=2)
        ax1.set_ylabel('Pressure (psi)', fontsize=14)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Time (s)', fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(timevec, T2_arr, label='Saturation Temperature (K)', color='red', linewidth=2)
        ax2.set_ylabel('Temperature (K)', fontsize=14)

        ax1.legend(
            ax1.get_legend_handles_labels()[0] +
            ax2.get_legend_handles_labels()[0],
            ax1.get_legend_handles_labels()[1] +
            ax2.get_legend_handles_labels()[1],
            loc='best'
        )

        plt.show()

        ################## ISP AND THRUST


        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title('Engine Thrust and Specific Impulse', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        ax1.plot(timevec, Isp_arr, label='Specific Impulse (s)', color='orchid', linewidth=2)
        ax1.set_ylabel('Specific Impulse (s)', fontsize=14)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Time (s)', fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(timevec, F_arr/1000, label='Thrust (kN)', color='aquamarine', linewidth=2)
        ax2.set_ylabel('Thrust (kN)', fontsize=14)

        ax1.legend(
            ax1.get_legend_handles_labels()[0] +
            ax2.get_legend_handles_labels()[0],
            ax1.get_legend_handles_labels()[1] +
            ax2.get_legend_handles_labels()[1],
            loc='best'
        )

        plt.show()

        ################## MASS FLOWS

        fig, ax1 = plt.subplots(figsize=(12, 9))
        plt.title('Feed System Mass Flow Rates', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        ax1.plot(timevec, mdot_oxarr, label=f'{Pressurant} Mass Flow Rate', color='lightsalmon', linewidth=2)
        ax1.plot(timevec, mdot_fuelarr, label=f'{Fuel} Mass Flow Rate', color='cornflowerblue', linewidth=2)
        ax1.set_ylabel('Mass Flow Rate (kg/s)', fontsize=14)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Time (s)', fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(timevec, massratio_arr, label='O/F Ratio', color='gainsboro', linewidth=2, linestyle='dashdot')
        ax2.set_ylabel('Mass Ratio', fontsize=14)

        ax1.legend(
            ax1.get_legend_handles_labels()[0] +
            ax2.get_legend_handles_labels()[0],
            ax1.get_legend_handles_labels()[1] +
            ax2.get_legend_handles_labels()[1],
            loc='best'
        )

        plt.show()
        
    regen_times_arr = np.array(regen_snap_times)


    return (np.asarray(Pc_arr), np.asarray(P2_arr), np.asarray(T2_arr),
                np.asarray(x2_arr), np.asarray(F_arr),  np.asarray(Isp_arr),
                np.asarray(mdot_arr), np.asarray(massratio_arr),
                regen_times_arr, Tcool_3d, Pcool_3d, hg_3d, Twall_3d, Qflux_3d,transport_3d,tempsC_3d)
            




(Pc_arr, P2_arr, T2_arr, x2_arr, F_arr, Isp_arr, mdot_arr, massratio_arr,
 regen_times, Tcool_3d, Pcool_3d, hg_3d, Twall_3d, Qflux_3d,transport_3d,tempsC_3d) = CATNAP(290,2.548,0.784,0.118,0.01,dt,54,Altair,Pintle,Ethoxide,regen_times=np.linspace(0.5,18,7),         
                                                                                                        Tcool_init=290, 
                                                                                                        regen=True, plot=True)


    
#############################


##################################################

# dashboard stuff


import json
import os
import webbrowser

regen_results = {
    'timevec':       timevec.tolist(),
    'Pc_arr':        Pc_arr.tolist(),
    'P2_arr':        P2_arr.tolist(),
    'T2_arr':        T2_arr.tolist(),
    'x2_arr':        x2_arr.tolist(),
    'F_arr':         F_arr.tolist(),
    'Isp_arr':       Isp_arr.tolist(),
    'mdot_arr':      mdot_arr.tolist(),
    'massratio_arr': massratio_arr.tolist(),
    'z_array':       Altair.z_array.tolist(),
    'regen_times':   regen_times.tolist(),
    'Tcool_3d':      Tcool_3d.tolist(),
    'Pcool_3d':      Pcool_3d.tolist(),
    'hg_3d':         hg_3d.tolist(),
    'Twall_3d':      Twall_3d.tolist(),
    'Qflux_3d':      Qflux_3d.tolist(),
    'transport_3d': transport_3d,
    'tempsC_3d': tempsC_3d,
    'nozzle_R': [float(R(zi)) for zi in Altair.z_array]
}

# Inject data into dashboard and write self-contained HTML

with open('catnap_dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

inject = '<script>var AUTOLOAD = ' + json.dumps(regen_results) + ';</script>'
html = html.replace('</head>', inject + '\n</head>')



output_path = os.path.abspath('catnap_results.html')

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Dashboard exported to {output_path}')
webbrowser.open('file:///' + output_path)









