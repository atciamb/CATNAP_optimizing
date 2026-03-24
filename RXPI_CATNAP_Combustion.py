# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:29:53 2026

@author: Maximilian Slavik
"""

import numpy as np
import scipy.interpolate
import CoolProp.CoolProp as CP
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq
from scipy.optimize import root


from rocketcea.cea_obj_w_units import CEA_Obj


class Props_obj:
    def __init__(self,oxname,fuelname,coolpropox,coolpropfuel):
        self.ox = coolpropox
        self.fuel = coolpropfuel
        self.oxname = oxname
        self.fuelname = fuelname
        
        self.C = CEA_Obj(oxName=self.oxname,
                 fuelName=self.fuelname,
                 pressure_units='Pa',
                 cstar_units='m/s',
                 isp_units='sec',
                 temperature_units='K',
                 sonic_velocity_units='m/s',
                 enthalpy_units='J/kg',
                 density_units='kg/m^3',
                 specific_heat_units='J/kg-K',    
                 viscosity_units='poise',        
                 thermal_cond_units='mcal/cm-K-s'  
                 )
        







def SolvePC(mdot_total,MR,At,Pc_init,Props_obj):
    '''
    
    Solves Pc using a brentq root finder
    
    Parameters
    ----------
    mdot_total : float
        Total propellant mass flow rate (kg/s).
    MR : float
        Oxidizer-to-fuel mass ratio (O/F).
    At : float
        Nozzle throat area (m^2).
    Fuel : str
        Fuel name as recognized by RocketCEA (e.g. 'ETHANOL').
    Pc_init : float
        Chamber pressure at time t - dt (or initial Pc for t=0) (Pa). Used to bracket the root.
    Ox : str, optional
        Oxidizer name as recognized by RocketCEA. Default is 'N2O'.

    Returns
    -------
    Pc : float
        Converged chamber pressure (Pa).
    '''
    
    targ = 10 #psi root finder shift

    C = Props_obj.C

    def Res(Pc):     
        cstar = C.get_Cstar(Pc,MR)
        
        term2 = cstar*mdot_total/At
        
        Res = Pc - term2
        # print(f"  Pc={Pc:.1f} Pa, cstar={cstar:.2f} m/s, term2={term2:.1f}")
        return Res
        
    Pc = brentq(Res,Pc_init*0.001,Pc_init*3)
    
    
    return Pc



def CombustionPerformance(mdot_total,MR,At,Pc,Pamb,eps,Props_obj):
    """
    Computes nozzle thrust coefficient, thrust, and specific impulse
    using RocketCEA for a given propellant combination and operating condition.
    
    Parameters
    ----------
    mdot_total : float
        Total propellant mass flow rate (kg/s).
    MR : float
        Oxidizer-to-fuel mass ratio (O/F).
    At : float
        Nozzle throat area (m^2).
    Fuel : str
        Fuel name as recognized by RocketCEA (e.g. 'ETHANOL').
    Pc : float
        Chamber pressure (Pa).
    Pamb : float
        Ambient pressure (Pa).
    eps : float
        Nozzle area expansion ratio (Ae/At).
    Ox : str, optional
        Oxidizer name as recognized by RocketCEA. Default is 'N2O'.
    
    Returns
    -------
    Cf : float
        Ambient thrust coefficient (dimensionless).
    Thrust : float
        Engine thrust (N).
    Isp : float
        Specific impulse (s).
        
    """
    
    C = Props_obj.C
    
    Cfvac, Cf, mode = C.get_PambCf(Pamb, Pc, MR=MR, eps=eps)
    
    mode_type = mode.split('(')[0].strip()

    if mode_type == 'Separated':
        print("Separated Nozzle Flow Probable")
        
    Thrust = Cf*Pc*At
    
    Isp = Thrust/(mdot_total*9.81)

    cstar = C.get_Cstar(Pc,MR)
    
    return Cf,Thrust,Isp,cstar

    


def ChamberTransport(mdot_total,MR,Pc,geom,eps,Props_obj):
    """
    Retrieves CEA transport properties at the chamber, throat, and exit,
    and returns PCHIP interpolators for each property along the engine axis.
    
    Parameters
    ----------
    mdot_total : float
        Total propellant mass flow rate (kg/s). Reserved for future use.
    MR : float
        Oxidizer-to-fuel mass ratio (O/F).
    Pc : float
        Chamber pressure (Pa).
    geom : engine geometry parameters (m) - see main script for array structure
    eps : float
        Nozzle area expansion ratio (Ae/At).
    Fuel : str
        Fuel name as recognized by RocketCEA (e.g. 'ETHANOL').
    Ox : str, optional
        Oxidizer name as recognized by RocketCEA. Default is 'N2O'.
    
    Returns
    -------
    Cptransport : PchipInterpolator
        Specific heat Cp along engine axis (J/kg-K).
    viscositytransport : PchipInterpolator
        Dynamic viscosity along engine axis (poise).
    thermalcondtransport : PchipInterpolator
        Thermal conductivity along engine axis (W/m-K).
    prantltransport : PchipInterpolator
        Prandtl number along engine axis (dimensionless).
    gammatransport : PchipInterpolator
        Specific heat ratio gamma along engine axis (dimensionless).
    """


    C = Props_obj.C
    
    _, gammacham = C.get_Chamber_MolWt_gamma(Pc,MR,eps)
    
    _, gammathroat = C.get_Throat_MolWt_gamma(Pc,MR,eps)
    
    _, gammaexit = C.get_exit_MolWt_gamma(Pc,MR,eps)
    
    ###
    
    Cpcham, mucham, kcham, prantlcham = C.get_Chamber_Transport(Pc,MR,eps)
    
    Cpthroat, muthroat, kthroat, prantlthroat = C.get_Throat_Transport(Pc,MR,eps)
    
    Cpexit, muexit, kexit, prantlexit = C.get_Exit_Transport(Pc,MR,eps)
    
    # geom = [Lnozzle,Lcon1,Lcon2,Lcham,Rexit,Rc1,Rc2,Rcham]
    
    xthroat = geom[0]
    
    xcham = geom[0] + geom[1] + geom[2]
    
    xinj = geom[0] + geom[1] + geom[2] + geom[3]
    
    xpts = [0,xthroat,xcham,xinj]
    
    # interpolated pchip functions
    
    Cptransport = PchipInterpolator(xpts,[Cpexit,Cpthroat,Cpcham,Cpcham])
    
    viscositytransport = PchipInterpolator(xpts,0.1*np.array([muexit,muthroat,mucham,mucham])) # 0.1 to convert to Pa*s
    
    thermalcondtransport = PchipInterpolator(xpts,0.4184*np.array([kexit,kthroat,kcham,kcham]))
    
    prantltransport = PchipInterpolator(xpts,[prantlexit,prantlthroat,prantlcham,prantlcham])
    
    gammatransport = PchipInterpolator(xpts,[gammaexit,gammathroat,gammacham,gammacham])
    
    return Cptransport, viscositytransport, thermalcondtransport, prantltransport, gammatransport









def TPRhoStag(mdot_total,MR,Pc,Mach,geom,eps,Props_obj):
    """
    Computes isentropic stagnation and recovery properties along the engine axis,
    returning callable functions for temperature, pressure, and density.
    
    Parameters
    ----------
    mdot_total : float
        Total propellant mass flow rate (kg/s).
    MR : float
        Oxidizer-to-fuel mass ratio (O/F).
    Pc : float
        Chamber pressure (Pa).
    Mach : callable
        Function of axial position z returning local Mach number.
    geom : list or array
        Engine geometry parameters
    eps : float
        Nozzle area expansion ratio (Ae/At).
    Fuel : str
        Fuel name as recognized by RocketCEA (e.g. 'ETHANOL').
    Ox : str, optional
        Oxidizer name as recognized by RocketCEA. Default is 'N2O'.
    
    Returns
    -------
    TempsC : callable
        Function(z, Mach) returning (T_r, T0):
            T_r : Adiabatic wall / recovery temperature (K).
            
            NOTE: CATNAP assumes a completely turbulent boundary layer along the hotwall
            
            T0  : Stagnation temperature (K).
    PressuresC : callable
        Function(z, Mach) returning P0, the local stagnation pressure (Pa).
    RhosC : callable
        Function(z, Mach) returning rho0, the local stagnation density (kg/m^3).
    """
    C = Props_obj.C


    
    Tstag = C.get_Tcomb(Pc,MR)

    rhostag = C.get_Chamber_Density(Pc,MR,eps)
    
    
    _, _, _, prantltransport, gammatransport = ChamberTransport(mdot_total,MR,Pc,geom,eps,Props_obj)
    
    def TempsC(z, Mach):
        r = prantltransport(z)**0.333
        gamma = gammatransport(z)
        denom = 1 + (gamma - 1) / 2 * Mach(z)**2
        T0 = Tstag     
        T_static = Tstag / denom
        T_r = T_static * (1 + r * (gamma - 1) / 2 * Mach(z)**2)
        return T_r, T0

    def RhosC(z, Mach):
        gamma    = gammatransport(z)
        rho_local = rhostag * (1 + (gamma - 1) / 2 * Mach(z)**2)**(-1 / (gamma - 1))
        return rho_local

    def PressuresC(z, Mach):
        gamma   = gammatransport(z)
        P_local = Pc * (1 + (gamma - 1) / 2 * Mach(z)**2)**(-gamma / (gamma - 1))
        return P_local
    
    
    return TempsC,Tstag,PressuresC,RhosC




def MachArea(z, R, geom, gamma):

    Lnozzle = geom[0]

    # gamma = gammatransport(Lnozzle) # gamma from throat assumed for entire nozzle (after combustion, approximate)

    def area_mach_residual(M, AR, gamma):
        return (1/M) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * M**2)) \
               **((gamma+1) / (2*(gamma-1))) - AR

    Rt      = R(Lnozzle)
    At      = pi*(Rt**2)


    A  = pi * R(z)**2
    AR = A / At

    if AR == 1.0 or abs(z - Lnozzle) < 1e-9:
        Mach = 1.0

    elif z < Lnozzle:  
        Mach = brentq(area_mach_residual, 1.0, 50.0, args=(AR, gamma))

    else:         
        Mach = brentq(area_mach_residual, 1e-6, 1.0, args=(AR, gamma))

    return Mach






class Transport_obj:
    def __init__(self,mdot_total,MR,Athroat,Props_obj,geom,eps,Pc_init):
        self.mdot_total = mdot_total
        self.MR = MR
        self.Athroat = Athroat
        self.geom = geom
        self.eps = eps
        self.ox = Props_obj.ox
        self.fuel = Props_obj.fuel
        self.Pc_init = Pc_init

        self.Props_obj = Props_obj

        self.Pc = SolvePC(self.mdot_total,self.MR,self.Athroat,self.Pc_init,self.Props_obj)

    def Tcomb(self):
        Tcomb = self.Props_obj.C.get_Tcomb(self.Pc,self.MR)

        return Tcomb


    def Combustionperformance(self,Pamb):
        
        Cf,Thrust,Isp,cstar = CombustionPerformance(self.mdot_total,self.MR,self.Athroat,self.Pc,Pamb,self.eps,self.Props_obj)
        
        return Cf,Thrust,Isp,cstar
    
    def getCstar(self):
        
        MR = self.MR

        cstar = self.Props_obj.C.get_Cstar(self.Pc,MR)

        return cstar

    def Chambertransport(self):


        Cptransport, viscositytransport, thermalcondtransport, prantltransport, gammatransport = ChamberTransport(self.mdot_total,self.MR,self.Pc,self.geom,self.eps,self.Props_obj)

        return Cptransport, viscositytransport, thermalcondtransport, prantltransport, gammatransport
        
    
    def TPRhostag(self):

        MachArea = self.Mach

        TempsC,Tstag,PressuresC,RhosC = TPRhoStag(self.mdot_total,self.MR,self.Pc,MachArea,self.geom,self.eps,self.Props_obj)

        return TempsC,Tstag,PressuresC,RhosC
    

    def Mach(self,z,R):

        geom = self.geom

        _, gamma = self.Props_obj.C.get_Throat_MolWt_gamma(self.Pc,self.MR,self.eps)
        
        mach = MachArea(z,R,self.geom,gamma)

        return mach






















