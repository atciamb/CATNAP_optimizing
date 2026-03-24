# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:28:24 2026

@author: Maximilian Slavik
"""



import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import math
from math import pi, tan
from scipy.optimize import brentq


################ NHNE Modeling Function ###############


##### Written by: Eli Macdonald 

##### Reviewed by: Maximilian Slavik, Andrew Ciambella
    
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
  
def mdot_spi_hem_nhne(P1, T1, P2, Cd, Ainj, fluid='N2O', k_override=None):
    # --- Upstream thermo (h1, s1)
    # h1 = CP.PropsSI("Hmass", "P", P1, "T", T1, fluid) #Modified to clamp liquid state
    h1 = CP.PropsSI("Hmass", "T", T1, "Q", 0, fluid)
    s1 = CP.PropsSI("Smass", "T", T1, "Q", 0, fluid)

    # --- Upstream liquid density for SPI (Evaluated at P and T for subcooled liquid!)
    rho1 = CP.PropsSI("Dmass", "Q", 0, "T", T1, fluid)

    # --- SPI model
    dP = max(P1 - P2, 0.0)
    mdot_spi = Cd * Ainj * math.sqrt(2.0 * rho1 * dP)

    # --- Check for Vapor Pressure (Flashing condition)
    Pv1 = CP.PropsSI("P", "T", T1, "Q", 0, fluid)  # Vapor pressure at inlet T
    
    # If vapor pressure is lower than downstream pressure, no flashing occurs.
    if Pv1 <= P2:
        return mdot_spi
        

    # --- HEM: isentropic to P2 (s2 = s1)
    h2 = CP.PropsSI("Hmass", "P", P2, "Smass", s1, fluid)

    # Saturation props at P2 (for quality + mixture density)
    hf   = CP.PropsSI("Hmass", "P", P2, "Q", 0, fluid)
    hg   = CP.PropsSI("Hmass", "P", P2, "Q", 1, fluid)
    rhof = CP.PropsSI("Dmass", "P", P2, "Q", 0, fluid)
    rhog = CP.PropsSI("Dmass", "P", P2, "Q", 1, fluid)

    # Exit quality from enthalpy
    tiny = 1e-12
    x2 = (h2 - hf) / max(hg - hf, tiny)
    x2 = clamp(x2, 0.0, 1.0)

    # Homogeneous mixture density at exit
    if x2 > 0:
        rho2 = 1.0 / (x2 / rhog + (1.0 - x2) / rhof)
    else:
        rho2 = rhof

    # HEM mass flow
    dh = max(h1 - h2, 0.0)
    mdot_hem = Cd * Ainj * rho2 * math.sqrt(2.0 * dh)

    # --- Dyre/NHNE k-factor
    if k_override is not None:
        k = float(k_override)
    else:
        # Dyer parameter
        k = math.sqrt(max(P1 - P2, 0.0) / (Pv1 - P2))

    # Blend weights
    w = (1.0 / (1.0 + k))

    mdot_nhne = (1-w) * mdot_spi + w * mdot_hem
    
    return mdot_nhne

###########################


class Injector_obj:
    def __init__(self,Cd_fuel,Cd_ox,Cd_film,numox,numfuel,numfilm,Dox,Dfuel,Dfilm,Props_obj):
        self.Cd_fuel = Cd_fuel
        self.Cd_ox = Cd_ox
        self.Cd_film = Cd_film
        self.numox = numox
        self.Aox = 0.25*numox*pi*(Dox**2)
        self.Afuel = 0.25*numfuel*pi*(Dfuel**2)
        self.Afilm = 0.25*numfilm*pi*(Dfilm**2)

        self.CdA_ox = Cd_ox*self.Aox
        self.CdA_fuel = Cd_fuel*self.Afuel
        self.CdA_film = Cd_film*self.Afilm

        self.fuel = Props_obj.fuel
        self.ox = Props_obj.ox




    def mdot_fuel(self,P1,T1,P2):
        
        rhof = CP.PropsSI('D','T',T1,'P',P1,self.fuel)

        mdot = self.CdA_fuel*(np.sqrt(2*rhof*(P1 - P2)))

        return mdot

    def mdot_film(self,P1,T1,P2):
        
        rhof = CP.PropsSI('D','T',T1,'P',P1,self.fuel)

        mdot = self.CdA_film*(np.sqrt(2*rhof*(P1 - P2)))

        return mdot
    
    def mdot_ox_nhne(self,P1,T1,P2):

        mdot = mdot_spi_hem_nhne(P1,T1,P2,self.Cd_ox,self.Aox,self.ox)

        return mdot
    
    def mdot_vapor_orifice(self,P1,T1,P2):

        Cd = self.Cd_ox

        N_holes = self.numox

        mdot = mdot_vapor_orifice(P1,T1,P2,Cd,N_holes,self.Aox)

        return mdot
        


def mdot_vapor_orifice(P1, T1, P2, Cd, N_holes, A_inj, fluid='N2O'):
    """
    Compressible vapor flow through orifice.
    
    Parameters
    ----------
    P1     : float - upstream pressure [Pa]
    T1     : float - upstream temperature [K]
    P2     : float - downstream pressure [Pa]
    Cd     : float - discharge coefficient [-]
    N_holes: int   - number of orifice holes
    d_m    : float - orifice diameter [m]
    fluid  : str   - CoolProp fluid name
    
    Returns
    -------
    mdot : float - mass flow rate [kg/s]
    """
    A = A_inj

    # Get gas properties at upstream state (superheated vapor)
    gamma  = CP.PropsSI('Cpmass', 'T|gas', T1, 'P', P1, fluid) / \
             CP.PropsSI('Cvmass', 'T|gas', T1, 'P', P1, fluid)
    R_spec = CP.PropsSI('gas_constant', fluid) / CP.PropsSI('molar_mass', fluid)
    rho1   = CP.PropsSI('D', 'T|gas', T1, 'P', P1, fluid)

    # Critical pressure ratio
    PR_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))

    # Vandenkerckhove / choked flow factor
    Gamma = math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

    if P2 / P1 <= PR_crit:
        # Choked flow
        mdot = Cd * A * P1 * Gamma / math.sqrt(R_spec * T1)

    else:
        # Unchoked - isentropic subsonic
        PR = P2 / P1
        mdot = Cd * A * math.sqrt(
            2 * rho1 * P1 * (gamma / (gamma - 1)) *
            (PR**(2/gamma) - PR**((gamma+1)/gamma))
        )

    return mdot

def nozzle(mdot, gamma, R_spec, Tc, A_t, A_e, P_amb):
    """
    Solve for chamber pressure and thrust from mass flow rate.

    Parameters
    ----------
    mdot   : float - total mass flow rate [kg/s]
    gamma  : float - ratio of specific heats [-]
    R_spec : float - specific gas constant [J/kg·K]
    Tc     : float - chamber temperature [K]
    A_t    : float - throat area [m²]
    A_e    : float - exit area [m²]
    P_amb  : float - ambient pressure [Pa]

    Returns
    -------
    Pc : float - chamber pressure [Pa]
    F  : float - thrust [N]
    Pe : float - exit pressure [Pa]
    ve : float - exit velocity [m/s]
    """
    # c* and chamber pressure
    Gamma = np.sqrt(gamma) * (2/(gamma+1)) ** ((gamma+1) / (2*(gamma-1)))
    cstar = np.sqrt(R_spec * Tc) / Gamma
    Pc    = (mdot * cstar) / A_t

    # Exit Mach from area-Mach relation (supersonic root)
    AR = A_e / A_t
    Me = brentq(lambda Me: (1/Me) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * Me**2))
                **((gamma+1)/(2*(gamma-1))) - AR, 1.0, 50.0)

    # Isentropic exit conditions
    Pe = Pc * (1 + (gamma-1)/2 * Me**2) ** (-gamma/(gamma-1))
    Te = Tc * (1 + (gamma-1)/2 * Me**2) ** (-1)
    ve = Me * np.sqrt(gamma * R_spec * Te)

    # Thrust
    F = mdot * ve + (Pe - P_amb) * A_e

    return Pc, F, Pe, ve