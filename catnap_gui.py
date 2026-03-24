
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import json
import os
import time
import webbrowser
from math import pi, tan, sqrt, cos
import numpy as np
from scipy.optimize import brentq
import CoolProp.CoolProp as CP

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


from RXPI_CATNAP_Fluids     import Injector_obj
from RXPI_CATNAP_Combustion import (SolvePC, CombustionPerformance,
                                     Props_obj, Transport_obj)
from RXPI_CATNAP_Regen      import Regen_obj



def _rootT2(T2, v2, u2, Pressurant):
    uf = CP.PropsSI('U', 'T', T2, 'Q', 0, Pressurant)
    ug = CP.PropsSI('U', 'T', T2, 'Q', 1, Pressurant)
    vf = 1 / CP.PropsSI('D', 'T', T2, 'Q', 0, Pressurant)
    vg = 1 / CP.PropsSI('D', 'T', T2, 'Q', 1, Pressurant)
    x2 = (v2 - vf) / (vg - vf)
    return u2 - uf - x2 * (ug - uf)


def _timestep(T1, mdotox, mdotf, x1, dt, m1, v1, props, dP_piston_psi):
    """Adiabatic tank timestep — identical physics to RXPI_CATNAP.timestep()."""
    Pressurant = props.ox
    Fuel       = props.fuel

    if 0.0 <= x1 <= 0.999:
        phase = 'saturated'
        uf1 = CP.PropsSI('U', 'T', T1, 'Q', 0, Pressurant)
        ug1 = CP.PropsSI('U', 'T', T1, 'Q', 1, Pressurant)
        u1  = uf1 + x1 * (ug1 - uf1)
        hf1 = CP.PropsSI('H', 'T', T1, 'Q', 0, Pressurant)
        vf1 = 1 / CP.PropsSI('D', 'T', T1, 'Q', 0, Pressurant)
        vg1 = 1 / CP.PropsSI('D', 'T', T1, 'Q', 1, Pressurant)
        v1  = vf1 + x1 * (vg1 - vf1)
        P1  = CP.PropsSI('P', 'T', T1, 'Q', 0, Pressurant)
        vfuel = 1 / CP.PropsSI('D', 'T', T1, 'P',
                                P1 - dP_piston_psi * 6894.75729, Fuel)
        Wp  = -P1 * mdotf * vfuel * dt
        u2  = (Wp - mdotox * dt * hf1 + m1 * u1) / (m1 - mdotox * dt)
        v2  = (v1 * m1 + mdotf * vfuel * dt)      / (m1 - mdotox * dt)
        T2  = brentq(_rootT2, T1 - 20,
                     CP.PropsSI('Tcrit', Pressurant) - 2,
                     args=(v2, u2, Pressurant))
        vf2 = 1 / CP.PropsSI('D', 'T', T2, 'Q', 0, Pressurant)
        vg2 = 1 / CP.PropsSI('D', 'T', T2, 'Q', 1, Pressurant)
        x2  = (v2 - vf2) / (vg2 - vf2)
        m2  = m1 - mdotox * dt
    else:
        phase = 'vapor'
        x2    = 1.0
        D1    = 1 / v1
        u1    = CP.PropsSI('U', 'T', T1, 'D', D1, Pressurant)
        hg1   = CP.PropsSI('H', 'T', T1, 'D', D1, Pressurant)
        P1    = CP.PropsSI('P', 'T', T1, 'D', D1, Pressurant)
        vfuel = 1 / CP.PropsSI('D', 'T', T1, 'P',
                                P1 - dP_piston_psi * 6894.75729, Fuel)
        Wp  = -P1 * mdotf * vfuel * dt
        u2  = (Wp - mdotox * dt * hg1 + m1 * u1) / (m1 - mdotox * dt)
        v2  = (v1 * m1 + mdotf * vfuel * dt)      / (m1 - mdotox * dt)
        m2  = m1 - mdotox * dt
        D2  = 1 / v2
        T2  = CP.PropsSI('T', 'U', u2, 'D', D2, Pressurant)

    return T2, x2, m2, v2, phase


def _make_R(cfg):
    """Return R(z) closure from cfg geometry (all lengths stored in inches)."""
    i2m = 1 / 39.3700787
    Ln  = cfg['Lnozzle'] * i2m
    Lc1 = cfg['Lcon1']   * i2m
    Lc2 = cfg['Lcon2']   * i2m
    Lch = cfg['Lcham']   * i2m
    Re  = cfg['Rexit']   * i2m
    Rc1 = cfg['Rc1']     * i2m
    Rc2 = cfg['Rc2']     * i2m
    ang = cfg['expansionangle']

    def R(z):
        if 0 <= z < Ln:
            return Re - tan(pi * ang / 180) * z
        elif Ln <= z < Ln + Lc1:
            zz = z - Ln
            return Re - tan(pi * ang / 180) * Ln + Rc1 - sqrt(Rc1**2 - zz**2)
        elif Ln + Lc1 <= z <= Ln + Lc1 + Lc2:
            zz = z - (Ln + Lc1); zi = Lc1
            return (Re - tan(pi * ang / 180) * Ln
                    + Rc1 - sqrt(Rc1**2 - zi**2)
                    + sqrt(Rc2**2 - (zz - Lc2)**2)
                    - sqrt(Rc2**2 - Lc2**2))
        elif Ln + Lc1 + Lc2 <= z < Ln + Lc1 + Lc2 + Lch + 1e-2:
            zz = Lc2; zi = Lc1
            return (Re - tan(pi * ang / 180) * Ln
                    + Rc1 - sqrt(Rc1**2 - zi**2)
                    + sqrt(Rc2**2 - (zz - Lc2)**2)
                    - sqrt(Rc2**2 - Lc2**2))
        else:
            raise ValueError(f'z={z:.5f} outside nozzle range')
    return R


class CATNAPGui:

 
    BG     = '#07090f'
    PANEL  = '#0d1117'
    BORDER = '#1a2233'
    ACCENT = '#00d4ff'
    ORANGE = '#ff6b35'
    GREEN  = '#34d399'
    PURPLE = '#a78bfa'
    RED    = '#f87171'
    YELLOW = '#facc15'
    TEXT   = '#c9d1d9'
    MUTED  = '#586069'

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('CATNAP  —  Altair Engine Solver')
        self.root.configure(bg=self.BG)
        self.root.minsize(1120, 720)

        self.q         = queue.Queue()
        self.cancel_ev = threading.Event()
        self.running   = False
        self.inputs    = {}          # {key: tk.StringVar}

        # live-plot data
        self._lv = {k: [] for k in
                    ('t', 'Pc', 'P2', 'T2', 'F', 'Isp', 'x2', 'mdot')}

        self._build_ui()
        self._poll()

    # ─────────────────── UI CONSTRUCTION ───────────────────────

    def _build_ui(self):
        left  = tk.Frame(self.root, bg=self.BG, width=400)
        right = tk.Frame(self.root, bg=self.BG)
        left.pack(side='left', fill='y', padx=(10, 0), pady=10)
        right.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        left.pack_propagate(False)

        # header
        hdr = tk.Frame(left, bg=self.BG)
        hdr.pack(fill='x', pady=(0, 6))
        tk.Label(hdr, text='CAT', font=('Rajdhani', 24, 'bold'),
                 bg=self.BG, fg='white').pack(side='left')
        tk.Label(hdr, text='NAP', font=('Rajdhani', 24, 'bold'),
                 bg=self.BG, fg=self.ACCENT).pack(side='left')
        tk.Label(hdr, text='  RXPI',
                 font=('Courier New', 9), bg=self.BG,
                 fg=self.MUTED).pack(side='left', padx=(4, 0))

        # scrollable input area
        wrap = tk.Frame(left, bg=self.BG)
        wrap.pack(fill='both', expand=True)
        vsb = tk.Scrollbar(wrap, orient='vertical', bg=self.BG)
        vsb.pack(side='right', fill='y')
        cvs = tk.Canvas(wrap, bg=self.BG, bd=0,
                        highlightthickness=0, yscrollcommand=vsb.set)
        cvs.pack(side='left', fill='both', expand=True)
        vsb.config(command=cvs.yview)
        self._inp_frame = tk.Frame(cvs, bg=self.BG)
        wid = cvs.create_window((0, 0), window=self._inp_frame, anchor='nw')
        self._inp_frame.bind('<Configure>',
            lambda e: (cvs.configure(scrollregion=cvs.bbox('all')),
                       cvs.itemconfig(wid, width=cvs.winfo_width())))
        cvs.bind('<Configure>',
            lambda e: cvs.itemconfig(wid, width=e.width))
        cvs.bind_all('<MouseWheel>',
            lambda e: cvs.yview_scroll(int(-1*(e.delta/120)), 'units'))

        self._add_inputs()

        # run / stop
        bf = tk.Frame(left, bg=self.BG, pady=8)
        bf.pack(fill='x')
        self._run_btn = tk.Button(
            bf, text='▶   RUN SIMULATION',
            font=('Courier New', 10, 'bold'),
            bg=self.ACCENT, fg='#000', activebackground='#00b8d9',
            relief='flat', cursor='hand2',
            command=self.start_run, padx=10, pady=8)
        self._run_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))

        self._stop_btn = tk.Button(
            bf, text='■  STOP',
            font=('Courier New', 10),
            bg=self.PANEL, fg=self.RED, activebackground='#1a0a0a',
            relief='flat', cursor='hand2', state='disabled',
            command=self.stop_run, padx=10, pady=8)
        self._stop_btn.pack(side='left')

        self._build_plots(right)

    # ─── section / field helpers ───

    def _section(self, text):
        f = tk.Frame(self._inp_frame, bg=self.BG)
        f.pack(fill='x', pady=(10, 1), padx=4)
        tk.Label(f, text=f'── {text} ──',
                 font=('Courier New', 9, 'bold'),
                 bg=self.BG, fg=self.ACCENT, anchor='w').pack(fill='x')

    def _field(self, key, label, default, unit=''):
        row = tk.Frame(self._inp_frame, bg=self.BG)
        row.pack(fill='x', padx=8, pady=1)
        tk.Label(row, text=label, font=('Courier New', 9),
                 bg=self.BG, fg=self.TEXT, width=26,
                 anchor='w').pack(side='left')
        var = tk.StringVar(value=str(default))
        tk.Entry(row, textvariable=var, font=('Courier New', 9),
                 bg=self.PANEL, fg=self.ACCENT,
                 insertbackground=self.ACCENT,
                 relief='flat', bd=4, width=11).pack(side='left')
        if unit:
            tk.Label(row, text=unit, font=('Courier New', 8),
                     bg=self.BG, fg=self.MUTED,
                     width=7, anchor='w').pack(side='left', padx=2)
        self.inputs[key] = var

    def _add_inputs(self):
        self._section('SIMULATION')
        self._field('simtime',   'Duration',               25,        's')
        self._field('numsteps',  'Time steps',             150,       '')

        self._section('TANK / FEED')
        self._field('m1',        'Init propellant mass',   53.3438,   'kg')
        self._field('T_init',    'Init tank temp',         290,       'K')
        self._field('dP_piston', 'Piston ΔP',              15,        'psi')

        self._section('INITIAL FLOW RATES')
        self._field('mdotox0',   'Init ox flow',           2.548,     'kg/s')
        self._field('mdotf0',    'Init fuel flow',         0.784,     'kg/s')
        self._field('mdotfilm',  'Film cool. flow',        0.118,     'kg/s')
        self._field('x1',        'Init vapor quality',     0.01,      '')

        # Note: mdot_coolant is derived each step as mdotf + mdotfilm (new commit)
        self._note('  ↳  Coolant flow = fuel + film (computed per step)')

        self._section('ENGINE GEOMETRY  (inches)')
        self._field('Lnozzle',        'Nozzle length',     4.19,      'in')
        self._field('Lcon1',          'Con arc 1',         0.367,     'in')
        self._field('Lcon2',          'Con arc 2',         2.912,     'in')
        self._field('Lcham',          'Chamber length',    4.841,     'in')
        self._field('Rexit',          'Exit radius',       2.13,      'in')
        self._field('Rc1',            'Curvature R1',      0.589,     'in')
        self._field('Rc2',            'Curvature R2',      4.678,     'in')
        self._field('expansionangle', 'Diverge angle',     15,        'deg')

        self._section('NOZZLE  (mm)')
        self._field('Dt_mm',     'Throat diameter',        52.77,     'mm')
        self._field('De_mm',     'Exit diameter',          111.04,    'mm')

        self._section('INJECTOR')
        self._field('Cd_ox',     'Cd oxidiser',            0.55,      '')
        self._field('Cd_fuel',   'Cd fuel',                0.65,      '')
        self._field('Cd_film',   'Cd film cool.',          0.65,      '')
        self._field('numox',     'N ox holes',             25,        '')
        self._field('numfuel',   'N fuel holes',           20,        '')
        self._field('numfilm',   'N film holes',           20,        '')
        self._field('Dox',       'Ox hole ⌀',              2.533,     'mm')
        self._field('Dfuel',     'Fuel hole ⌀',            1.240,     'mm')
        self._field('Dfilm',     'Film hole ⌀',            0.480,     'mm')

        self._section('REGEN COOLING')
        self._field('Tcool',     'Coolant inlet T',        290,       'K')
        # Pcool_init is now P2f at each snapshot — no longer a fixed input
        self._note('  ↳  Coolant inlet P = tank P − piston ΔP − channel ΔP  (coupled)')
        self._field('twall',     'Wall thickness',         1.5,       'mm')
        self._field('lfmin',     'Channel height',         1.0,       'mm')
        self._field('wcmin',     'Min channel width',      1.5,       'mm')
        self._field('numch',     'N channels',             90,        '')
        self._field('k_wall',    'Wall conductivity',      237,       'W/mK')
        self._field('eps_rough', 'Roughness ε',            15,        'μm')
        self._field('numpts_z',  'Axial points',           80,        '')
        self._field('throat_rc', 'Throat curvature rad',   25.4,      'mm')
        self._field('genangle',  'Generatrix angle',       35,        'deg')

        self._section('REGEN SNAPSHOT TIMES')
        row = tk.Frame(self._inp_frame, bg=self.BG)
        row.pack(fill='x', padx=8, pady=2)
        tk.Label(row, text='Times (comma-separated)',
                 font=('Courier New', 9), bg=self.BG,
                 fg=self.TEXT, anchor='w').pack(fill='x')
        var = tk.StringVar(value='0.5, 3.3, 6.1, 8.9, 11.7, 14.4, 17.2')
        tk.Entry(row, textvariable=var, font=('Courier New', 9),
                 bg=self.PANEL, fg=self.ACCENT,
                 insertbackground=self.ACCENT,
                 relief='flat', bd=4).pack(fill='x', pady=2)
        self.inputs['regen_times'] = var

    def _note(self, text):
        """Small grey informational label — not an input."""
        tk.Label(self._inp_frame, text=text,
                 font=('Courier New', 8), bg=self.BG,
                 fg=self.MUTED, anchor='w').pack(fill='x', padx=8)

    # ─── right panel: live plots ───

    def _build_plots(self, parent):
        fig = Figure(figsize=(7, 7.5), facecolor=self.BG)
        fig.subplots_adjust(left=0.10, right=0.88,
                            top=0.96, bottom=0.07, hspace=0.44)
        GC = '#1a2233'; TC = '#586069'

        def style(ax, yl, y2l=None, y2c=None):
            ax.set_facecolor(self.BG)
            ax.tick_params(colors=TC, labelsize=9)
            ax.set_ylabel(yl, color=TC, fontsize=9, fontfamily='monospace')
            ax.set_xlabel('Time  (s)', color=TC, fontsize=9,
                          fontfamily='monospace')
            for sp in ax.spines.values():
                sp.set_color(GC)
            ax.grid(color=GC, linestyle='--', linewidth=0.5)
            if y2l:
                ax2 = ax.twinx()
                ax2.set_ylabel(y2l, color=y2c or TC, fontsize=9,
                               fontfamily='monospace')
                ax2.tick_params(colors=y2c or TC, labelsize=9)
                for sp in ax2.spines.values():
                    sp.set_color(GC)
                return ax2

        # ── subplot 1: chamber P, tank P, tank T ──
        self.ax1  = fig.add_subplot(311)
        self.ax1b = style(self.ax1, 'Pressure  (MPa)',
                          'Tank Temp  (K)', self.ORANGE)
        self.l_Pc, = self.ax1.plot([], [], color=self.ACCENT, lw=1.8,
                                   label='Pc chamber (MPa)')
        self.l_P2, = self.ax1.plot([], [], color=self.PURPLE, lw=1.8,
                                   label='P₂ tank (MPa)')
        self.l_T2, = self.ax1b.plot([], [], color=self.ORANGE, lw=1.5,
                                    ls='--', label='Tank T (K)')
        self.ax1.set_title('Chamber & Tank Conditions',
                           color=self.TEXT, fontsize=10,
                           fontfamily='monospace', pad=4)
        self.ax1.legend(
            handles=[self.l_Pc, self.l_P2, self.l_T2],
            loc='upper right', fontsize=8,
            facecolor=self.PANEL, edgecolor=self.BORDER,
            labelcolor=self.TEXT, framealpha=0.9)

        # ── subplot 2: thrust & Isp ──
        self.ax2  = fig.add_subplot(312)
        self.ax2b = style(self.ax2, 'Thrust  (kN)',
                          'Isp  (s)', self.PURPLE)
        self.l_F,   = self.ax2.plot([], [], color=self.GREEN,  lw=1.8,
                                    label='Thrust (kN)')
        self.l_Isp, = self.ax2b.plot([], [], color=self.PURPLE, lw=1.5,
                                     ls='--', label='Isp (s)')
        self.ax2.set_title('Engine Performance',
                           color=self.TEXT, fontsize=10,
                           fontfamily='monospace', pad=4)
        self.ax2.legend(
            handles=[self.l_F, self.l_Isp],
            loc='upper right', fontsize=8,
            facecolor=self.PANEL, edgecolor=self.BORDER,
            labelcolor=self.TEXT, framealpha=0.9)

        # ── subplot 3: vapor quality & mass flow ──
        self.ax3  = fig.add_subplot(313)
        self.ax3b = style(self.ax3, 'Vapor Quality  x',
                          'Mass Flow  (kg/s)', self.RED)
        self.ax3.set_ylim(0, 1.08)
        self.l_x2,   = self.ax3.plot([], [], color=self.ACCENT, lw=1.8,
                                     label='Vapor quality x')
        self.l_mdot, = self.ax3b.plot([], [], color=self.RED, lw=1.5,
                                      ls='--', label='ṁ total (kg/s)')
        self.ax3.axhline(1.0, color=self.ORANGE, lw=0.8, ls=':', alpha=0.5)
        self.ax3.set_title('Tank Phase & Mass Flow',
                           color=self.TEXT, fontsize=10,
                           fontfamily='monospace', pad=4)
        self.ax3.legend(
            handles=[self.l_x2, self.l_mdot],
            loc='upper right', fontsize=8,
            facecolor=self.PANEL, edgecolor=self.BORDER,
            labelcolor=self.TEXT, framealpha=0.9)

        self._canvas = FigureCanvasTkAgg(fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        # ── status strip ──
        sf = tk.Frame(parent, bg=self.PANEL, pady=5)
        sf.pack(fill='x')
        self._phase_var  = tk.StringVar(value='READY')
        self._status_var = tk.StringVar(
            value='Fill in parameters and click  ▶ RUN')
        self._pct_var    = tk.DoubleVar(value=0)

        tk.Label(sf, textvariable=self._phase_var,
                 font=('Courier New', 9, 'bold'),
                 bg=self.PANEL, fg=self.ORANGE,
                 width=12, anchor='w').pack(side='left', padx=8)

        ttk.Progressbar(sf, variable=self._pct_var,
                        maximum=100, length=180,
                        mode='determinate').pack(side='left', padx=4)

        tk.Label(sf, textvariable=self._status_var,
                 font=('Courier New', 9), bg=self.PANEL,
                 fg=self.TEXT, anchor='w').pack(side='left', padx=8,
                                                fill='x', expand=True)

    # ─────────────────── INPUT READING ─────────────────────────

    def _g(self, key, typ=float):
        try:
            return typ(self.inputs[key].get())
        except Exception:
            raise ValueError(f"Bad value for '{key}': "
                             f"{self.inputs[key].get()!r}")

    def _collect_cfg(self):
        return {
            'simtime':        self._g('simtime'),
            'numsteps':       self._g('numsteps', int),
            'm1':             self._g('m1'),
            'T_init':         self._g('T_init'),
            'dP_piston_psi':  self._g('dP_piston'),
            'mdotox0':        self._g('mdotox0'),
            'mdotf0':         self._g('mdotf0'),
            'mdotfilm':       self._g('mdotfilm'),
            'x1':             self._g('x1'),
            # geometry (inches)
            'Lnozzle':        self._g('Lnozzle'),
            'Lcon1':          self._g('Lcon1'),
            'Lcon2':          self._g('Lcon2'),
            'Lcham':          self._g('Lcham'),
            'Rexit':          self._g('Rexit'),
            'Rc1':            self._g('Rc1'),
            'Rc2':            self._g('Rc2'),
            'expansionangle': self._g('expansionangle'),
            # nozzle throat/exit mm
            'Dt_mm':          self._g('Dt_mm'),
            'De_mm':          self._g('De_mm'),
            # injector
            'Cd_ox':          self._g('Cd_ox'),
            'Cd_fuel':        self._g('Cd_fuel'),
            'Cd_film':        self._g('Cd_film'),
            'numox':          self._g('numox', int),
            'numfuel':        self._g('numfuel', int),
            'numfilm':        self._g('numfilm', int),
            'Dox':            self._g('Dox')   / 1000,   # mm → m
            'Dfuel':          self._g('Dfuel') / 1000,
            'Dfilm':          self._g('Dfilm') / 1000,
            # regen
            'Tcool':          self._g('Tcool'),
            # Pcool now computed per-step — not a fixed input
            'twall':          self._g('twall')     / 1000,   # mm → m
            'lfmin':          self._g('lfmin')     / 1000,
            'wcmin':          self._g('wcmin')     / 1000,
            'numch':          self._g('numch', int),
            'k_wall':         self._g('k_wall'),
            'eps_rough':      self._g('eps_rough') * 1e-6,   # μm → m
            'numpts_z':       self._g('numpts_z', int),
            'throat_rc':      self._g('throat_rc') / 1000,   # mm → m
            'genangle':       self._g('genangle'),            # degrees
            'regen_times':    [float(s.strip()) for s in
                               self.inputs['regen_times'].get().split(',')],
        }

    # ─────────────────── RUN / STOP ────────────────────────────

    def start_run(self):
        if self.running:
            return
        try:
            cfg = self._collect_cfg()
        except ValueError as e:
            messagebox.showerror('Input Error', str(e))
            return

        for lst in self._lv.values():
            lst.clear()
        for ln in (self.l_Pc, self.l_P2, self.l_T2,
                   self.l_F, self.l_Isp, self.l_x2, self.l_mdot):
            ln.set_data([], [])
        self._canvas.draw_idle()

        self.cancel_ev.clear()
        self.running = True
        self._run_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self._pct_var.set(0)
        self._phase_var.set('STARTING')
        self._status_var.set('Initialising solver…')

        threading.Thread(target=self._solver_thread,
                         args=(cfg,), daemon=True).start()

    def stop_run(self):
        self.cancel_ev.set()
        self._phase_var.set('STOPPING')
        self._status_var.set('Cancelling — waiting for current step…')

    # ─────────────────── SOLVER THREAD ─────────────────────────

    def _solver_thread(self, cfg):
        try:
            self._run_catnap(cfg)
        except Exception as exc:
            import traceback
            self.q.put(('error', str(exc), traceback.format_exc()))

    def _run_catnap(self, cfg):
        i2m   = 1 / 39.3700787
        P_amb = 14.7 * 6895 * 0.8      # Pa ambient

        # ── build objects ──
        props  = Props_obj('N2O', 'ETHANOL', 'N2O', 'ETHANOL')
        pintle = Injector_obj(
            cfg['Cd_fuel'], cfg['Cd_ox'], cfg['Cd_film'],
            cfg['numox'], cfg['numfuel'], cfg['numfilm'],
            cfg['Dox'], cfg['Dfuel'], cfg['Dfilm'], props)

        R_fn    = _make_R(cfg)
        Le      = (cfg['Lnozzle'] + cfg['Lcon1'] +
                   cfg['Lcon2']   + cfg['Lcham']) * i2m

        Dt_m    = cfg['Dt_mm'] / 1000
        De_m    = cfg['De_mm'] / 1000
        A_t     = pi * (Dt_m / 2) ** 2
        eps     = (De_m / Dt_m) ** 2
        Rthroat = Dt_m / 2

        # geom array in metres (same order as RXPI_CATNAP)
        geom = np.array([
            cfg['Lnozzle'] * i2m, cfg['Lcon1'] * i2m,
            cfg['Lcon2']   * i2m, cfg['Lcham'] * i2m,
            cfg['Rexit']   * i2m, cfg['Rc1']   * i2m,
            cfg['Rc2']     * i2m, 0.0])

        # ── Regen_obj now takes genangle as final argument ──
        altair = Regen_obj(
            cfg['twall'], cfg['lfmin'], cfg['wcmin'],
            R_fn, Rthroat, cfg['throat_rc'],
            cfg['numch'], props.fuel,
            cfg['k_wall'], cfg['eps_rough'],
            cfg['numpts_z'], Le,
            cfg['genangle'])              # ← new commit

        # ── initial conditions ──
        simtime  = cfg['simtime']
        numsteps = cfg['numsteps']
        dt       = simtime / numsteps
        timevec  = np.linspace(0, simtime, numsteps)

        T1       = cfg['T_init']
        m1       = cfg['m1']
        x1       = cfg['x1']
        dP       = cfg['dP_piston_psi']
        Tcool_init = cfg['Tcool']

        mdotox   = cfg['mdotox0']
        mdotf    = cfg['mdotf0']
        mdotfilm = cfg['mdotfilm']

        # ── mdot_coolant = mdotf + mdotfilm, updated each step ──
        mdot_coolant = mdotf + mdotfilm   # initial value

        Pc_init = 340 * 6895   # Pa, initial bracket
        Pc = SolvePC(mdotox + mdotf + mdotfilm,
                     mdotox / (mdotf + mdotfilm),
                     A_t, Pc_init, props)

        vf0 = 1 / CP.PropsSI('D', 'Q', 0, 'T', T1, 'N2O')
        vg0 = 1 / CP.PropsSI('D', 'Q', 1, 'T', T1, 'N2O')
        v1  = vf0 + x1 * (vg0 - vf0)

        # result collectors
        Pc_r=[]; P2_r=[]; T2_r=[]; x2_r=[]
        F_r=[]; Isp_r=[]; mdot_r=[]; MR_r=[]

        # regen snapshot setup
        snap_map = {}
        for ts in cfg['regen_times']:
            idx = int(round(ts / dt))
            if 0 <= idx < numsteps:
                snap_map[idx] = ts

        nz      = altair.numpts_z
        ns      = len(snap_map)
        Tc3     = np.zeros((ns, nz)); Pc3 = np.zeros((ns, nz))
        hg3     = np.zeros((ns, nz)); Tw3 = np.zeros((ns, nz))
        Qf3     = np.zeros((ns, nz))
        tr3=[]; tC3=[]; snap_times=[]; sc=0

        # ── MAIN LOOP ──
        for i in range(numsteps):
            if self.cancel_ev.is_set():
                self.q.put(('cancelled',))
                return

            t = (i / numsteps) * simtime

            T2, x2, m2, v2, phase = _timestep(
                T1, mdotox, mdotf, x1, dt, m1, v1, props, dP)

            # ── tank pressure ──
            if phase == 'saturated':
                P2     = CP.PropsSI('P', 'T', T2, 'Q', 0, 'N2O')
                mdotox = pintle.mdot_ox_nhne(P2, T2, Pc)
            else:
                D2     = 1 / v2
                P2     = CP.PropsSI('P', 'T', T2, 'D', D2, 'N2O')
                mdotox = pintle.mdot_vapor_orifice(P2, T2, Pc)

            # ── NEW: subtract piston ΔP and channel ΔP before injector ──
            P2f = P2 - (dP * 6894.75729)

            # channel pressure loss coupling (new commit: dP_channel_Approx)
            try:
                dPchannels = altair.dP_channel_Approx(
                    Tcool_init, P2f, mdot_coolant)
            except Exception:
                dPchannels = 0.0   # graceful fallback if geometry edge case

            P2inj = P2f - dPchannels

            mdotf    = pintle.mdot_fuel(P2inj, T2, Pc)
            mdotfilm = pintle.mdot_film(P2inj, T2, Pc)

            # ── mdot_coolant updates with fuel + film each step ──
            mdot_coolant = mdotf + mdotfilm

            mdot_tot = mdotox + mdotf + mdotfilm
            MR       = mdotox / (mdotf + mdotfilm)

            Pc = SolvePC(mdot_tot, MR, A_t, Pc, props)
            _, F, Isp, _ = CombustionPerformance(
                mdot_tot, MR, A_t, Pc, P_amb, eps, props)

            # ── regen snapshot ──
            if i in snap_map and sc < ns:
                try:
                    # Pcool_init = P2f (live coupling — new commit)
                    Pcool_init_snap = P2f
                    trn = Transport_obj(
                        mdot_tot, MR, A_t, props, geom, eps, Pc)
                    Tc_a, Pc_a, hg_a, Tw_a, Qf_a = \
                        altair.SOLVE_REGEN(
                            mdot_coolant, Tcool_init,
                            Pcool_init_snap, trn)

                    # transport props snapshot
                    Cp_t, mu_t, k_t, Pr_t, gm_t = trn.Chambertransport()
                    Mf = lambda zz: trn.Mach(zz, altair.R)
                    tr3.append([[float(Cp_t(z)), float(mu_t(z)),
                                 float(k_t(z)),  float(Pr_t(z)),
                                 float(gm_t(z)), float(Mf(z))]
                                for z in altair.z_array])

                    Tc_fn, _, _, _ = trn.TPRhostag()
                    tC3.append([[float(r), float(t0)]
                                for r, t0 in
                                [Tc_fn(z, Mf) for z in altair.z_array]])

                    Tc3[sc]=Tc_a; Pc3[sc]=Pc_a; hg3[sc]=hg_a
                    Tw3[sc]=Tw_a; Qf3[sc]=Qf_a
                    snap_times.append(snap_map[i])
                    sc += 1
                except Exception:
                    pass   # bad snapshot → skip, don't abort run

            # store normalised results
            Pc_r.append(Pc / 1e6)      # Pa → MPa
            P2_r.append(P2 / 1e6)
            T2_r.append(T2)
            x2_r.append(x2)
            F_r.append(F  / 1000)     # N → kN
            Isp_r.append(Isp)
            mdot_r.append(mdot_tot)
            MR_r.append(MR)

            T1=T2; x1=x2; m1=m2; v1=v2

            pct = 100 * (i + 1) / numsteps
            self.q.put(('step', t,
                        Pc/1e6, P2/1e6, T2,
                        F/1000, Isp, x2, mdot_tot,
                        phase, pct))

            # Yield the GIL so tkinter's main thread can run the
            # after() poll callback and actually paint the canvas.
            # Without this, CoolProp/scipy hold the GIL for the
            # entire run and the UI freezes until the solver finishes.
            time.sleep(0.01)

        # ── total impulse (kN·s) ──
        Imp = sum(0.5*(F_r[j]+F_r[j+1])*dt
                  for j in range(numsteps-1))

        # ── nozzle contour for dashboard ──
        nozzle_R = [float(R_fn(z)) for z in altair.z_array]

        results = {
            'timevec':       timevec.tolist(),
            'Pc_arr':        Pc_r,          # already MPa
            'P2_arr':        P2_r,          # already MPa
            'T2_arr':        T2_r,
            'x2_arr':        x2_r,
            'F_arr':         F_r,           # already kN
            'Isp_arr':       Isp_r,
            'mdot_arr':      mdot_r,
            'massratio_arr': MR_r,
            'z_array':       altair.z_array.tolist(),
            'regen_times':   snap_times,
            'nozzle_R':      nozzle_R,
            'Tcool_3d':      Tc3[:sc].tolist(),
            'Pcool_3d':      (Pc3[:sc] / 1e6).tolist(),    # Pa → MPa
            'hg_3d':         hg3[:sc].tolist(),
            'Twall_3d':      Tw3[:sc].tolist(),
            'Qflux_3d':      (Qf3[:sc] / 1e6).tolist(),    # W/m² → MW/m²
            'transport_3d':  tr3,
            'tempsC_3d':     tC3,
        }

        here     = os.path.dirname(os.path.abspath(__file__))
        json_out = os.path.join(here, 'catnap_results.json')
        with open(json_out, 'w') as fj:
            json.dump(results, fj)

        dash     = os.path.join(here, 'catnap_dashboard.html')
        html_out = os.path.join(here, 'catnap_results.html')
        try:
            with open(dash, 'r', encoding='utf-8') as fh:
                html = fh.read()
            inject = ('<script>var AUTOLOAD = '
                      + json.dumps(results) + ';</script>')
            html = html.replace('</head>', inject + '\n</head>')
            with open(html_out, 'w', encoding='utf-8') as fh:
                fh.write(html)
        except FileNotFoundError:
            html_out = None

        self.q.put(('done', html_out, Imp))

    # ─────────────────── QUEUE POLL ────────────────────────────

    def _poll(self):
        """
        Drain every pending message from the solver thread, accumulate all
        the step data, then call _redraw() exactly once per poll cycle.
        This is the key to live updates: draw_idle() never fires while a
        tight while-loop is running, so we do one explicit draw() at the end.
        """
        needs_redraw = False
        # latest status text to show (only the most recent step matters)
        last_status  = None
        last_phase   = None
        last_pct     = None

        try:
            while True:
                msg  = self.q.get_nowait()
                kind = msg[0]

                if kind == 'step':
                    # accumulate data — do NOT redraw yet
                    _, t, Pc, P2, T2, F, Isp, x2, mdot, phase, pct = msg
                    lv = self._lv
                    lv['t'].append(t);    lv['Pc'].append(Pc)
                    lv['P2'].append(P2);  lv['T2'].append(T2)
                    lv['F'].append(F);    lv['Isp'].append(Isp)
                    lv['x2'].append(x2);  lv['mdot'].append(mdot)
                    # keep only the last values for status labels
                    last_phase  = phase.upper()
                    last_pct    = pct
                    last_status = (f't={t:.2f} s   Pc={Pc:.3f} MPa   '
                                   f'F={F:.2f} kN   T₂={T2:.1f} K   '
                                   f'x={x2:.4f}')
                    needs_redraw = True

                elif kind == 'done':
                    _, html_out, Imp = msg
                    # draw final state before opening browser
                    if needs_redraw:
                        self._redraw()
                        needs_redraw = False
                    self._finish(html_out, Imp)

                elif kind == 'cancelled':
                    self._reset()
                    self._phase_var.set('STOPPED')
                    self._status_var.set('Run cancelled by user.')

                elif kind == 'error':
                    _, err, tb = msg
                    self._reset()
                    self._phase_var.set('ERROR')
                    self._status_var.set(f'Error: {err}')
                    messagebox.showerror('Solver Error',
                                         f'{err}\n\n{tb}')

        except queue.Empty:
            pass
        finally:
            # update status labels from last step (cheap Tk ops)
            if last_phase  is not None: self._phase_var.set(last_phase)
            if last_pct    is not None: self._pct_var.set(last_pct)
            if last_status is not None: self._status_var.set(last_status)

            # ONE render call per poll tick — this is what actually
            # makes the curves appear while the solver is running
            if needs_redraw:
                self._redraw()

            self.root.after(100, self._poll)

    def _redraw(self):
        """Push accumulated data into the plot lines and render once."""
        t = self._lv['t']
        self.l_Pc.set_data(t, self._lv['Pc'])
        self.l_P2.set_data(t, self._lv['P2'])
        self.l_T2.set_data(t, self._lv['T2'])
        self.ax1.relim();  self.ax1.autoscale_view()
        self.ax1b.relim(); self.ax1b.autoscale_view()

        self.l_F.set_data(t, self._lv['F'])
        self.l_Isp.set_data(t, self._lv['Isp'])
        self.ax2.relim();  self.ax2.autoscale_view()
        self.ax2b.relim(); self.ax2b.autoscale_view()

        self.l_x2.set_data(t, self._lv['x2'])
        self.l_mdot.set_data(t, self._lv['mdot'])
        self.ax3.relim();  self.ax3.autoscale_view()
        self.ax3b.relim(); self.ax3b.autoscale_view()

        # draw() renders synchronously — the frame actually appears now.
        # flush_events() then pushes it to the screen immediately,
        # bypassing any remaining Tk event queue delay.
        self._canvas.draw()
        self._canvas.flush_events()

    def _finish(self, html_out, Imp):
        self._reset()
        self._pct_var.set(100)
        self._phase_var.set('COMPLETE')
        self._status_var.set(
            f'Done!  Total impulse = {Imp:.1f} kN·s  '
            f'— opening dashboard…')
        if html_out and os.path.exists(html_out):
            webbrowser.open('file:///' + html_out.replace('\\', '/'))

    def _reset(self):
        self.running = False
        self._run_btn.config(state='normal')
        self._stop_btn.config(state='disabled')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    CATNAPGui(root)
    root.mainloop()