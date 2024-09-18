##!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# Modified from Pyfrac M-vertex solution


class PennyShape_Mvertex:
    def __init__(self, Eprime: float, Q0: float, muPrime: float):
        """
        M-vertex solution for penny-shaped hydraulic fractures.
        Args:
        Eprime (float):       -- plain strain elastic modulus.
        Q0 (float):           -- injection rate.
        muPrime (float):      -- 12*viscosity.
        """
        self.Eprime = Eprime
        self.Q0 = Q0
        self.muPrime = muPrime

        # Table 4, n=4
        self.gamma_0 = 0.6976
        return

    def scaling(self, t: float):
        """
        Returns the scalings for the M-vertex solution.
        Args:
            t (float): time.
        Returns:
            R_star (float): rupture radius scaling
            w_star (float): opening scaling
            p_star (float): net pressure scaling
        """
        #  Eq. 27
        episilon = (self.muPrime / (self.Eprime * t)) ** (1 / 3)
        # Eq. 27
        Lm = (self.Eprime ** (1 / 9) * self.Q0 ** (1 / 3) * t ** (4 / 9)) / (
            self.muPrime ** (1 / 9)
        )

        # Eq. 13
        p_star = episilon * self.Eprime
        # Eq. 12
        w_star = episilon * Lm
        # Eq. 14
        r_star = Lm

        return r_star, w_star, p_star

    def time_at_rupture_radius(self, R):
        t = (2.24846 * R ** (9 / 4) * self.muPrime ** (1 / 4)) / (
            self.Eprime ** (1 / 4) * self.Q0 ** (3 / 4)
        )
        return t

    def velocity_at_time(self, t):
        v = (
            (4.0 / 9)
            * (t ** (-5 / 9))
            * (
                (0.6976 * self.Eprime ** (1 / 9) * self.Q0 ** (1 / 3))
                / self.muPrime ** (1 / 9)
            )
        )
        return v

    def rupture_radius_at_time(self, t):
        R = (
            0.6976 * self.Eprime ** (1 / 9) * self.Q0 ** (1 / 3) * t ** (4 / 9)
        ) / self.muPrime ** (1 / 9)
        return R

    def opening_profile_at_time(self, col_pts: np.ndarray, t: float) -> np.ndarray:
        """
        col_pts: np.ndarray[:, 2]
        """
        nelems = col_pts.shape[0]
        R = self.rupture_radius_at_time(t)
        rho = (
            (col_pts[:, 0]) ** 2 + (col_pts[:, 1]) ** 2
        ) ** 0.5 / R  # normalized distance from center
        actvElts = np.where(rho <= 1)[0]  # active cells (inside fracture)
        # temporary variables to avoid recomputation
        var1 = -2 + 2 * rho[actvElts]
        var2 = 1 - rho[actvElts]

        w = np.zeros((nelems,))
        w[actvElts] = (
            (1 / (self.Eprime ** (2 / 9)))
            * 0.6976
            * self.Q0 ** (1 / 3)
            * t ** (1 / 9)
            * self.muPrime ** (2 / 9)
            * (
                1.89201 * var2 ** (2 / 3)
                + 0.000663163
                * var2 ** (2 / 3)
                * (35 / 9 + 80 / 9 * var1 + 38 / 9 * var1**2)
                + 0.00314291
                * var2 ** (2 / 3)
                * (
                    455 / 81
                    + 1235 / 54 * var1
                    + 2717 / 108 * var1**2
                    + 5225 / 648 * var1**3
                )
                + 0.000843517
                * var2 ** (2 / 3)
                * (
                    1820 / 243
                    + 11440 / 243 * var1
                    + 7150 / 81 * var1**2
                    + 15400 / 243 * var1**3
                    + (59675 * var1**4) / 3888
                )
                + 0.102366
                * var2 ** (2 / 3)
                * (1 / 3 + 13 / 3 * (-1 + 2 * rho[actvElts]))
                + 0.237267
                * (
                    (1 - rho[actvElts] ** 2) ** 0.5
                    - rho[actvElts] * np.arccos(rho[actvElts])
                )
            )
        )
        return w

    def pressure_profile_at_time(self, col_pts: np.ndarray, t: float) -> np.ndarray:
        nelems = col_pts.shape[0]
        R = self.rupture_radius_at_time(t)

        p = np.zeros((nelems,))
        rho = (
            (col_pts[:, 0]) ** 2 + (col_pts[:, 1]) ** 2
        ) ** 0.5 / R  # normalized distance from center
        actvElts = np.where(rho <= 1)[0]  # active cells (inside fracture)

        p[actvElts] = (
            0.0931746
            * self.Eprime ** (2 / 3)
            * self.muPrime ** (1 / 3)
            * (
                -2.20161
                + 8.81828 * (1 - rho[actvElts]) ** (1 / 3)
                - 0.0195787 * rho[actvElts]
                - 0.171565 * rho[actvElts] ** 2
                - 0.103558 * rho[actvElts] ** 3
                + (1 - rho[actvElts]) ** (1 / 3) * np.log(1 / rho[actvElts])
            )
        ) / (t ** (1 / 3) * (1 - rho[actvElts]) ** (1 / 3))
        if np.isinf(p[actvElts]).any():
            p[p == np.NINF] = min(p[p != np.NINF])
            p[p == np.Inf] = max(p[p != np.inf])

        return p


# %%
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root


@dataclass(frozen=False)
class HFShearModelAxiSym:
    """
    3D Axi Symm HF and Shear Model
    """

    # Elasticity
    E: float
    nu: float
    sig0: float
    tau0: float
    # Flow
    S: float
    w0: float
    mu: float
    # Interface
    fs: float
    kn: float
    # Injection
    Q0: float

    p0: float = 0.0
    ks: float = 1e10

    def __post_init__(self):
        self.G = self.E / (2 * (1 + self.nu))
        self.Ep = self.E / (1 - self.nu**2)
        self.mup = 12 * self.mu
        self.alpha = self.w0**2 / (self.mup * self.S)
        self.sig0p = self.sig0 - self.p0
        self.skempton_coeff = 1 / (1 + self.S * self.w0 * self.kn)

        self.weff = self.w0 + self.sig0p / self.kn
        self.c = (self.weff**2 / (12 * self.mu)) / (1 / (self.kn * self.weff) + self.S)
        self.hf_model: PennyShape_Mvertex = PennyShape_Mvertex(
            self.Ep, self.Q0, self.mup
        )

    def hf_scaling(self, t: np.ndarray) -> tuple:
        L_m, w_scale, p_scale = self.hf_model.scaling(t)
        return L_m, w_scale, p_scale

    def dimensionless_storage(self, t: np.ndarray) -> np.ndarray:
        # S sigstar
        L_m, width_scale, p_scale = self.hf_scaling(t)
        return self.S * p_scale

    def dimensionless_stresschange(self, t: np.ndarray) -> np.ndarray:
        # sigstar / sig0p
        L_m, width_scale, p_scale = self.hf_scaling(t)
        return p_scale / self.sig0p

    def dimensionless_hydraulic_aperture(self, t: np.ndarray) -> np.ndarray:
        # w0 / width_scale
        L_m, width_scale, p_scale = self.hf_scaling(t)
        return self.w0 / width_scale

    def dimensionless_stress_change_spring(self, t: np.ndarray) -> np.ndarray:
        # sigstar / (wo  * kn)
        L_m, width_scale, p_scale = self.hf_scaling(t)
        return p_scale / (self.w0 * self.kn)

    def pressdrop_by_hflength(self, t: np.ndarray) -> tuple:
        """
        1 / beta  or (alpha / ldot) * (1 / l)
        """
        L = self.hf_model.rupture_radius_at_time(t)
        v = self.hf_model.velocity_at_time(t)
        return (self.alpha / v) * (1 / L)

    def critical_overpressure(self, t: np.ndarray) -> np.ndarray:
        """
        Critical overpressure for shear failure
        """
        qw = self.Q0 / (2 * self.w0)
        k = self.w0**2 / 12
        pc = qw * self.mu * np.sqrt(self.alpha * t) / (np.sqrt(np.pi) * k)
        return pc

    def storage_time(self, tguess=0.1, tol=1e-2) -> float:
        fun = lambda t: self.dimensionless_storage(t) - tol
        tc = root(fun, tguess).x
        return tc[0]

    def hydraulic_aperture_time(self, tguess=0.1, tol=1e-2) -> float:
        fun = lambda t: self.dimensionless_hydraulic_aperture(t) - tol
        tc = root(fun, tguess).x
        return tc[0]

    def stresschange_time(self, tguess=0.1, tol=1e-2) -> float:
        fun = lambda t: self.dimensionless_stresschange(t) - tol
        tc = root(fun, tguess).x
        return tc[0]

    def pressure_drop_time(self, tguess=0.1, tol=1e-2) -> float:
        fun = lambda t: self.pressdrop_by_hflength(t) - tol
        tc = root(fun, tguess).x
        return tc[0]

    def opening_time(self, tguess=0.1) -> float:
        """
        time for HF failure
        """
        # fun = lambda t: self.critical_overpressure(t) - self.sig0p
        # tc = root(fun, tguess).x
        # return tc
        qw = self.Q0 / (2 * self.w0)
        k = self.w0**2 / 12
        cr_stress = self.sig0p
        to = (cr_stress**2 * np.pi * k**2) / (qw**2 * self.mu**2 * self.alpha)
        return to

    def shear_time(self, tguess=0.1) -> float:
        """
        time for shear failure
        """
        # fun = lambda t: self.critical_overpressure(t) - self.sig0p
        # tc = root(fun, tguess).x
        # return tc
        qw = self.Q0 / (2 * self.w0)
        k = self.w0**2 / 12
        cr_stress = self.sig0p - self.tau0 / self.fs
        ts = (cr_stress**2 * np.pi * k**2) / (qw**2 * self.mu**2 * self.alpha)
        return ts

    def netpressurescalebysigop_time(self):
        tc = self.Ep**2 * self.mup / self.sig0p**3
        return tc
    
    def rbyrootct_time(self):
        tc = self.Ep**2 * self.Q0**6 / self.mup**3 / self.c**9
        return tc

    def wobywidthscale_time(self):
        tc = self.Ep**2 * self.w0**9 / self.mup**2 / self.Q0**3
        return tc

    def shear_by_open_ratio_axisym(self):
        """
        Eq. 7b   Pg. 386 Olesiak and Wnuk 1968

        return R_s / R_o

        p0 = tau0
        Y = fs * (sig0 - p0)
        """
        # lam = p0 / Y
        lam = self.tau0 / (self.fs * (self.sig0 - self.p0))
        # m = l / a
        m = np.sqrt(1 - lam**2)
        # R_s / R_o = a / l
        return 1 / m

    def shear_open_slip_profile_axisym_dugdale_model(self, r, l):
        """
        r: radial distance from center
        l : R opening front
        a : Rs shear front

        We have case 2
        Eq. 5   Pg. 386 Olesiak and Wnuk 1968

        E:Incomplete elliptic integral of the second kind
        F:Incomplete elliptic integral of the first kind (K in scipy)
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipeinc.html#scipy.special.ellipeinc

        rho = r / a
        p0 = tau0
        Y = fs * (sig0 - p0)
        a : Rs shear front
        """
        from scipy.special import ellipeinc as E
        from scipy.special import ellipkinc as F

        p0 = self.tau0
        Y = self.fs * (self.sig0 - self.p0)

        lam = p0 / Y
        # Eq 7b
        m = np.sqrt(1 - lam**2)

        # shear front, a in paper
        a = l / m
        print(a, l, m)


        # multpliplied by 2 as w in paper is half opening
        delstar = (8 * (1 - self.nu**2) * a * Y) / (np.pi * self.E)

        # rho = r / a
        rho = r / a

        #  Pg 394 Appendix A
        sinmu1 = np.sqrt((1 - m**2) / (1 - rho**2))
        mu1 = np.arcsin(sinmu1)

        sinmu2 = np.sqrt((1 - rho**2) / (1 - m**2))
        mu2 = np.arcsin(sinmu2)

        # Eq. 5a
        eq5a = lam * np.sqrt(1 - rho**2) - sinmu1 + m * E(mu1, rho / m) 

        # Eq. 5b
        eq5b = lam * np.sqrt(1 - rho**2) - sinmu2 + rho * E(mu2, m / rho) + ((m**2-rho**2)/rho) * F(mu2, m / rho)

        slip = np.zeros_like(rho)
        fl = rho < m
        notfl = np.logical_not(fl)
        slip[fl] = eq5a[fl]
        slip[notfl] = eq5b[notfl]

        fl = rho >= 1
        slip[fl] = 0.0


        return slip * delstar

    def slip_scale(self, l: float) -> tuple:
        """
        l: opening front
        """
        p0 = self.tau0
        Y = self.fs * (self.sig0 - self.p0)
        lam = p0 / Y
        # Eq 7b
        m = np.sqrt(1 - lam**2)
        # shear front, a in paper
        a = l / m
        # multpliplied by 2 as w in paper is half opening
        deltastar = (8 * (1 - self.nu**2) * a * Y) / (np.pi * self.E)
        return deltastar

    def cohesive_length_estimate(self, t: np.ndarray) -> np.ndarray:
        """
        estimate cohesive length, l_c
        """
        v = self.hf_model.velocity_at_time(t)
        return self.alpha / v

import json
import glob, re

class AxisymmHFShearSimulation:

    def __init__(self, foldername: str):
        self.read_mesh(foldername)
        self.read_parameters(foldername)
        self.read_timestep(foldername)
        self.calculate_fronts(self.soln)

    def read_mesh(self, folder):
        file = open(folder + "/Mesh.json", "r")
        mesh = json.load(file)
        coord = np.array(mesh["Coordinates"])
        conn = np.array(mesh["Connectivity"])
        nelems = int(mesh["Nelts"])
        dim = int(mesh["Dimension"])
        colPts = np.array((coord[1:, 0] + coord[0:-1, 0]) / 2.0)
        h_x = coord[1, 0] - coord[0, 0]
        coor1D = coord[:, 0]
        col_pts = np.zeros((nelems, 2))
        col_pts[:, 0] = colPts[:]

        self.col_pts = col_pts
        self.colPts = colPts
        self.hx = h_x
        self.coord  = coord
        self.coor1D = coor1D

    def read_parameters(self, folder):

        file = open(folder + "/Parameters.json", "r")
        params = json.load(file)
        E = params["Elasticity"]["Young"]
        nu = params["Elasticity"]["Nu"]
        G = E / (2 * (1 + nu))  # Shear modulus
        kernel = params["Elasticity"]["Kernel"]
        who = params["Flow"]["who"]
        mu = params["Flow"]["Viscosity"]
        S = params["Flow"]["Compressibility"]
        Q = params["Injection"]["Constant Rate"]
        sigo = params["Initial stress"][1]
        tauo = params["Initial stress"][0]
        po = params["Pore pressure"]
        fs = params["Friction coef"]
        kn = params["Normal spring"]
        ks = params["Shear spring"]

        sigoprime = sigo - po

        alpha = who**2 / (12 * S * mu)
        fluid_visc_prime = mu * 12

        sigoprime = sigo - po
        Dqout = sigoprime * who**3 / (Q * fluid_visc_prime)
        alphaout = Q / (alpha * who)

        Dalphaout = (Q / (alpha * who)) * (1 - Dqout)

        self.model = HFShearModelAxiSym(
            E=E,
            nu=nu,
            S=S,
            mu=mu,
            Q0=Q * (1 - 0 * 2 * np.pi * Dqout) * 1e0,
            fs=fs,
            sig0=sigo,
            tau0=tauo,
            w0=who,
            kn=kn,
            p0=po,
        )

    def calculate_fronts(self, soln):
        colPts = self.colPts
        h_x = self.hx
        sigoprime = self.model.sig0 - self.model.p0
        po = self.model.p0
        open_front = np.zeros(len(soln))
        shear_front = np.zeros(len(soln))
        pressure_front = np.zeros(len(soln))
        for i in range(len(soln)):
            trac = np.array(soln[i]["effective_tractions"]) / (-sigoprime)
            pressure = (np.array(soln[i]["pressure"]) - po) / sigoprime
            pres_colpts = (
                np.array(
                    [
                        0.5 * (soln[i]["pressure"][j] + soln[i]["pressure"][j + 1])
                        for j in range(colPts.shape[0])
                    ]
                )
                - po
            ) / sigoprime
            try:
                fl_press = np.where(pres_colpts < 1.74e-2)[0][0]
                pressure_front[i] = np.abs(colPts[fl_press])
            except:
                pressure_front[i] = 0.0
            try:
                # fl_open = np.where(pressure[10:] / sigo < 1-1e-5)[0][0]
                fl_open = np.where(trac[1::2] > 1e-8)[0][0]
                open_front[i] = np.abs(colPts[fl_open])
            except:
                open_front[i] = 0.0
            try:
                shear_front[i] = np.sum(soln[i]["Nyielded"]) * h_x
            except:
                shear_front[i] = 0.0
        self.tts = np.array([soln[i]["time"] for i in range(len(soln))], np.float_)
        self.open_front = open_front
        self.shear_front = shear_front
        self.pressure_front = pressure_front


    def read_timestep(self, folder):

        def get_key(fp):
            # pattern = r"(\d+\.\d+)\.json$"
            pattern = r"(\d+)\.json$"
            match = re.search(pattern, fp)
            return float(match.group(1))
        basename = "3DAxiSymmHF"
        files = sorted(glob.glob(folder + "/" + basename + "*.json"), key=get_key)
        soln = []

        for filename in files:
            file = open(filename, "r")
            soln.append(json.load(file))

        self.soln = soln

    def compute_flux(self):
        soln = self.soln
        colPts = self.colPts

        fluxlist = []
        flux_inj_list = []
        crack_vol_list = []
        crack_vol_anal_list = []
        crack_vol_elas_list = []
        crack_vol_who_list = []
        crack_vol_M_list = []
        com_flux_list = []
        w0_list = []
        w_list = []
        grad_list = []
        velout_list = []

        who = self.model.w0
        mu = self.model.mu
        Q = self.model.Q0
        S = self.model.S
        col_pts = self.col_pts

        for j in range(len(soln)):
            time = soln[j]["time"]
            pres_colpts = (
                np.array(
                    [
                        0.5 * (soln[j]["pressure"][i] + soln[j]["pressure"][i + 1])
                        for i in range(colPts.shape[0])
                    ]
                )
                - self.model.p0
            )
            pressure = np.array(soln[j]["pressure"])
            press_grad = np.gradient(pressure, self.coord[:, 0])
            dd = np.array(soln[j]["DDs"])
            dde = dd - np.array(soln[j]["DDs_plastic"])
            ddp = dd - dde
            l = self.open_front[j]
            open_front_index = int(l / self.hx) + 1
            R = colPts[open_front_index]
            grad = -press_grad[open_front_index]
            w = dd[2 * open_front_index + 1] + who
            w_list.append(w / who)
            flux = 2 * np.pi * R * grad * w**3 / ((12 * mu))
            velout = grad * w**2 / ((12 * mu))
            fluxlist.append(flux)
            com_flux_list.append(np.trapz(fluxlist, self.tts[: j + 1]))
            velout_list.append(velout)

            fl = 0
            R0 = colPts[fl]
            grad0 = -press_grad[fl]
            w0 = dd[2 * fl + 1] + 0 * who
            flux_inj = 2 * np.pi * R0 * grad0 * w0**3 / ((12 * mu) * Q)
            flux_inj_list.append(flux_inj)
            w0_list.append(w0 / who)

            crack_vol_M = np.trapz(
                2
                * np.pi
                * colPts[:open_front_index]
                * (dd[1 : 2 * open_front_index + 1 : 2] + 1 * who)
                * S
                * pres_colpts[:open_front_index],
                colPts[:open_front_index],
            )
            crack_vol_M_list.append(crack_vol_M)

            wanal = self.model.hf_model.opening_profile_at_time(col_pts, time)
            crack_vol_anal = np.trapz(2 * np.pi * colPts * (wanal), colPts) / (Q * time)
            crack_vol = np.trapz(
                2 * np.pi * colPts[:open_front_index] * (ddp[1 : 2 * open_front_index + 1 : 2]),
                colPts[:open_front_index],
            )
            crack_vol_elas = np.trapz(
                2 * np.pi * colPts[:open_front_index] * (dde[1 : 2 * open_front_index + 1 : 2]),
                colPts[:open_front_index],
            )
            crack_vol_who = np.trapz(
                2 * np.pi * colPts[:open_front_index] * (who), colPts[:open_front_index]
            )
            crack_vol_list.append(crack_vol)
            crack_vol_anal_list.append(crack_vol_anal)
            crack_vol_elas_list.append(crack_vol_elas)
            crack_vol_who_list.append(crack_vol_who)

        w0_list = np.array(w0_list)
        w_list = np.array(w_list)
        fluxlist = np.array(fluxlist)
        grad_list = np.array(grad_list)
        crack_vol_list = np.array(crack_vol_list)
        crack_vol_anal_list = np.array(crack_vol_anal_list)
        velout_list = np.array(velout_list)

        return fluxlist, crack_vol_list, velout_list



