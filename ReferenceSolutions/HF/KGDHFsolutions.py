#
# This file is part of PyFracX.
#
# Created by Brice Lecampion on 08.02.22.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2022.  All rights reserved.
# See the LICENSE.TXT file for more details.
#

# PLANE - STRAIN HYDRAULIC FRACTURE - Analytical solutions

import numpy as np
import scipy


### M- vertex
# viscosity dominated solution / zero leak off  - Newtonian fluid....
class KGD_Mvertex:
    def __init__(self, Ep=1.0, Qo=1.0, mu=1.0 / 12.0):
        # epsilon_ = (mup / (Ep t))^1/3
        # L_m = Qo^1/2 * Ep^(1/6) t^2/3   / mup^1/6
        self.Eprime_ = Ep
        self.Qo_ = Qo
        self.mup_ = 12.0 * mu
        self.gamma_0 = 0.61524

        self.B_ = 0.06858
        self.A_ = np.array(
            [
                1.61750,
                0.39650,
                1.00297 * 1.0e-2,
                1.06179 * 1e-2,
                -2.97947 * 1.0e-3,
                -0.74268 * 1.0e-3,
                -1.34675 * 1.0e-3,
            ]
        )

    def half_length(self, t=np.array([1.0], np.float_)):
        L_m = (
            (self.Qo_**0.5)
            * (self.Eprime_ ** (1.0 / 6.0))
            * t ** (2.0 / 3.0)
            / ((self.mup_) ** (1.0 / 6))
        )
        L = self.gamma_0 * L_m
        return L

    def om_star(self, j, xi):
        return ((1.0 - xi * xi) ** (2.0 / 3.0)) * (
            scipy.special.gegenbauer(2 * j - 2, 2.0 / 3.0 - 1 / 2, monic=0)(xi)
        )  # n=2j-2  alpha=2/3-1/2)

    def omega(self, xi):
        # omegabar=omega / gamma  - self simlar profile
        sq_aux = np.sqrt(1.0 - xi * xi)
        if xi == 0.0:
            om_star_star = 4.0 * sq_aux
        else:
            om_star_star = 4.0 * sq_aux + 2 * (xi * xi) * np.log(
                np.abs((1 - sq_aux) / (1 + sq_aux))
            )
        om_star = np.zeros(7)
        for j in range(7):
            om_star[j] = (self.om_star(j + 1, xi)) * self.A_[j]
        #
        om_bar = self.B_ * om_star_star + np.sum(om_star)
        return om_bar * self.gamma_0

    def width(self, t, xi_list):
        L_m = (
            (self.Qo_**0.5)
            * (self.Eprime_ ** (1.0 / 6.0))
            * t ** (2.0 / 3.0)
            / ((self.mup_) ** (1.0 / 6))
        )
        eps_m = (self.mup_ / (self.Eprime_ * t)) ** (1.0 / 3)
        width_scale = eps_m * L_m
        om = np.zeros(len(xi_list))
        for i in range(len(xi_list)):
            om[i] = self.omega(xi_list[i])
        return width_scale * om

    def pi_star_1(self, xi):  # for j=1
        return (
            (2.0 / 3)
            / (2.0 * np.pi)
            * (scipy.special.beta(0.5, 2.0 / 3))
            * (scipy.special.hyp2f1(0.5 - 2.0 / 3, 1, 0.5, xi * xi))
        )

    def pi_star(self, j, xi):  # for j>1
        alpha = 2.0 / 3
        pref = (
            ((2 * alpha - 1) * (2 * j - 1) / (4.0 * np.pi * (j - 1 + alpha)))
            * (scipy.special.beta(0.5 - j, alpha + j))
            * (
                alpha
                * xi
                * xi
                * (scipy.special.hyp2f1(5 / 2.0 - j - alpha, j, 3.0 / 2, xi * xi))
                - 0.5 * scipy.special.hyp2f1(3.0 / 2 - j - alpha, j - 1, 0.5, xi * xi)
            )
        )
        return pref

    def Pi_m(self, xi):
        pi_star_star = 2.0 - np.pi * (np.abs(xi))
        pi_st = np.zeros(7)
        pi_st[0] = self.pi_star_1(xi)
        for j in range(6):
            pi_st[j + 1] = self.pi_star(j + 2, xi)
        for j in range(7):
            pi_st[j] = pi_st[j] * self.A_[j]
        return np.sum(pi_st) + self.B_ * pi_star_star

    def pressure(self, t, xi_list):
        eps_m = (self.mup_ / (self.Eprime_ * t)) ** (1.0 / 3)
        p_scale = eps_m * self.Eprime_
        pim = np.zeros(len(xi_list))
        for i in range(len(xi_list)):
            pim[i] = self.Pi_m(xi_list[i])
        return p_scale * pim
