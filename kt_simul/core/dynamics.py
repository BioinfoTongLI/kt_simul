import numpy as np
import pandas as pd


class Model:

    def __init__(self, structure, dt=1):

        self.structure = structure
        self.dt = dt
        self.n_points = len(self.structure.points)
        self.Amat = np.zeros((self.n_points * 3, self.n_points * 3))
        self.Bvect = np.zeros(self.n_points * 3)

    def solve(self):
        speeds = np.linalg.solve(self.Amat, -self.Bvect)
        for i, c in enumerate(coords):
            self.structure.point_df['v'+c] = speeds[i::3]
            self.structure.point_df[c] += speeds[i::3] * self.dt
        self.structure.update_geometry()


def viscous(model, point, mu):
    ''' Viscous drag with coeff mu on point '''
    model.Amat[point.idxs, point.idxs] -= np.eye(3) * mu


def dashpot(model, link, mu):

    block = mu * link.outer
    model.Amat[link.idxs_ii] += block
    model.Amat[link.idxs_ij] -= block
    model.Amat[link.idxs_jj] -= block
    model.Amat[link.idxs_ji] += block


def spring(model, link, kappa, d_eq=0):

    idx_i, idx_j = link.idx
    idxs_i = slice(idx_i*3, idx_i*3 + 3)
    idxs_j = slice(idx_j*3, idx_j*3 + 3)

    F = -kappa * (link.length - d_eq) * link.unit
    model.Bvect[link.idxs_i] += F
    model.Bvect[link.idxs_j] -= F


def dampedspring(model, link, mu, kappa, d_eq=0):

    dashpot(model, link, mu)
    spring(model, link, kappa, d_eq)


def contraction(model, link, F):

    model.Bvect[link.idxs_i] += F * link.unit
    model.Bvect[link.idxs_j] -= F * link.unit


def linear_fv(model, link, F_stall, v_max, gamma=-1):

    dashpot(model, link, F_stall / v_max)
    contraction(model, link, gamma * F_stall)


class SpindleModel(Model):

    def __init__(self, spindle):
        self.params = spindle.params
        Model.__init__(self, spindle, self.params['dt'])
        self.prng = spindle.prng
        self.spindle = spindle
        L0 = self.params['L0']
        Mk = int(self.params['Mk'])
        self.duration = self.params['span']
        self.dt = self.params['dt']
        self.num_steps = int(self.duration / self.dt)
        self.time_invariantA()
        self.simulation_done = False
        self.anaphase = False

    def time_invariantA(self):

        viscous(self.spindle.spbL, self.params['mus'])
        viscous(self.spindle.spbR, self.params['mus'])

        for ch in spindle.chromosomes:
            viscous(ch.cen_A, self.params['muco'])
            viscous(ch.cen_A, self.params['muco'])
            for ps in ch.cen_A.plugsites:
                viscous(ps, self.params['muco']/100)  # to avoid singluar mat.
            for ps in ch.cen_B.plugsites:
                viscous(ps, self.params['muco']/100)  # to avoid singluar mat.
        self.A0mat = self.Amat.copy()

    def update_AB(self):
        self.Amat = self.A0mat
        self.Bmat = np.zeros(self.n_points)
        mz = self.spindle.links[(self.spbR.idx, self.spbL.idx)]
        linear_fv(self, mz, self.params['Fmz'], self.params['Vmz'], gamma=1)
        for ch in spindle.chromosomes:
            chromatin = self.spindle.links[(ch.cen_A.idx, ch.cen_B.idx)]
            dampedspring(self, chromatin,
                         self.params['muc'],
                         self.params['kappa_c'],
                         self.params['d0'])
            for ps in ch.cen_A.plugsites:
                self.plugsite_forces(ps)
            for ps in ch.cen_B.plugsites:
                self.plugsite_forces(ps)

    def plugsite_forces(self, ps):
        kt = self.spindle.links[(ps.centromere, ps)]
        dampedspring(self, kt,
                     self.params['muk'],
                     self.params['kappa_k'], 0)
        if ps.plug_state == 0:
            return
        if ps.plug_state == -1:
            ktMT = self.spindle.links[(self.spindle.spbL, ps)]
        elif ps.plug_state == 1:
            ktMT = self.spindle.links[(self.spindle.spbR, ps)]
        linear_fv(self, ktMT, 1, 1, -1)

    def one_step(self, step):
        if not self.anaphase:
            for plugsite in self.all_plugsites:
                plugsite.plug_unplug()
        self.upate_AB()
        self.solve()
        self.spindle.register_history()
