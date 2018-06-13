# -*- coding: utf-8 -*-
import sys
sys.path.append("../kt_simul")
from kt_simul.core import parameters
from .test_mk_variation import teardown_log
import numpy as np
from pytest import approx
import logging
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
from kt_simul.core.spindle_dynamics import KinetoDynamics
from kt_simul.core.components import Spindle, Spb, Chromosome, Centromere, PlugSite, Organite

def prepare_default_parameters():
    param_tree = parameters.get_default_paramtree()
    measure_tree = parameters.get_default_measuretree()

    parameters.reduce_params(param_tree, measure_tree, force_parameters=[])
    params_for_kd = param_tree.relative_dic
    # Reset explicitely the unit parameters to their
    # dimentionalized value
    params_for_kd['Vk'] = param_tree.absolute_dic['Vk']
    params_for_kd['Fk'] = param_tree.absolute_dic['Fk']
    params_for_kd['dt'] = param_tree.absolute_dic['dt']
    return params_for_kd


# global parameters
"""
    Since the class Organite, the parent of all physical objects, requires the KD object for its initialisation.
    Here we initialize it as a global variable.
"""
params_for_kd = prepare_default_parameters()
KD = KinetoDynamics(params_for_kd, initial_plug='random')
N = params_for_kd["N"]
Mk = params_for_kd["Mk"]
L0 = params_for_kd["L0"]
# for pytest
precision = 0.001


class TestOrganite:

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    ch = None
    dummy_ch_id = 99
    centromere = None
    dummy_centro_tag = 'A'
    plugsite = None
    dummy_plugsite_id = 98
    organite = None
    dummy_organite_init_pos = 0.65

    @classmethod
    def setup_class(cls):
        TestOrganite.logger.info("Setting up %s" % TestOrganite.__name__)
        cls.ch = Chromosome(KD.spindle, cls.dummy_ch_id)
        cls.centromere = Centromere(cls.ch, cls.dummy_centro_tag)
        cls.plugsite = PlugSite(cls.centromere, cls.dummy_plugsite_id)
        cls.organite = Organite(KD.spindle, cls.dummy_organite_init_pos)

    def test_chromosome_number(self):
        assert len(KD.chromosomes) == 3

    def test_spindle_pluginsite_number(self):
        assert len(KD.spindle.get_all_plugsites()) == 3 * 3 * 2

    def test_organite_span(self):
        assert TestOrganite.organite.num_steps == approx(1000, precision)

    def test_organite_traj(self):
        assert TestOrganite.organite.traj.shape == (1000, )
        assert TestOrganite.organite.traj[0] == approx(0.65, precision)

    def test_spb_position(self):
        assert KD.spbR.side == 1
        assert KD.spbR.pos == approx(0.3 / 2, precision)
        assert KD.spbL.side == approx(-1, precision)
        assert KD.spbL.pos == approx(-1 * (0.3 / 2), precision)

    def test_chromosome_initialisation(self):
        assert TestOrganite.ch.ch_id == 99
        assert TestOrganite.ch.id == 99
        assert TestOrganite.ch.correct_history.shape == (1000, 2)
        assert TestOrganite.ch.erroneous_history.shape == (1000, 2)

    def test_chromosome_correction(self):
        pass
        # TODO
        # assert ch.correct_history[0] == self.correct()

    def test_chromosome_erronous_initialisation(self):
        pass
        # TODO
        # self.erroneous_history[0] = self.erroneous()

    def test_centromere(self):
        assert TestOrganite.centromere.tag == 'A'

    def test_plugsite(self):
        assert TestOrganite.plugsite.site_id == 98

    @classmethod
    def teardown_class(cls):
        teardown_log(TestOrganite)
