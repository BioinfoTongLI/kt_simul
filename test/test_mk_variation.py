# -*- coding: utf-8 -*-
import sys
sys.path.append("../kt_simul")
from kt_simul.core import parameters
import numpy as np
import pandas as pd
import pytest
import pyximport
import logging
pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
from kt_simul.core.spindle_dynamics import KinetoDynamics


def get_diffs(df1, df2):
    d1 = df1.sort_index(axis=1)
    d2 = df2.sort_index(axis=1)
    idx = np.where(d1 != d2)
    changed_from = d1.values[idx]
    changed_to = d2.values[idx]
    return pd.DataFrame({'from': changed_from, 'to': changed_to})


def assertion(param_tree, name, expected_value, precision=0.001):
    idx = np.where(param_tree.params["name"] == name)[0][0]
    assert param_tree.params.iloc[idx]["value"] == pytest.approx(expected_value, precision)


def teardown_log(cls):
    cls.logger.info("%s teared down" % cls.__name__)


class TestParameterProcessing:

    TEST_DEDUCE_DATA = [
        [1, 22.222, 50.00, 222.222, 600, 0.1, 0.02, 26.3888],
        [2, 44.444, 100.00, 444.444, 1200, 0.1, 0.02, 52.2222],
        [3, 66.666, 150.00, 666.666, 1800, 0.1, 0.02, 78.0555],
        [4, 88.888, 200.00, 888.888, 2400, 0.1, 0.02, 103.8888]
    ]
    PARAM_TREE = None
    MEASURE_TREE = None
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    @classmethod
    def setup_class(cls):
        cls.PARAM_TREE = parameters.get_default_paramtree()
        cls.MEASURE_TREE = parameters.get_default_measuretree()

    def test_mk_related_params(self):
        """
        Verify that only 7 parameters will be changed when calling parameters.reduce_params
        """
        param_tree = TestParameterProcessing.PARAM_TREE.copy()
        before = param_tree.params
        parameters.reduce_params(param_tree, TestParameterProcessing.MEASURE_TREE)
        after = param_tree.params
        assert get_diffs(before, after).shape == (7, 2)

    @pytest.mark.parametrize("mk,exp_kappa_c,exp_kappa_k,exp_muc,exp_muk, exp_d0, exp_vmz, exp_fmz", TEST_DEDUCE_DATA)
    def test_deduce_params(self, mk, exp_kappa_c, exp_kappa_k, exp_muc, exp_muk, exp_d0, exp_vmz, exp_fmz):
        """
        Test the inference of certain parameters in param_tree, including:
        17, kappa_c,    Cohesin spring constant
        18, kappa_k,    Chromatid spring constant
        19, muc,        Chromatid drag coefficient
        20, muk,        Kt structural friction coef.
        21, d0,         Kinetochore - kinetochore rest length
        23, Vmz,        Midzone force max. speed
        25, Fmz,        Midzone stall Force
        """

        param_tree = TestParameterProcessing.PARAM_TREE.copy()
        param_tree['Mk'] = mk

        parameters.reduce_params(param_tree, TestParameterProcessing.MEASURE_TREE)
        # param_tree.to_csv("test/%i.csv" % mk)
        assertion(param_tree, "kappa_c", exp_kappa_c)
        assertion(param_tree, "kappa_k", exp_kappa_k)
        assertion(param_tree, "muc", exp_muc)
        assertion(param_tree, "muk", exp_muk)
        assertion(param_tree, "Fmz", exp_fmz)
        # These two parameters should not be changed when changing Mk
        assertion(param_tree, "d0", exp_d0)
        assertion(param_tree, "Vmz", exp_vmz)

    @classmethod
    def teardown_class(cls):
        cls.PARAM_TREE = None
        cls.TEST_DATA = None
        cls.MEASURE_TREE = None
        teardown_log(TestParameterProcessing)


class TestKD:

    # TEST_KD_DATA = [
    #     [1, 22.222, 50.00, 222.222, 600, 0.1, 0.02, 26.3888]
    # ]
    PARAM_TREE = None
    MEASURE_TREE = None
    PARAMS_FOR_KD = None
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    @classmethod
    def setup_class(cls):
        cls.PARAM_TREE = parameters.get_default_paramtree()
        cls.MEASURE_TREE = parameters.get_default_measuretree()

        parameters.reduce_params(cls.PARAM_TREE, cls.MEASURE_TREE,
                                 force_parameters=[])
        cls.PARAMS_FOR_KD = cls.PARAM_TREE.relative_dic
        # Reset explicitely the unit parameters to their
        # dimentionalized value
        cls.PARAMS_FOR_KD['Vk'] = cls.PARAM_TREE.absolute_dic['Vk']
        cls.PARAMS_FOR_KD['Fk'] = cls.PARAM_TREE.absolute_dic['Fk']
        cls.PARAMS_FOR_KD['dt'] = cls.PARAM_TREE.absolute_dic['dt']

    # def test_kd(self):
    #     self.KD = KinetoDynamics(TestKD.PARAMS_FOR_KD, initial_plug='random')
    #     self.KD.anaphase = False
    #     # self.logger.info(""+str(self.KD.spindle))
    #     # self.logger.info("" + str(TestKD.METAPHASE.KD.all_plugsites))

    # def test_simul(self):
    #     TestKD.METAPHASE.simul()

    @classmethod
    def teardown_class(cls):
        TestKD.PARAM_TREE = None
        TestKD.MEASURE_TREE = None
        # TestKD.TEST_KD_DATA = None
        teardown_log(TestKD)
