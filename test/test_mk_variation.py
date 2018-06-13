# -*- coding: utf-8 -*-
import sys

from kt_simul.core.simul_spindle import Metaphase
from kt_simul.io.simuio import SimuIO

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


class TestMetaphase:

    # TEST_KD_DATA = [
    #     [1, 22.222, 50.00, 222.222, 600, 0.1, 0.02, 26.3888]
    # ]
    PARAM_TREE = None
    MEASURE_TREE = None
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    SPAN = 50

    @classmethod
    def setup_class(cls):
        cls.PARAM_TREE = parameters.get_default_paramtree()
        cls.PARAM_TREE['span'] = cls.SPAN
        cls.MEASURE_TREE = parameters.get_default_measuretree()

        parameters.reduce_params(cls.PARAM_TREE, cls.MEASURE_TREE,
                                 force_parameters=[])

    def test_metaphase(self):
        random_state = np.random.RandomState(2230)
        # These two lines are required to launch more than one simulation in a single run in a for loop for example
        # state = randomState.get_state()
        # randomState.set_state(state)

        meta = Metaphase(verbose=True, paramtree=TestMetaphase.PARAM_TREE, measuretree=TestMetaphase.MEASURE_TREE,
                         prng=random_state,
                         initial_plug='random')
        meta.simul()

        run_id = 1

        SimuIO(meta).save("%s_steps_%s.h5" % (TestMetaphase.SPAN, run_id))

        meta.show().savefig("%s_traj%i.png" % (TestMetaphase.SPAN, run_id))

    # def test_solve(self):
    #     """
    #         This is the core to be tested
    #     :return:
    #     """
    #     # cdef solve(self):
    #     # cdef np.ndarray[DTYPE_t, ndim = 1] X, C, pos_dep
    #     # cdef np.ndarray[DTYPE_t, ndim = 2] A, B
    #     X = KD.get_state_vector()
    #     A = KD.calc_A()
    #     B = KD.B_mat
    #     C = KD.calc_C()
    #     pos_dep = np.dot(B, X) + C
    #     KD.speeds = np.linalg.solve(A, -pos_dep)
        

    @classmethod
    def teardown_class(cls):
        TestMetaphase.PARAM_TREE = None
        TestMetaphase.MEASURE_TREE = None
        teardown_log(TestMetaphase)
