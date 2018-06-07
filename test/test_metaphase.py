# -*- coding: utf-8 -*-
import sys
sys.path.append("../kt_simul")
from kt_simul.core.simul_spindle import Metaphase
from kt_simul.core import parameters
import numpy as np
import pandas as pd
import pytest


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


class TestMk:

    TEST_DATA =[
        [1, 22.222, 50.00, 222.222, 600, 0.1, 0.02, 26.3888],
        [2, 44.444, 100.00, 444.444, 1200, 0.1, 0.02, 52.2222],
        [3, 66.666, 150.00, 666.666, 1800, 0.1, 0.02, 78.0555],
        [4, 88.888, 200.00, 888.888, 2400, 0.1, 0.02, 103.8888]
    ]
    PARAM_TREE = None
    MEASURE_TREE = None

    @classmethod
    def setup_class(cls):
        cls.PARAM_TREE = parameters.get_default_paramtree()
        cls.MEASURE_TREE = parameters.get_default_measuretree()

    def test_mk_related_params(self):
        """
        Verify that only 7 parameters will be changed when calling parameters.reduce_params
        """
        param_tree = TestMk.PARAM_TREE.copy()
        before = param_tree.params
        parameters.reduce_params(param_tree, TestMk.MEASURE_TREE)
        after = param_tree.params
        assert get_diffs(before, after).shape == (7, 2)

    @pytest.mark.parametrize("mk,exp_kappa_c,exp_kappa_k,exp_muc,exp_muk, exp_d0, exp_vmz, exp_fmz", TEST_DATA)
    def test_metaphase(self, mk, exp_kappa_c, exp_kappa_k, exp_muc, exp_muk, exp_d0, exp_vmz, exp_fmz):
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

        param_tree = TestMk.PARAM_TREE.copy()
        param_tree['Mk'] = mk

        parameters.reduce_params(param_tree, TestMk.MEASURE_TREE)
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
        print("all teared down")
