import pytest
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

import SimBench_EHV_HV_excerpt as sbe


def check_element_number(net, expected_numbers):
    for et, number in expected_numbers.items():
        assert len(net[et]) == number


def check_profiles(net, time_steps):
    keys = {"load.p_mw", "load.q_mvar", "sgen.p_mw", "gen.p_mw", "storage.p_mw", "gen.vm_pu"}
    assert "profiles" in net.keys()
    assert set(net.profiles.keys()) == keys
    for key in keys:
        is_stor = key == "storage.p_mw"
        assert is_stor or not len(net.profiles[key].index.symmetric_difference(pd.Index(
            time_steps)))
        assert is_stor or not len(net.profiles[key].columns.symmetric_difference(net[key.split(
            ".")[0]].index))


def default_element_numbers():
    return {
        "bus": 261,
        "load": 244,
        "sgen": 181,
        "gen": 72,
        "line": 399,
        "trafo": 24,
    }


def test_element_numbers():
    net = sbe.SimBench_for_phd()
    check_element_number(net, default_element_numbers())


def test_SO_zones():
    def _expected_zones(net, ehv_zones):
        zones = net.bus.zone.value_counts()

        # zones exist as expected
        assert not len(zones.index.symmetric_difference(pd.Index(ehv_zones + [3, 4])))
        # with relevant number of buses per zone
        assert (zones > 30).all()

        # zones 1, 2 are EHV zones
        for zone in [1, 2]:
            assert (net.bus.vn_kv.loc[net.bus.zone == zone] >= 145).all()
        # zones 3, 4 are EHV zones
        for zone in [3, 4]:
            assert (net.bus.vn_kv.loc[net.bus.zone == zone] < 145).all()


    # --- net with two EHV grids
    net1 = sbe.SimBench_for_phd()
    _expected_zones(net1, [1, 2])

    # --- net with only one EHV grid
    net2 = sbe.SimBench_for_phd(ehv_grids=1)
    _expected_zones(net2, [1])


def test_function_params():
    net = sbe.SimBench_for_phd()

    # wbb
    net1 = sbe.SimBench_for_phd(wbb=True)
    check_element_number(net1, default_element_numbers())

    # merged_same_bus_gens
    net2 = sbe.SimBench_for_phd(wbb=True, merged_same_bus_gens=True)
    nets_expected_numbers = default_element_numbers()
    del nets_expected_numbers["sgen"]
    del nets_expected_numbers["gen"]
    check_element_number(net2, nets_expected_numbers)
    for et in ["gen", "sgen"]:
        assert 0 < len(net2[et]) < len(net1[et])

    # time_steps
    time_steps = [0, 1, 2]
    net3 = sbe.SimBench_for_phd(time_steps=time_steps)
    check_element_number(net3, default_element_numbers())
    check_profiles(net3, time_steps)

    # control
    time_steps = [5, 7]
    net4 = sbe.SimBench_for_phd(time_steps=time_steps, control="LocalCtrl")
    assert len(net4.controller)
    check_element_number(net4, default_element_numbers())
    check_profiles(net4, time_steps)


def test_powers():
    """Running time series calculations to check expected results
    """
    # select a few time steps
    time_steps = [0, 24, 48, 72, 96, 120, 144, 168]

    # get the net
    net = sbe.SimBench_for_phd(time_steps=time_steps)

    # check input profiles
    expected_p_load = np.array([
        7793.4, 11350.67, 11566.52, 10841.58, 10661.13, 10560.55, 7984.62, 8820.05])
    expected_p_gen = np.array([
        2990.56, 5720.07, 6300.70, 6692.37, 6874.37, 7341.96, 4938.36, 6286.56])
    expected_p_sgen = np.array([
        5002.54, 5924.16, 5552.16, 4396.12, 4004.95, 3435.95, 3214.30, 2752.20])
    assert np.allclose(net.profiles["load.p_mw"].sum(axis=1), expected_p_load, atol=0.1)
    assert np.allclose(net.profiles["gen.p_mw"].sum(axis=1), expected_p_gen, atol=0.1)
    assert np.allclose(net.profiles["sgen.p_mw"].sum(axis=1), expected_p_sgen, atol=0.1)

    # apply controllers
    no_const_ctrls = sbe.add_control_strategy(net, "LocalCtrl")
    sb.apply_const_controllers(net, net.profiles, "no_const_ctrls")

    # define tracked variables
    res_cols = [(f"res_bus", col) for col in ["vm_pu", "p_mw"]]
    res_cols += [(f"res_gen", col) for col in ["vm_pu", "p_mw", "q_mvar"]]
    res_cols += [(f"res_sgen", col) for col in ["p_mw", "q_mvar"]]
    res_cols += [(f"res_line", col) for col in ["loading_percent", "pl_mw"]]
    res_cols += [(f"res_trafo", col) for col in ["loading_percent", "pl_mw"]]
    res_keys = [f"{res_et}.{col}" for res_et, col in res_cols]

    # define OutputWriter
    ow = pp.timeseries.OutputWriter(net, time_steps)
    for res_et, col in res_cols:
        ow.log_variable(res_et, col)

    # run timeseries
    pp.timeseries.run_timeseries(net, time_steps=time_steps, max_iter=200)
    res = ow.output

    # results exist and have expected keys and correct dimension
    assert all([key in res.keys() for key in res_keys])
    assert all([len(res[key]) == len(time_steps) for key in res_keys])
    assert res["res_bus.vm_pu"].shape[1] == len(net.bus)

    # only slack is 338
    slack = net.gen.index[net.gen.slack]
    assert np.allclose(res["res_gen.p_mw"].drop(columns=slack),
                       net.profiles["gen.p_mw"].drop(columns=slack), atol=1e-4)

    # check losses
    t_dep_losses1 = res["res_line.pl_mw"].sum(axis=1) + res["res_trafo.pl_mw"].sum(axis=1)
    t_dep_losses2 = -res["res_bus.p_mw"].sum(axis=1)
    assert np.allclose(t_dep_losses1, t_dep_losses2, atol=1e-2)
    expected_losses = np.array([203.95, 299.22, 290.13, 249.22, 219.9, 218.92, 169.21, 220.14])
    assert np.allclose(t_dep_losses1, expected_losses, atol=0.1)

    # TODO: check that LocalCtrl are doing something


@pytest.mark.skip(reason="test is not finished")
def test_h5():
    pass  # TODO


if __name__ == "__main__":
    # pytest.main([__file__])  # run all tests

    # test_element_numbers()
    # test_SO_zones()
    # test_function_params()
    # test_powers()
    test_h5()

    pass
