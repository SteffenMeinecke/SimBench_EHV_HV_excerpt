import pytest
import os
import tempfile
import pandas as pd
import pandapower as pp

import SimBench_EHV_HV_excerpt as sbe


def test_combined_profile_fcts():
    # input data
    profiles = {
        "load.p_mw": pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
        "sgen.q_mvar": pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
    }
    test_file1 = os.path.join(tempfile.gettempdir(), "test_profile1.h5")
    test_file2 = os.path.join(tempfile.gettempdir(), "test_profile2.h5")

    # --- store to file (from dict)
    sbe.toolbox.store_profiles_to_hdf5_file(profiles, test_file1)

    # load data from hdf5 file into net
    net1 = pp.create_empty_network()
    sbe.toolbox.add_profiles_from_h5_to_net(net1, test_file1, True, False)

    pd.testing.assert_frame_equal(net1.profiles["load.p_mw"], profiles["load.p_mw"])
    pd.testing.assert_frame_equal(net1.profiles["sgen.q_mvar"], profiles["sgen.q_mvar"])

    # --- store to file (from HDFStore - probably seldom needed)
    with pd.HDFStore(test_file1) as hdf_store:
        sbe.toolbox.store_profiles_to_hdf5_file(hdf_store, test_file2)

    # test that file2 equals file1
    net2 = pp.create_empty_network()
    sbe.toolbox.add_profiles_from_h5_to_net(net2, test_file2, True, False)
    for key in net2.profiles.keys():
        pd.testing.assert_frame_equal(net2.profiles[key], net1.profiles[key])

    # load data with time_steps
    net3 = pp.create_empty_network()
    sbe.toolbox.add_profiles_from_h5_to_net(net3, test_file1, [2, 1], False)
    pd.testing.assert_frame_equal(net3.profiles["load.p_mw"], profiles["load.p_mw"].loc[[1, 2]])
    pd.testing.assert_frame_equal(net3.profiles["sgen.q_mvar"], profiles["sgen.q_mvar"].loc[[1, 2]])

    os.remove(test_file1)
    os.remove(test_file2)


if __name__ == "__main__":
    # pytest.main([__file__])  # run all tests

    test_combined_profile_fcts()

    pass
