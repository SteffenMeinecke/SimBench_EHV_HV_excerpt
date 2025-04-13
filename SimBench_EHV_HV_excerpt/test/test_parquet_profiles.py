import pytest
import tempfile
import shutil
import pandas as pd
import pandapower as pp

import SimBench_EHV_HV_excerpt as sbe


def test_combined_profile_fcts():
    # input data
    profiles = {
        "load.p_mw": pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
        "sgen.q_mvar": pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
    }
    temp_dir = tempfile.mkdtemp()

    # --- store to file (from dict)
    sbe.toolbox.store_profiles_to_parquet_files(profiles, temp_dir)

    # load data from parquet files into net
    net1 = pp.create_empty_network()
    sbe.toolbox.add_profiles_from_parquet_to_net(net1, True, False, profiles_folder=temp_dir)

    pd.testing.assert_frame_equal(net1.profiles["load.p_mw"], profiles["load.p_mw"])
    pd.testing.assert_frame_equal(net1.profiles["sgen.q_mvar"], profiles["sgen.q_mvar"])

    # load data with time_steps
    net3 = pp.create_empty_network()
    sbe.toolbox.add_profiles_from_parquet_to_net(net3, [2, 1], False, profiles_folder=temp_dir)
    pd.testing.assert_frame_equal(net3.profiles["load.p_mw"], profiles["load.p_mw"].loc[[1, 2]])
    pd.testing.assert_frame_equal(net3.profiles["sgen.q_mvar"], profiles["sgen.q_mvar"].loc[[1, 2]])

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # pytest.main([__file__])  # run all tests

    test_combined_profile_fcts()

    pass
