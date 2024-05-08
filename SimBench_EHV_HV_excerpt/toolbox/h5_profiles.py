import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pandapower as pp

from SimBench_EHV_HV_excerpt import sb_excerpt_dir
from SimBench_EHV_HV_excerpt.toolbox.set_values_to_net import set_time_step

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
home = str(Path.home())
data_path = os.path.join(sb_excerpt_dir, "data")
profiles_file = os.path.join(data_path, "profiles.h5")


def check_file_existence(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"file {file} does not exist.")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"{file} is not file.")
    if not os.access(file, os.R_OK):
        raise PermissionError(f"There are no reading rights for file {file}.")


def add_profiles_from_h5_to_net(net:pp.pandapowerNet,
                                h5_file:str,
                                time_steps:bool|list[int]|np.ndarray|pd.Index,
                                always_set_time_step:bool) -> None:

    if time_steps is False:
        pass  # nothing to do

    elif time_steps is True:
        check_file_existence(h5_file)
        with pd.HDFStore(h5_file) as hdf:
            net.profiles = {slash_to_dot(key): hdf.get(key) for key in hdf.keys()}

        if always_set_time_step:
            first_profile_key = sorted(net.profiles.keys())[0]
            first_time_step = net.profiles[first_profile_key].index[0]
            logger.debug(f"{first_time_step=} has been taken from {first_profile_key=}.")
            set_time_step(net, first_time_step)

    else:
        if sorted(time_steps) != list(time_steps):
            logger.warning("net.profiles is provided with ascending time steps, although given "
                            "time_steps are not sorted.")
            time_steps = sorted(time_steps)

        check_file_existence(h5_file)
        with pd.HDFStore(h5_file) as hdf:
            net.profiles = {slash_to_dot(key): hdf.select(
                key, start=time_steps[0], stop=time_steps[-1]+1) for key in hdf.keys()}

        if (time_steps[-1] - time_steps[0]) != len(time_steps)+1:
            reduce_profiles_by_time_steps(net.profiles, time_steps)

        if always_set_time_step or time_steps[0] != 0:
            set_time_step(net, time_steps[0])


def store_profiles_to_hdf5_file(
        profiles:dict[str, pd.DataFrame]|pd.io.pytables.HDFStore, file:str, mode:str="w",
        format:str="table", data_columns:dict[str,list]|list|bool|None=None, **kwargs) -> None:
    """Generates a hdf5 file from a dictionary of DataFrames or a pandas HDFStore

    Parameters
    ----------
    profiles : dict[str, pd.DataFrame] | pd.io.pytables.HDFStore
        data to be written to a hdf5 file
    file : str
        path to the file to be written
    mode : str, optional
        Mode to use when opening the file, by default "w"
    format : str, optional
        format of writing; Possible values: [fixed, table], by default "table"
    data_columns : dict[str,list] | list | bool | None, optional
        columns to create as indexed data columns for on-disk queries, or True to use all
        columns. Applicable only to format='table'; by default None
    """

    with pd.HDFStore(file, mode=mode) as store:
        for key in profiles.keys():
            d_c = data_columns if not isinstance(data_columns, dict) else data_columns.get(key, None)
            store.put(key.replace(".", "/"), profiles[key], format=format, data_columns=d_c, **kwargs)


def reduce_profiles_by_time_steps(
        profiles:dict[str, pd.DataFrame],
        time_steps:list[int]|np.ndarray|pd.Index
        ) -> None:

    for key in profiles.keys():
        if profiles[key].shape[0]:
            idx = profiles[key].index.intersection(set(time_steps))
            profiles[key] = profiles[key].loc[idx]


def slash_to_dot(st:str) -> str:
    if st[0] == "/":
        st = st[1:]
    return st.replace("/", ".")


def SimBench_for_phd_hdf5_profiles(create_copy=True, **kwargs):
    if create_copy:  # needed for multi-threading and to avoid data loss of the original data
        temp = tempfile.mktemp()
        shutil.copyfile(profiles_file, temp)
        return pd.HDFStore(temp, mode="r+", **kwargs)
    else:
        return pd.HDFStore(profiles_file, mode="r+", **kwargs)
