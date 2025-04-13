import os
import shutil
from itertools import product
import numpy as np
import pandas as pd
import pandapower as pp

from SimBench_EHV_HV_excerpt import data_path
from SimBench_EHV_HV_excerpt.toolbox.set_values_to_net import set_time_step

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def check_file_existence(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"file {file} does not exist.")
    if not os.path.isfile(file):
        raise FileNotFoundError(f"{file} is not a file.")
    if not os.access(file, os.R_OK):
        raise PermissionError(f"There are no reading rights for file {file}.")
    if not os.access(file, os.W_OK):
        raise PermissionError(f"There are no writing rights for file {file}.")


def add_profiles_from_parquet_to_net(
        net:pp.pandapowerNet,
        time_steps:bool|list[int]|np.ndarray|pd.Index,
        always_set_time_step:bool,
        profiles_folder:str|None=None,
        **kwargs) -> None:
    """Reads time series profile data from parquet files and adds the data to net.profiles

    Parameters
    ----------
    net : pp.pandapowerNet
        net to be filled with profile data
    time_steps : bool | list[int] | np.ndarray | pd.Index
        time_steps that should be provided. If True, the whole year is provided, if False, no
        profile data is provided
    always_set_time_step : bool
        decides whether profiles data should always be set to element tables, e.g. net.sgen.p_mw,
        even if time_steps starts with 0 (in that case, setting the time step is usually not needed)
    profiles_folder : str | None, optional
        Folder with profiles data. If None, this repositories data_path is used, by default None
    """

    if time_steps is False or (not isinstance(time_steps, bool) and not len(time_steps)):
        return  # nothing to do

    folders, stored_time_steps = _folders_and_time_steps(profiles_folder=profiles_folder)

    folders = [folder for folder, time_stepss in zip(folders, stored_time_steps) if pd.Series(
        time_steps).isin(time_stepss).any()]

    filenames = os.listdir(folders[0])
    profiles = dict()
    for filename, folder in product(filenames, folders):
        file = os.path.join(folder, filename)
        key = filename.replace(".parquet", "")
        check_file_existence(file)
        if key not in profiles:
            profiles[key] = pd.read_parquet(file, **kwargs)
        else:
            profiles[key] = pd.concat([profiles[key], pd.read_parquet(file, **kwargs)])

    net.profiles = profiles

    if time_steps is not True:
        reduce_profiles_by_time_steps(net.profiles, time_steps)

    if always_set_time_step or (time_steps is not True and time_steps[0] != 0):
        set_time_step(net, time_steps[0])


def store_profiles_to_parquet_files(
        profiles:dict[str, pd.DataFrame], profiles_folder:str, except_permission_error:bool=False,
        **kwargs) -> None:
    """Generates parquet files for each DataFrame of a dictionary

    Parameters
    ----------
    profiles : dict[str, pd.DataFrame]
        data to be written to parquet files
    profiles_folder : str
        path to the folders of the parquet files to be written
    except_permission_error : bool, optional
        whether to raise an error if a file cannot be removed due to missing permission rights,
        by default False

    Optional Parameters
    -------------------
    kwars
        key word arguments for pandas' to_parquet() function
    """
    folders, time_steps = _folders_and_time_steps(profiles_folder=profiles_folder)

    # clean up folders
    for folder in folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                os.mkdir(folder)
            except PermissionError as e:
                if not except_permission_error:
                    raise PermissionError(e)
                else:
                    logger.info(e)
        elif os.path.exists(folder):
            os.remove(folder)
            os.makedirs(folder)
        else:
            os.makedirs(folder)

    # write files
    for key, df in profiles.items():
        for folder, time_stepss in zip(folders, time_steps):
            to_store = df.loc[df.index.isin(time_stepss)]
            to_store.to_parquet(os.path.join(folder, f"{key}.parquet"), **kwargs)


def _folders_and_time_steps(profiles_folder:str|None=None) -> tuple[list[str], list]:
    if profiles_folder is None:
        profiles_folder = data_path
    folders = [os.path.join(profiles_folder, "two_days"),
               os.path.join(profiles_folder, "rest_of_the_year")]
    stored_time_steps = [range(2*96), range(2*96, 366*96)]
    return folders, stored_time_steps


def reduce_profiles_by_time_steps(
        profiles:dict[str, pd.DataFrame],
        time_steps:list[int]|np.ndarray|pd.Index
        ) -> None:

    for key in profiles.keys():
        if profiles[key].shape[0]:
            idx = profiles[key].index.intersection(set(time_steps))
            profiles[key] = profiles[key].loc[idx]
