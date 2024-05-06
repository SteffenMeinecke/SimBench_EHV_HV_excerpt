import os
import pandas as pd
import pandapower as pp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def write_ts_results_to_json(results_dict, path, time_steps=None, ignore_keys=None,
                             overwrite=False):
    """Writes timeseries values from results_dict to json files.

    Parameters
    ----------
    results_dict : dict[str, DataFrame]
        results to store
    path : str
        path where to store the results
    time_steps : iterable, optional
        can be given to reduce the amount of results that should be stored, by default None
    ignore_keys : iterable, optional
        can be given to reduce the amount of results that should be stored, by default None
    overwrite : bool, optional
        whether an existing folder to store jsons in it can be overwritten, by default False
    """
    if ignore_keys is None:
        keys = set(results_dict.keys())
    else:
        keys = set(results_dict.keys()) - set(ignore_keys)
    keys_without_dot = [key for key in keys if "." not in key]
    if len(keys_without_dot):
        logger.warning("In results_dict exist keys without '.' in between res_elm and variable: " +
                       str(keys_without_dot) + " These are ignored when writing to json files.")
    res_elm_cols = [tuple(key.split(".")) for key in keys if "." in key]
    if not len(res_elm_cols):
        logger.info("There are no res_elm_cols in results_dict.")
        return

    if not os.path.exists(path):
        os.makedirs(path)
    elif not overwrite:
        for i, (subdir, _, files) in enumerate(os.walk(path)):
            if i == 0:
                continue
            if os.path.split(subdir) in pp.pp_elements(res_elements=True) and \
                    any([f[-5:] == ".json" for f in files]):
                logger.warning("write_ts_results_to_json() is aborted since res_*folder with "
                               "jsons exist already.")
                return

    for (res_elm, col) in res_elm_cols:
        folder = os.path.join(path, res_elm)
        if not os.path.exists(folder):
            os.makedirs(folder)
        loc_None(results_dict["%s.%s" % (res_elm, col)], time_steps).to_json(
            os.path.join(folder, col + ".json"))


def read_ts_results_from_json(path, ignore=None, include_only=None, time_steps=None,
                              add_empty=False):
    """
    Reads a folder of folders with jsons including timeseries results.
    Returns a dict of keys such as "res_bus.vm_pu" and values with time_steps in the index and
    element index in columns.
    """
    ignore = pp.ensure_iterability(ignore) if ignore is not None else []
    include_only = pp.ensure_iterability(include_only) if include_only is not None else None
    results_dict = dict()
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file[-5:] == ".json" and "param" not in file and "net" not in file:
                key = "%s.%s" % (os.path.basename(subdir), file[:-5])
                if (include_only is None or key in include_only) and key not in ignore:
                    df = pd.read_json(os.path.join(subdir, file))
                    if df.shape[1]:
                        results_dict[key] = loc_None(df, time_steps)
                    elif add_empty:
                        results_dict[key] = pd.DataFrame(index=time_steps)
    return results_dict


def loc_None(df, idx):
    if idx is None:
        return df
    else:
        return df.loc[idx]
