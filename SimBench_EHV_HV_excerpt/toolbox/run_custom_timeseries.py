from copy import deepcopy
import pandas as pd
import pandapower as pp
import simbench as sb

from SimBench_EHV_HV_excerpt.toolbox.set_values_to_net import set_time_step
from SimBench_EHV_HV_excerpt.toolbox.json_io import write_ts_results_to_json

try:
    from pandaplan.core.timeseries.run_profile_cython import run_static_profile
    paco_imported = True
except ImportError:
    paco_imported = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def run_custom_timeseries(net, time_steps, kernel, output_path:str|None,
                          run_control_fct=None, **kwargs):
    """Runs a time series with fast time series or with controllers and regular time series module.

    Parameters
    ----------
    net : pp.pandapowernet
        pandapowernet
    time_steps : iterable[int]
        time steps to run
    kernel : str
        "pp" or "numba"
    output_path : str
        where to store the results
    run_control_fct : callable, optional
        another function can be passed than the usual run_control function of pandapower,
        by default None

    Other Parameters
    ----------------
    profiles : dict[pandas.DataFrame] (columns:element index, index:time_step), optional
        if not given here, net.profiles is used

    output_vals : list[tuple[str, str]], optional
        such as [("line", "loading"), ("trafo", "i_hv_ka")]

    add_output_vals : list[tuple[str, str]], optional
        for kernel=="pp" e.g. [("gen", "p_mw"), ("gen", "vm_pu")]

    include_bus_pq_results : bool, optional
        by default True

    del_profiles : bool, optional
        try to reduce RAM, by default False

    no_const_ctrls : dict | None, optional
        exclude_elms_dict input for simbench.apply_const_controllers()

    Returns
    -------
    dict
        resulting output dict
    """
    if run_control_fct is not None and kernel != "pp":
        logger.warning("'kernel' has been changed to 'pp' to make use of run_control_fct.")
        kernel = "pp"
    if "profiles" not in kwargs.keys() and ("profiles" not in net.keys() or not isinstance(
        net.profiles, dict)):
        raise ValueError("No profiles are available.")
    profiles = kwargs.pop("profiles", getattr(net, "profiles", None))
    assert profiles is not None

    # define output values
    output_vals = kwargs.get("output_vals", default_outputs_from_kernel(kernel))
    if "add_output_vals" in kwargs.keys() and kernel == "pp":
        output_vals += kwargs["add_output_vals"]

    # --- kernel specific code
    if kernel == "numba":
        if not paco_imported:
            raise ModuleNotFoundError("Not open-source module pandaplan-core is needed for "
                                      "run_custom_timeseries(kernel='numba').")
        set_time_step(net, time_steps[0], abs_profiles=profiles)
        profile_arrays = {tuple(key.split(".")): val.loc[time_steps, net[key.split(".")[
            0]].index].values for key, val in profiles.items() if val.shape[0]}
        pp.runpp(net, **kwargs)
        res = run_static_profile(
            net, profile_arrays, output_vals, kernel="numba", num_threads=8,
            errors="ignore", tolerance_mva=1e-7,
            include_bus_pq_results=kwargs.pop("include_bus_pq_results", True), **kwargs)
        res = {f"res_{key[0]}.{key[1]}" if len(key) == 2 and key[0] in pp.pp_elements() \
               else key: val if not isinstance(val, pd.DataFrame) else val.set_index(pd.Index(
                time_steps)) for key, val in res.items()}
        if kwargs.get("drop_non_df_result_data", False):
            res = {key: val for key, val in res.items() if isinstance(val, pd.DataFrame)}
        if output_path is not None:
            write_ts_results_to_json(res, output_path)

    elif kernel == "pp":

        ctrls = deepcopy(net.controller.index)
        sb.apply_const_controllers(net, profiles, kwargs.get("no_const_ctrls", None))
        if kwargs.get("del_profiles", False):
            del net["profiles"]

        # define OutputWriter
        ow = pp.timeseries.OutputWriter(net, time_steps, output_path=output_path,
                                        output_file_type=".json")
        not_logged = list()
        for et, col in output_vals:
            if col in net[et].columns:
                ow.log_variable(et, col)
            else:
                not_logged.append((et, col))
        if len(not_logged):
            logger.warning(f"This output_vals could not be logged: {not_logged}")

        # run ts
        if run_control_fct is None:
            pp.timeseries.run_timeseries(net, time_steps=time_steps, **kwargs)
        else:
            pp.timeseries.run_timeseries(
                net, time_steps=time_steps, run_control_fct=run_control_fct, **kwargs)
        res = ow.output

        net.controller.drop(net.controller.index.difference(ctrls), inplace=True)

    else:
        raise ValueError(f"kernel '{kernel}' is unknown.")

    return res


def default_outputs_from_kernel(kernel):
    is_pp = kernel == "pp"
    bools = is_pp, False, is_pp, is_pp
    return default_outputs(*bools)


def default_outputs(res_prefix:bool, include_bus_vm_pu:bool, include_trafo_tap_pos:bool,
                    include_bus_elm_vq:bool=True):
    """
    Example
    -------
    >>> default_outputs(True, True, True)
    [('res_bus', 'vm_pu'), ('res_line', 'loading_percent'), ('res_line', 'p_from_mw'), ...]
    """
    prefix = "res_" if res_prefix else ""
    default_out_vals = [(f"{prefix}bus", "vm_pu")] if include_bus_vm_pu else []
    default_out_vals += [(f"{prefix}line", col) for col in bra2w_cols(et="line")]
    default_out_vals += [(f"{prefix}trafo", col) for col in bra2w_cols(et="trafo")]
    if include_trafo_tap_pos:
        default_out_vals.append(("trafo", "tap_pos"))
    if include_bus_elm_vq:
        default_out_vals += [(f"{prefix}gen", "vm_pu"), (f"{prefix}sgen", "q_mvar")]
    return default_out_vals


def bra2w_cols(et:str="line") -> list[str]:
    """
    Example
    -------
    >>> bra2w_cols()
    ['loading_percent', 'p_from_mw', 'p_to_mw', 'q_from_mvar', 'q_to_mvar', 'i_from_ka', 'i_to_ka']
    >>> bra2w_cols("trafo)
    ['loading_percent', 'p_hv_mw', 'p_lv_mw', 'q_hv_mvar', 'q_lv_mvar', 'i_hv_ka', 'i_lv_ka']
    """
    buses = pd.Series(pp.branch_element_bus_dict()[et]).str.split("_", expand=True)[0].tolist()
    powers = [("p", "mw"), ("q", "mvar"), ("i", "ka")]
    cols = ["loading_percent"]
    cols += [f"{power[0]}_{bus}_{power[1]}" for power in powers for bus in buses]
    return cols
