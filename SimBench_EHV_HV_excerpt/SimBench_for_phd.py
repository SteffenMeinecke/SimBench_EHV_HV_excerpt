import typing
import os
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

from SimBench_EHV_HV_excerpt import home, data_path
from SimBench_EHV_HV_excerpt.toolbox import *

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def SimBench_for_phd(
        time_steps:typing.Any = False,
        from_json:bool = True,
        merged_same_bus_gens:bool = False,
        control:str|None = None,
        wbb:bool = False,
        ehv_grids:int = 2,
        **kwargs
    ) -> pp.pandapowerNet:
    """Returns the pandapower net that is the main outcome of this repository: an excerpt of
    SimBench' EHV grid model including the two HV grids as well. This grid is used in the
    Dissertation 'TSO-DSO Coordinated Distributed Optimizations in Grid Operations Considering
    System Operator Sovereignty' and two referenced papers.

    Parameters
    ----------
    time_steps : typing.Any, optional
        list of time steps to be included in net.profiles. Special values:
            True -> time steps of a complete year are included;
            False -> no net.profiles is provided;
            By default False
    from_json : bool, optional
        whether the net should be loaded from json file or, if False,
        created by calculations, by default True
    merged_same_bus_gens : bool, optional
        whether to merge generators and corresponding profiles connected to the same buses.
        This is needed for current REI implementation, by default False
    control : str | None, optional
        name for control strategy to add to the net, e.g. "localCtrl" which corresponds to the model
        of state-of-the-art grid operation, by default None
    wbb : bool, optional
        If True, buses of neutral zone are placed between net zone areas (grid connection option
        "neutral buses", cf. Fig. 3.8). If False, all buses are assigned to a zone of a system
        operator (Options 2 "neutral branches" or 3 "complete assignment"), by default False
    ehv_grids : int, optional
        number of ehv zone areas. Possible are 1 and 2. 1 is provided only for experimental codings,
        by default 2

    Other Parameters
    ----------------
    fixed_p : bool, optional
        If True, the active power limits of the generators must not be reduced to the current active
        power value since generators active power are not considered as variables in OPFs,
        by default True

    Returns
    -------
    pp.pandapowerNet
        _description_
    """
    if control not in ["LocalCtrl", "QofV", "QofV_old", "NoControl", None]:
        logger.warning(f"{control=} is unknown and thus ignored.")
        control = None

    # --- only read from json file -----------------------------------------------------------------
    if from_json:
        net = pp.from_json(os.path.join(data_path, "net.json"))
        add_profiles_from_parquet_to_net(net, time_steps, False, kwargs.get("profiles_folder", None))

    # --- create net -------------------------------------------------------------------------------
    else:
        # start creating the net
        code = "1-EHVHV-mixed-all-0-no_sw"
        net = sb.get_simbench_net(code)
        boundary_buses, zone_boundary_buses, inner_buses = pre_manipulation_simbench_data(net)

        # get variable "time_steps"
        if time_steps is True:
            time_steps = range(net.profiles["load.p_mw"].shape[0])
        elif time_steps != False:
            downcast_profiles(net.profiles)
            reduce_profiles_by_time_steps(net.profiles, time_steps)

        # -- timeseries run incl. distr. slack and gen.vm_pu profiles determination
        if not time_steps:
            del net["profiles"]

        else:  # --- run timeseries incl. distributed_slack and gen.vm_pu
            if 0:  # writing net to json may lead to memory error. Then you can restart the script
                # setting this if clause to False
                ds_idx = consider_distr_slack(net, initial_slack_distribution=False,
                                              first_step_factor=2/3)
                q_range_diff = net.gen.max_q_mvar - net.gen.min_q_mvar

                # add some "safety gap" around timeseries for resulting gen.vm_pu by enforce_q_lims
                net.gen["max_q_mvar"] -= q_range_diff*0.01
                net.gen["min_q_mvar"] += q_range_diff*0.01

                res = run_custom_timeseries(net, time_steps, None, enforce_q_lims=True,
                    add_output_vals=[("gen", "p_mw"), ("gen", "vm_pu")])

                # remove back the "safety gap"
                net.gen["max_q_mvar"] += q_range_diff*0.01
                net.gen["min_q_mvar"] -= q_range_diff*0.01

                # remove back the distr slack controller
                net.controller = net.controller.drop(ds_idx)

            else:  # don't run time consuming time series but use precalculated gen ts results
                jsons_path = os.path.join(data_path, "net_creation_timeseries_results")
                res = read_ts_results_from_json(jsons_path)

            # add gen.vm_pu and gen.p_mw to profiles
            gen_vm_key = "res_gen.vm_pu" if "res_gen.vm_pu" in res.keys() else "gen.vm_pu"
            net.profiles["gen.vm_pu"] = res[gen_vm_key][net.gen.index]
            gen_p_key = "res_gen.p_mw" if "res_gen.p_mw" in res.keys() else "gen.p_mw"
            net.profiles["gen.p_mw"] = res[gen_p_key][net.gen.index]
            del res

            # downcast profiles
            downcast_profiles(net.profiles)

            # -- reduce the net to the relevant part
            logger.info("reduce_ehv() starts.")
            net = reduce_ehv(net, time_steps, boundary_buses, zone_boundary_buses, inner_buses)

            # set first time_step to power columns and remove results from time series
            set_time_step(net, time_steps[0])
            pp.clear_result_tables(net)

        for elm in ["bus", "res_bus"]:
            net[elm] = net[elm].sort_index()

        # change parallel appearance
        sb.convert_parallel_branches(
            net, multiple_entries=False, elm_to_convert=["trafo"],
            exclude_cols_from_parallel_finding=net.trafo.columns.difference(
                {"hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv", "std_type"}))

        # add origin_id
        for elm in pp.pp_elements():
            net[elm]["origin_id"] = net[elm].name

        # controllable sgens
        net.sgen["controllable"] = ~net.sgen.profile.str.startswith("mv_")
        ctrl_sgens = net.sgen.index[net.sgen["controllable"]]

        # q constraint information (VDE 4130 & 4120)
        net["sgen"]["qcurve1"] = ""
        sgens4120 = pd.Index(sb.voltlvl_idx(net, "sgen", 3)).intersection(ctrl_sgens)
        sgens4130_220 = net.sgen.index[(net.bus.vn_kv.loc[net.sgen.bus] > 145).values &
                                       (net.bus.vn_kv.loc[net.sgen.bus] < 255).values &
                                       net.sgen.controllable]
        sgens4130_380 = net.sgen.index[(net.bus.vn_kv.loc[net.sgen.bus] > 255).values &
                                       net.sgen.controllable]
        net["sgen"].loc[sgens4120, "qcurve1"] = "4120_v2"
        net["sgen"].loc[sgens4130_220, "qcurve1"] = "4130_220_v2"
        net["sgen"].loc[sgens4130_380, "qcurve1"] = "4130_380_v2"

        # add slack_weight column to sgens
        net.sgen["slack_weight"] = 0.
        net.sgen["slack_weight"] = net.sgen["slack_weight"].astype(float)

        # add name to net
        net.name = "SimBench_EHV_HV_excerpt"

        # --- store resulting net and profiles to json and h5 file
        _store_files_to_desktop(net)

    # --- create end -------------------------------------------------------------------------------

    if merged_same_bus_gens:
        # do merge of same bus gens
        pp.merge_same_bus_generation_plants(net, gen_elms=["ext_grid", "gen"])
        pp.merge_same_bus_generation_plants(net, gen_elms=["sgen"])

    # set sgen limits according to Q(P) constraint of VDE 4130 & 4120
    set_sgen_limits(net, fixed_p=kwargs.get("fixed_p", True))
    for col in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        net.sgen[col] = net.sgen[col].astype(float)

    for elm in ["ext_grid", "gen", "sgen"]:
        assert not np.any(np.isclose(net[elm].sn_mva.values, 0))

    # --- consider wbb
    if not wbb:
        is_bb = net.bus.zone == 0
        net.bus.loc[is_bb & (net.bus.subnet == "EHV1_HV1"), "zone"] = 1
        net.bus.loc[is_bb & (net.bus.subnet == "EHV1_HV2"), "zone"] = 2
        net.bus.loc[is_bb & (net.bus.subnet == "EHV1"), "zone"] = 2

    # --- consider ehv_grids
    if ehv_grids not in [1, 2]:
        raise ValueError(f"'ehv_grids' is implemented only for [1, 2], not for {ehv_grids=}.")
    elif ehv_grids == 1:
        if wbb:
            net.bus.loc[(net.bus.zone == 0) & (net.bus.subnet == "EHV1"), "zone"] = 1
        net.bus.loc[net.bus.zone == 2, "zone"] = 1

    # fix some dtypes
    for elm_type in pp.pp_elements():
        for col in net[elm_type].columns.intersection({
                "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"}):
            try:
                net[elm_type][col] = net[elm_type][col].astype(float)
            except:
                pass
        if "voltLvl" in net[elm_type].columns:
            net[elm_type].voltLvl = net[elm_type].voltLvl.astype("Int8")

    add_control_strategy(net, control)
    return net


def _store_files_to_desktop(net) -> None:
    # --- store net to json
    net_wo_profiles = deepcopy(net)
    del net_wo_profiles["profiles"]
    pp.to_json(net_wo_profiles, os.path.join(home, "Desktop", "net.json"))
    logger.info("net is stored to a json file at your desktop.")
    del net_wo_profiles

    # --- store profiles to h5
    profiles_folder = os.path.join(home, "Desktop", "profiles")
    dir_exists = os.path.exists(profiles_folder)
    store_profiles_to_parquet_files(net.profiles, profiles_folder)
    if dir_exists:
        logger.info("The profile folders at the desktop were refilled by parquet files.")
    else:
        logger.info("profiles were written to parquet files in the profiles folder on your Desktop.")


if __name__ == "__main__":

    net = SimBench_for_phd()  # from_json=True, time_steps=False, merged_same_bus_gens=False

    if False:
        net = SimBench_for_phd(from_json=False, time_steps=True)
        downcast_profiles(net.profiles)
        store_profiles_to_hdf5_file(net.profiles, os.path.join(home, "Desktop", "profiles.h5"))
