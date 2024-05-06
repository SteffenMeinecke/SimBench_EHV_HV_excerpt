import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import simbench as sb

from SimBench_EHV_HV_excerpt.toolbox.run_custom_timeseries import run_custom_timeseries

try:
    import pandaplan.core.pplog as logging
    paco_imported = True
except ImportError:
    import logging
    paco_imported = False

logger = logging.getLogger(__name__)


def set_bus_zones(net):
    boundary_buses = {68}  # Borken
    boundary_buses |= {106}  # North-West of Borken
    boundary_buses |= {38}  # between Hamburg - Bremen
    boundary_buses |= {240, 1662}  # below Hamburgs in direction of Bremen
    boundary_buses |= {2}  # Hannover West direction
    boundary_buses |= {1678, 256}  # Hameln
    boundary_buses |= {48, 1482}  # Magdeburg
    boundary_buses |= {12, 60, 1490}  # Nunsdorf South West of Berlin
    boundary_buses |= {14, 1446}  # Neunhagen South East of Berlin
    # boundary_buses |= {100}  # Test
    # boundary_buses |= {434, 1464}  # North of Hamburg

    zone_boundary_buses = {66}  # NVP Tennet-50Hertz above the Harz, still belongs to Tennet
    zone_boundary_buses |= {8}  # NVP Tennet-50Hertz above the Harz, still belongs to Tennet

    # inner_buses = list(net.trafo.hv_bus.loc[net.trafo.vn_lv_kv == 110]) + [
    #     104, 320, 1902, 1894, 36, 1472, 3079]

    inner_buses = set(top.connected_component(top.create_nxgraph(net), net.trafo.hv_bus.loc[
        net.trafo.vn_lv_kv == 110].iat[0], notravbuses=boundary_buses)) | set(
        top.connected_component(top.create_nxgraph(net), 104, notravbuses=boundary_buses)) | \
        set(top.connected_component(top.create_nxgraph(net), 1902, notravbuses=boundary_buses)) | \
        set(top.connected_component(top.create_nxgraph(net), 1442, notravbuses=boundary_buses)) | \
        set(top.connected_component(top.create_nxgraph(net), 1492, notravbuses=boundary_buses)) | \
        set(top.connected_component(top.create_nxgraph(net), 254, notravbuses=boundary_buses)) | \
        boundary_buses
    net.bus["zone"] = 20
    net.bus.loc[list(inner_buses), "zone"] = 18
    net.bus.loc[list(boundary_buses), "zone"] = 19
    return boundary_buses, zone_boundary_buses, inner_buses


def reduce_ehv(net, time_steps, boundary_buses, zone_boundary_buses, inner_buses):

    # --- determine boundary_branches
    boundary_branches = dict()
    # determine boundary_branches["f_lines"] (lines which are connected to boundary_buses via
    # from_bus)
    boundary_branches["f_lines"] = net.line.index[net.line.from_bus.isin(
        boundary_buses) & ~net.line.to_bus.isin(boundary_buses | inner_buses)]

    # determine boundary_branches["t_lines"]
    boundary_branches["t_lines"] = net.line.index[net.line.to_bus.isin(
        boundary_buses) & ~net.line.from_bus.isin(boundary_buses | inner_buses)]

    # determine boundary_branches["hv_trafos"] (trafos which are connected to boundary_buses via
    # hv_bus)
    boundary_branches["hv_trafos"] = net.trafo.index[net.trafo.hv_bus.isin(
        boundary_buses) & ~net.trafo.lv_bus.isin(boundary_buses | inner_buses)]

    # determine boundary_branches["t_trafos"]
    boundary_branches["lv_trafos"] = net.trafo.index[net.trafo.lv_bus.isin(
        boundary_buses) & ~net.trafo.hv_bus.isin(boundary_buses | inner_buses)]

    assert not len(boundary_branches["hv_trafos"])
    assert not len(boundary_branches["lv_trafos"])
    del boundary_branches["hv_trafos"]
    del boundary_branches["lv_trafos"]
    f_line_buses = net.line.from_bus.loc[boundary_branches["f_lines"]]
    t_line_buses = net.line.to_bus.loc[boundary_branches["t_lines"]]

    # --- run ts
    output_vals = [("line", "p_from_mw"), ("line", "q_from_mvar"),
                   ("line", "p_to_mw"), ("line", "q_to_mvar")]
    logger.info("reduce_ehv timeseries started.")
    if paco_imported:
        res = run_custom_timeseries(net, time_steps, "numba", None, output_vals=output_vals,
                                    include_bus_pq_results=False)
    else:
        res = run_custom_timeseries(net, time_steps, "pp", None, output_vals=output_vals)
    logger.info("reduce_ehv timeseries finished.")

    # remove unused res data:
    for power in [("p", "mw"), ("q", "mvar")]:
        for direction in ["from", "to"]:
            key = "res_line." + power[0] + "_" + direction + "_" + power[1]
            res[key] = res[key][boundary_branches[direction[0] + "_lines"]]

    # --- select Tennet, 50Hertz region
    net = pp.select_subnet(net, inner_buses, include_results=True, keep_everything_else=True)

    # add slack to net should be removed by selecting subnet
    if net.ext_grid.shape[0] + net.gen.slack.sum() == 0:
        idx = net.gen.p_mw.idxmax()
        net.gen.loc[idx, "slack"] = True
        net.gen.loc[idx, "slack_weight"] = 1

    # --- add loads at boundary_buses with pq values from timeseries
    f_name = [f"ExtL_{idx}" for idx in boundary_branches["f_lines"]]
    f_loads = pp.create_loads(net, f_line_buses, res["res_line.p_from_mw"].abs().max(),
                              res["res_line.q_from_mvar"].abs().max(), name=f_name)
    net.load.loc[f_loads, "profile"] = f_name
    t_name = [f"ExtL_{idx}" for idx in boundary_branches["t_lines"]]
    t_loads = pp.create_loads(net, t_line_buses, res["res_line.p_to_mw"].abs().max(),
                              res["res_line.q_to_mvar"].abs().max(), name=t_name)
    net.load.loc[t_loads, "profile"] = t_name
    net.profiles["load.p_mw"] = pd.concat([
        net.profiles["load.p_mw"], pd.DataFrame(
            res["res_line.p_from_mw"].values, columns=f_loads, index=list(time_steps)),
        pd.DataFrame(
            res["res_line.p_to_mw"].values, columns=t_loads, index=list(time_steps))],
        axis=1).fillna(0)  # fillna only for the case that 'time_steps' is smaller than the profiles
    net.profiles["load.q_mvar"] = pd.concat([
        net.profiles["load.q_mvar"], pd.DataFrame(
            res["res_line.q_from_mvar"].values, columns=f_loads, index=list(time_steps)),
        pd.DataFrame(
            res["res_line.q_to_mvar"].values, columns=t_loads, index=list(time_steps))],
        axis=1).fillna(0)  # fillna only for the case that 'time_steps' is smaller than the profiles

    # remove unused profiles data -> reduce (stored) data
    sb.filter_unapplied_profiles_pp(net, named_profiles=False)

    # --- set zone info - default is: (wbb == True) & (ehv_grids == 2)
    net.bus.loc[net.bus.subnet.str.contains("HV1"), "zone"] = 3
    net.bus.loc[net.bus.subnet.str.contains("HV2"), "zone"] = 4
    ehv_buses = set(net.bus.index[net.bus.vn_kv > 110])
    tennet_buses = set(top.connected_component(top.create_nxgraph(net), net.trafo.hv_bus.loc[
        net.trafo.vn_lv_kv == 110].iat[0], notravbuses=zone_boundary_buses)) & ehv_buses
    net.bus.loc[list(tennet_buses), "zone"] = 1
    # ext_grids == 2:
    net.bus.loc[list(ehv_buses - tennet_buses), "zone"] = 2
    # wbb == True:
    net.bus.zone.loc[net.bus.subnet.isin(["EHV1_HV1", "EHV1_HV2"])] = 0
    net.bus.loc[list(zone_boundary_buses), "zone"] = 0
    return net


def pre_manipulation_simbench_data(net):
    net.measurement = net.measurement.drop(net.measurement.index)
    del net["loadcases"]
    net.gen["min_p_mw"] = 0
    net.ext_grid["min_p_mw"] = 0
    net.sgen.drop(net.sgen.index[np.isclose(net.sgen.sn_mva, 0)], inplace=True)

    # kritische interne Leitungen # SimBench change!
    pp.change_std_type(net, 289, "LineType_8")  # -> Vereineinhalbfachung
    pp.change_std_type(net, 409, "LineType_10")  # -> Vervierfachung; 409, 425 bei Hameln
    pp.change_std_type(net, 425, "LineType_10")  # -> Vervierfachung; 409, 425 bei Hameln
    # -> Verdopplung; 417, 418 von Hamburg süd-ost Richtung Schwerin & Braunschweig
    pp.change_std_type(net, 418, "LineType_2")
    pp.change_std_type(net, 440, "LineType_8")  # -> Vereineinhalbfachung
    # -> Verdreifachung; Hannover-Ost nach Hannover Ost (220kV)
    pp.change_std_type(net, 579, "LineType_8")
    pp.change_std_type(net, 591, "LineType_1")  # -> Verdoppelung; 591 Hildesheim
    # -> Verdoppelung; 668, 669, 829 Hamburg - Elmshorn (NW von HH)
    pp.change_std_type(net, 668, "LineType_2")
    # -> Verdoppelung; 668, 669, 829 Hamburg - Elmshorn (NW von HH)
    pp.change_std_type(net, 669, "LineType_2")
    # -> Verdoppelung; 668, 669, 829 Hamburg - Elmshorn (NW von HH)
    pp.change_std_type(net, 829, "LineType_10")
    pp.change_std_type(net, 833, "LineType_8")  # -> Vereineinhalbfachung
    pp.change_std_type(net, 834, "LineType_8")  # -> Vereineinhalbfachung

    # external line change (to ensure convergence):
    net.line.parallel.loc[[824]] += 1
    net.line.parallel.loc[[11, 14, 32, 79, 100, 169, 190, 191, 192, 260, 315, 362, 363,
                           400, 450, 508, 593, 614, 665, 714, 754, 768, 808, 824]] += 1
    net.line.parallel.loc[[841]] += 1
    net.line.parallel.loc[[192, 227, 343, 768]] += 1
    net.line.parallel.loc[[39, 273, 838, 840]] += 1

    repl_ext_grid_by_gen_slack_weight_consideration(net)
    net.gen.slack.loc[[338, 344]] = np.array([False, True])
    net.gen.loc[338:339, "slack_weight"] = 0  # don't change p_mw of inner_buses gens.
    net.gen.controllable = net.gen.controllable.fillna(True)
    net.profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    use_opt_gen_vm = True
    if not use_opt_gen_vm:
        net.gen.vm_pu -= 0.05
    else:
        vm_pus = np.array([1.04308119, 1.04308119, 1.04308119, 1.04308119, 1.04308119,
                           1.04308119, 1.04308119, 1.04308119, 1.05856413, 1.05856413,
                           1.05253209, 1.05253209, 1.05253209, 1.05253209, 1.03611401,
                           1.03611401, 1.03936023, 1.04245460, 1.04245460, 1.04245460,
                           1.02235582, 1.01893757, 1.01893757, 1.01893757, 1.01893757,
                           1.01893757, 1.01893757, 1.01893757, 1.05059631, 1.03829218,
                           1.04741656, 1.02132639, 1.03936023, 1.03936023, 1.03936023,
                           1.03936023, 1.03936023, 1.03936023, 1.05243669, 1.05243669,
                           1.05555109, 1.04556988, 1.04556988, 1.02410671, 1.02410671,
                           1.04424325, 1.04424325, 1.04377559, 1.03715545, 1.03734276,
                           1.03715545, 1.03715545, 1.05063900, 1.01456952, 1.02250545,
                           1.04161462, 1.03775429, 1.01733415, 1.04146177, 1.04146177,
                           1.04231918, 1.04231918, 1.03985332, 1.03985332, 1.02049256,
                           1.02049256, 1.03756733, 1.02323092, 1.03125920, 1.02323092,
                           1.02158759, 1.02158759, 1.05348828, 1.05348828, 1.05348828,
                           1.02398963, 1.01570330, 1.03858866, 1.03858866, 1.01653085,
                           1.04204282, 1.04204282, 1.04204282, 1.02911289, 1.03810519,
                           1.03810519, 1.01023110, 1.04680427, 1.03811657, 1.04319049,
                           1.04254992, 1.04319049, 1.02385991, 1.04192358, 1.04192358,
                           1.04192358, 1.04177000, 1.04231918, 1.04231918, 1.05565292,
                           1.05565292, 1.01586818, 1.01586818, 1.01751838, 1.04173756,
                           1.04173756, 1.04774612, 1.04774612, 1.03916593, 1.02568986,
                           1.03734276, 1.03734276, 1.04161462, 1.01635788, 1.05082034,
                           1.05243669, 1.06235777, 1.06235777, 1.02323092, 1.04147194,
                           1.04147194, 1.04147194, 1.04147194, 1.02200277, 1.04299848,
                           1.04299848, 1.02091458, 1.02410671, 1.02220406, 1.02220406,
                           1.02220406, 1.02598917, 1.03466970, 1.01963290, 1.01963290,
                           1.01963290, 1.01963290, 1.01963290, 1.04192358, 1.04192358,
                           1.04192358, 1.05518502, 1.05565292, 1.06130742, 1.01586818,
                           1.01586818, 1.04173756, 1.03871512, 1.02568986, 1.04424325,
                           1.04424325, 1.04424325, 1.04424325, 1.06346105, 1.04741656,
                           1.03229039, 1.03595042, 1.03734276, 1.03673365, 1.03673365,
                           1.05043556, 1.04553442, 1.05388212, 1.03192469, 1.02805671,
                           1.02805671, 1.01456952, 1.01456952, 1.01456952, 1.04474577,
                           1.04926755, 1.03913452, 1.03032253, 1.04348216, 1.01733415,
                           1.01733415, 1.01799849, 1.01799849, 1.01799849, 1.01799849,
                           1.03777554, 1.03775429, 1.01475924, 1.04452692, 1.03535838,
                           1.03535838, 1.04245460, 1.04245460, 1.03985332, 1.03985332,
                           1.03947609, 1.01998950, 1.04133430, 1.03756733, 1.05243669,
                           1.05243669, 1.03125920, 1.04740174, 1.04740174, 1.03091275,
                           1.04008674, 1.04563364, 1.04261408, 1.04261408, 1.05094979,
                           1.05094979, 1.02409826, 1.03858866, 1.03858866, 1.04999707,
                           1.02091458, 1.02267374, 1.02132639, 1.01881233, 1.01881233, 1.04833519,
                           1.04833519, 1.03777554, 1.04348216, 1.02410671, 1.02410671,
                           1.02410671, 1.02410671, 1.02073881, 1.02073881, 1.02073881,
                           1.02073881, 1.02073881, 1.02193856, 1.07548798, 1.02834965,
                           1.05172625, 1.05172625, 1.03810519, 1.03810519, 1.03985126,
                           1.04177000, 1.04177000, 1.04177000, 1.04177000, 1.04177000,
                           1.04177000, 1.01799849, 1.03881345, 1.03881345, 1.03881345,
                           1.03881345, 1.03881345, 1.01282486, 1.01282486, 1.01282486,
                           1.01282486, 1.02488784, 1.03811657, 1.04563364, 1.04563364,
                           1.04754806, 1.03715545, 1.04133430, 1.04254992, 1.05172625,
                           1.04319049, 1.06104825, 1.06104825, 1.04614353, 1.0583450,
                           1.03518843, 1.01952551, 1.05043556, 1.04563364, 1.05518502,
                           1.05518502, 1.04377559, 1.04377559, 1.04377559, 1.04377559,
                           1.04377559, 1.04377559, 1.02091458, 1.02611034, 1.05388212,
                           1.04774612, 1.02611034, 1.03871512, 1.03289328, 1.03289328,
                           1.06427718, 1.03715545, 1.05043556, 1.04999707, 1.04161462,
                           1.03750724, 1.04614353, 1.05418802, 1.03091275, 1.05059631,
                           1.02200277, 1.02267374, 1.04895391, 1.02410671, 1.05172625,
                           1.01282486, 1.02055456, 1.00787993, 1.01963290, 1.04556988,
                           1.05043556, 1.04177000, 1.04177000, 1.04245460, 1.03916593,
                           1.04128966, 1.02207236, 1.04154757, 1.05051905, 1.05495474,
                           1.01742778, 1.04307668, 1.01793940, 1.03962442, 1.04281992,
                           1.04265537, 1.04184448, 1.01856843, 1.01816489, 1.04330767,
                           1.04209849, 1.06711330, 1.04271660, 1.05191503, 1.04257838,
                           1.04214861, 1.03437193, 1.03988165, 1.03909077, 1.04133430,
                           1.00620265, 1.01595459, 1.02834435, 1.05418802, 1.07666378,
                           1.06587579, 1.02073881, 1.05565292, 1.05269471]) - 0.01
        net.gen["vm_pu"] = vm_pus

    keys = list(net.profiles.keys())
    for (elm, col) in keys:
        net.profiles[f"{elm}.{col}"] = net.profiles.pop((elm, col))

    boundary_buses, zone_boundary_buses, inner_buses = set_bus_zones(net)
    return boundary_buses, zone_boundary_buses, inner_buses


def repl_ext_grid_by_gen_slack_weight_consideration(net):
    # --- replace ext_grids by generators to use distributed slacks
    if net.ext_grid.shape[0]:
        if (net.gen.slack_weight > 0).any():
            logger.warning("net.gen.slack_weight are overwritten.")
        net.gen["slack_weight"] = 0
        ext_grid_to_replace = net.ext_grid.index
        new_gens = pp.replace_ext_grid_by_gen(net, ext_grid_to_replace, add_cols_to_keep=[
            "profile", "phys_type", "type", "slack_weight", "sn_mva", "voltLvl", "subnet"])
        net.gen.loc[new_gens, "p_mw"] = net.gen.loc[new_gens, "max_p_mw"]
        net.gen.slack.loc[new_gens[0]] = True
