import pandas as pd
import pandapower as pp


def SimBench_for_phd_obj_weights() -> pd.Series:
    """Returns a Series containing predefined weights applied in the Dissertation with the grid
    from SimBench_for_phd().

    Returns
    -------
    pandas.Series
        predefined weights
    """
    return pd.Series({1: 1.175832, 2: 2.383970, 3: 0.256285, 4: 0.183914}, name="predefined_weights")


def grid_parameters(net:pp.pandapowerNet, net_zones:list|None=None) -> tuple[pd.DataFrame]:
    """Returns two DataFrames with relevant data to define weights independent of operational
    variables, cf. Table A.5

    Parameters
    ----------
    net : pp.pandapowerNet
        net with zones which should be weighted against each other
    net_zones : list
        list of zones. If None, net_zones is filled by data from net.bus.zone, by default None

    Returns
    -------
    tuple[pd.DataFrame]
        relevant data to define weights for the grid
    """
    if not "profiles" in net.keys() or "load.p_mw" not in net.profiles.keys():
        raise ValueError("grid_parameters() expectes load active power timeseries data stored as "
                         "dictionary in net.profiles.")
    if net_zones is None:
        net_zones = sorted(set(net.bus.zone))

    params = pd.DataFrame(0., index=net_zones, columns=["line_length_km", "load_p_gwh"])
    # copy from data_overview.py:
    line_length = {1: 3515.6,
                   2: 4010.3,
                   3: 1082.1 +   1.4,
                   4:  632.8 + 118.8,
                   }

    for zone in net_zones:
        # remark: the following allocation is a quick approximation and does not correspond to any
        # of the three definitions of boundaries presented in the dissertation
        zone_buses = net.bus.index[net.bus.zone == zone]
        zone_loads = net.load.zone[net.load.bus.isin(zone_buses)]
        params.at[zone, "line_length_km"] = line_length[zone]
        params.at[zone, "load_p_gwh"] = net.profiles["load.p_mw"][zone_loads].sum().sum() / \
            4 / 1000  # -> GWh
    params_rel = params / params.sum()
    params_rel["mean"] = params_rel.mean(axis=1)
    params_rel["weights"] = params_rel["mean"] * len(params_rel)
    return params, params_rel


def weights_from_opt(objective:str="profile_loadings", method:str="COPF") -> pd.Series:
    """Returns weights which would result when taking the results from local control or unweighted
    central optimization. These weights may be an orientation.

    Parameters
    ----------
    objective : str, optional
        defines the objective function from which the corresponding results are to be taken into
        account, by default "profile_loadings"
    method : str, optional
        defines the method from which the corresponding results are to be taken into
        account, by default "COPF"

    Returns
    -------
    pd.Series
        _description_
    """
    data = {"profile_loadings": pd.DataFrame({
        "localCtrl": [20.037488, 105.989305,  67.308047,  38.603703],
        "COPF": [16.728384, 94.654663, 40.587258, 20.976723]}, index=[1, 2, 3, 4]),
        "P_LOSS": pd.DataFrame({
        "localCtrl": [39.814284, 164.020201, 8.587810, 3.751839],
        "COPF": [34.915895, 140.440819, 6.461768, 2.379295]}, index=[1, 2, 3, 4])
        }[objective][method]
    data /= data.sum()
    data *= 4
    return data


if "__main__" == __name__:

    from SimBench_EHV_HV_excerpt.SimBench_for_phd import SimBench_for_phd

    net = SimBench_for_phd(time_steps=True)

    params, params_rel = grid_parameters(net)

    print(params)
    # line_length_km    load_p_gwh
    # 1       3515.6  13911.192222
    # 2       4010.3  50020.810181
    # 3       1083.5    812.065481
    # 4        751.6    764.235654
    print(params_rel)
    #    line_length_km  load_p_gwh      mean   weights
    # 1        0.375558    0.212358  0.293958  1.175832
    # 2        0.428405    0.763580  0.595992  2.383970
    # 3        0.115746    0.012396  0.064071  0.256285
    # 4        0.080291    0.011666  0.045978  0.183914

    # print(weights_from_opt(objective="profile_loadings", method="COPF"))
    # print(weights_from_opt(objective="profile_loadings", method="localCtrl"))
    # print(weights_from_opt(objective="P_LOSS", method="COPF"))
    # print(weights_from_opt(objective="P_LOSS", method="localCtrl"))
