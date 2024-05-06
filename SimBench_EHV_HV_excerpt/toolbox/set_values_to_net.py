import numpy as np
import pandas as pd
import pandapower as pp


def set_time_step(net: pp.pandapowerNet,
                  time_step: int,
                  abs_profiles: dict|pd.io.pytables.HDFStore|None = None,
                  intersection: bool = False):
    """
    Sets values from abs_profiles (or if not given from net.profiles) to the net.
    Can handle dict keys "et.col" and (et, col).
    """
    if abs_profiles is None:
        abs_profiles = net.profiles
    keys = abs_profiles.keys()
    for key in keys:
        et, col = key[1:].split("/") if isinstance(abs_profiles, pd.io.pytables.HDFStore) else \
            get_et_col(key)
        if net[et][col].shape[0]:
            if isinstance(abs_profiles, pd.io.pytables.HDFStore):
                val = abs_profiles.select(key, start=time_step, stop=time_step+1)
                if time_step != val.index[0]:
                    raise ValueError(f"The {time_step}th row has not index {time_step} in the {key}"
                                     " profile of the hdf5 data.")
            else:
                val = abs_profiles[key]
            if isinstance(val, pd.DataFrame):
                if intersection:
                    idx = net[et][col].index.intersection(val.loc[time_step].index)
                    net[et][col].loc[idx] = val.loc[time_step].loc[idx]
                else:
                    net[et][col] = val.loc[time_step]
            else:
                if intersection:
                    raise TypeError("intersection cannot be considered since profiles are not of "
                                    "type pd.DataFrame.")
                net[et][col] = val[time_step]


def set_sgen_limits(
        net: pp.pandapowerNet,
        variant: int = 0,
        version: int = 2018,
        fixed_p: bool = True,
        p_margin: float = 0.,
        drop_qcurve1_column: bool = False,
        set_to_limits: bool = True,
        ):
    """Sets net.sgen[["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]] with regard to the Q(P)
    dependency of the German VDE AR-N-4120 (equals AR-N-4130).
    The Q(Vm) dependency is not considered here. Instead, the maximum and minimum Q(Vm) limits are
    set.

    Note
    ----
    The content is also processed in the DERController.

    Parameters
    ----------
    net : pp.pandapowerNet
        net to be adapted
    variant : int, optional
        variant of VDE AR-N standard. Beside the variants 1, 2, and 3, if 0 is given, the variants
        provided by net data (net.sgen.qcurve1) is considered, by default 0
    version : int, optional
        year of VDE AR-N standard, by default 2018
    fixed_p : bool, optional
        makes use of the knowledge whether p_mw is fixed in the AMPL code;
        Lead to different behaviour of setting max_p_mw (directly to the limit or somewhere
        above (doesn't count anyway)), by default True
    p_margin : float, optional
        additional value to max_p_mw if fixed_p is False, by default 0.
    drop_qcurve1_column : bool, optional
        whether to drop the column "qcurve1", by default False
    set_to_limits: bool, optional
        whether net.sgen.q_mvar should be set into newly set limits, by default True

    Returns
    -------
    float
        maximum adaption of net.sgen.q_mvar to set it to the VDE limits (can only be other than 0
        if set_to_limits is True)
    """
    # --- set p limits for all sgens
    net.sgen["min_p_mw"] = 0
    if fixed_p:
        # since p is fixed, the limits can be >p to ensure that max_p_mw is not smaller or near to
        # p_mw which could raise in problems with the solver
        if "profiles" in net and isinstance(net, dict) and "sgen.p_mw" in net.profiles.keys():
            net.sgen["max_p_mw"] = pd.concat([net.sgen.max_p_mw, net.profiles["sgen.p_mw"].max()],
                                             axis=1).max(axis=1)
        else:
            net.sgen["max_p_mw"] = 2*net.sgen.p_mw
    else:
        net.sgen["max_p_mw"] = net.sgen.p_mw + p_margin

    if variant not in [0, 1, 2, 3]:
        raise NotImplementedError(f"{variant=}")

    # --- set q limits as given by variant
    v1_sgens = pd.Index([])
    v2_sgens = pd.Index([])
    v3_sgens = pd.Index([])
    if variant == 0:
        if "qcurve1" not in net.sgen.columns:
            raise ValueError(f"{variant=} but 'qcurve1' is not in net.sgen.columns.")
        v1_sgens = net.sgen.index[net.sgen.qcurve1.isin(["7", "10"]) |
                                  net.sgen.qcurve1.str.endswith("_v1")]
        v2_sgens = net.sgen.index[net.sgen.qcurve1.isin(["8", "11"]) |
                                  net.sgen.qcurve1.str.endswith("_v2")]
        v3_sgens = net.sgen.index[net.sgen.qcurve1.isin(["9", "12"]) |
                                  net.sgen.qcurve1.str.endswith("_v3")]
    elif variant == 1:
        v1_sgens = net.sgen.index
    elif variant == 2:
        v2_sgens = net.sgen.index
    elif variant == 3:
        v3_sgens = net.sgen.index

    # Handtuch curve points AR-N 4120
    handtuch_x = {2015: [0, 0.1, 0.2], 2018: [0, 0.05, 0.2]}[version]

    def handtuch_y(lim_type, variant):
        y_end = {"max_q_mvar": [0.484322, 0.410775, 0.328684],
                 "min_q_mvar": [0.227902, 0.328684, 0.410775]}[lim_type][
            variant-1]
        return [0, 0.1, y_end]

    # set q limits according to Q(P)
    for col, sign in zip(["max_q_mvar", "min_q_mvar"], [1, -1]):
        new = pd.concat([pd.Series(np.interp(
            net.sgen.p_mw.loc[sgens]/net.sgen.sn_mva.loc[sgens],
            handtuch_x, handtuch_y(col, i_sgens+1)) * net.sgen.sn_mva.loc[sgens], index=sgens
            ).fillna(1e-5) for i_sgens, sgens in enumerate(
                [v1_sgens, v2_sgens, v3_sgens])])
        new.loc[np.isclose(new, 0)] = 1e-5
        net.sgen.loc[new.index, col] = sign*new

    # set q limits according to Q(Vm) - requires "4130_220_v1" or similar (not integer values as in older times)
    vde_lims = VDE_Q_minmax()
    idx = net.sgen.qcurve1.isin(vde_lims.index)
    for lim, fct in zip(["min", "max"], [np.maximum, np.minimum]):
        net.sgen[f"{lim}_q_mvar"].loc[idx] = fct(
            net.sgen[f"{lim}_q_mvar"].loc[idx],
            vde_lims.loc[net.sgen.qcurve1.loc[idx], lim].values * net.sgen.p_mw.loc[idx])

    # set sgen.q_mvar into limits
    max_q_adaption = 0.
    if set_to_limits:
        in_lim_q = np.maximum(np.minimum(net.sgen.q_mvar.loc[new.index].values,
                                         net.sgen.max_q_mvar.loc[new.index].values),
                              net.sgen.min_q_mvar.loc[new.index].values).astype(float)
        if not np.allclose(net.sgen.q_mvar.loc[new.index].values, in_lim_q):
            max_q_adaption = np.max(np.abs(net.sgen.q_mvar.loc[new.index].values - in_lim_q))
            net.sgen.loc[new.index, "q_mvar"] = in_lim_q


    # drop_qcurve1_column
    if drop_qcurve1_column and "qcurve1" in net.sgen.columns:
        del net.sgen["qcurve1"]

    return max_q_adaption


def get_et_col(et_col):
    error = False
    if isinstance(et_col, tuple):
        assert len(et_col) == 2
        return et_col
    elif isinstance(et_col, str):
        split1 = et_col.split(".")
        split2 = et_col.split("/")
        if len(split1) == 2:
            return split1
        elif len(split2) == 2:
            return split2
        else:
            error=True
    else:
        error=True
    if error:
        raise NotImplementedError(
            "The input is expected as ('gen', 'vm_pu'), as 'gen.vm_pu', or as 'gen/vm_pu', however"
            f", {et_col=}")


def VDE_Q_minmax():
    return pd.DataFrame([
        [-0.227902, 0.484322],
        [-0.328684, 0.410775],
        [-0.410775, 0.328684],
        [-0.227902, 0.484322],
        [-0.328684, 0.410775],
        [-0.410775, 0.328684],
        [-0.227902, 0.484322],
        [-0.328684, 0.410775],
        [-0.410775, 0.328684]
        ], columns=["min", "max"], index=pd.Index([
            "4130_380_v1", "4130_380_v2", "4130_380_v3", "4130_220_v1", "4130_220_v2", "4130_220_v3",
            "4120_v1", "4120_v2", "4120_v3"], name="qcurve1"))
