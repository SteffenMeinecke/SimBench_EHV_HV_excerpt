import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import DiscreteTapControl

try:
    from pandapower.control.controller.DERController import DERController, QModelQV, \
        PQVArea4120V2, CosphiPCurve
    controllers_imported = True
except ImportError:
    try:
        from pandaplan.core.control import DERController, QModelQV, PQVArea4120V2, CosphiPCurve
        controllers_imported = True
    except ImportError:
        controllers_imported = False
try:
    from pandaplan.core.control import DistributedSlack
    distrSl_imported = True
except:
    distrSl_imported = False


def controller_type_index(net, controller_type):
    return pd.Index([idx for idx in net.controller.index if isinstance(
        net.controller.object.at[idx], controller_type)])


def consider_distr_slack(net, **kwargs):
    """ kwargs for DistributedSlack controller """
    if not distrSl_imported:
        raise ModuleNotFoundError(
            "consider_distr_slack() uses not open-source control functionality pandaplan-core.")
    if "controller" in net.keys() and isinstance(net.controller, pd.DataFrame):
        ds_idx = controller_type_index(net, DistributedSlack)
    else:
        ds_idx = pd.Index([])
    if not len(ds_idx):
        kwargs["level"] = kwargs.get("level", 1)
        kwargs["tol_mw"] = kwargs.get("tol_mw", 0.1)
        kwargs["keep"] = kwargs.get("keep", False)
        ds_idx = pd.Index([DistributedSlack(net, **kwargs).index])
    return ds_idx


def add_control_strategy(net:pp.pandapowerNet, control:str|None) -> dict[str,pd.Index]|None:
    if control in ["NoControl", None]:
        return

    if not controllers_imported:
        raise ImportError("Controllers are needed to add control strategy to net but have not "
                          "been imported.")

    if "profiles" in net:
        have_p_sgens = net.sgen.index[(net.profiles["sgen.p_mw"].abs() > 1e-4).any()]
        data_source = pp.timeseries.DFData(net.profiles["sgen.p_mw"][have_p_sgens])
    else:
        have_p_sgens = net.sgen.index
        data_source = pp.timeseries.DFData(pd.DataFrame(
            np.empty((0, len(have_p_sgens))), columns=have_p_sgens, index=[]))

    if control == "LocalCtrl":  # DERController - Q(Vm) and CosPhi(P)
        pv_idx = net.sgen.index[net.sgen.type.isin(["PV", "pv"])].intersection(have_p_sgens)
        cosp_idx = pv_idx[np.arange(0, len(pv_idx), 2, dtype=int)]
        cos1_idx = pv_idx.difference(cosp_idx)
        other_idx = net.sgen.index[net.sgen.type != "wind offshore"].difference(
            pv_idx).intersection(have_p_sgens)
        qofv_idx = other_idx[np.arange(0, len(other_idx), 2, dtype=int)].union(net.sgen.index[
            net.sgen.type == "wind offshore"]).intersection(net.sgen.index[net.sgen.controllable])
        cosp_idx = cosp_idx.union(other_idx.difference(qofv_idx)).intersection(net.sgen.index[
            net.sgen.controllable])

        if False:  # print apparent power sums of the sgens with different control characteristics:
            print(net.sgen.sn_mva.loc[cos1_idx].sum())
            print(net.sgen.sn_mva.loc[cosp_idx].sum())
            print(net.sgen.sn_mva.loc[qofv_idx].sum())
            print(len(have_p_sgens.difference(cos1_idx).difference(cosp_idx).difference(qofv_idx)))

        cosphip_model = QModelQV(qv_curve=CosphiPCurve(
            p_points=(0, 0.5, 1),
            cosphi_points=(1, 1, -0.9)))

        qofv_model = QModelQV({
            # "v_points_pu": (0, 0.96, 1.04),  # old
            "v_points_pu": (0, 0.98, 1.06),
            "q_points": (0.484, 0.484, -0.484)})

        pqu_area = PQVArea4120V2()  # PQVArea4120V1, PQVArea4120V3

        for idxs, q_model in zip([qofv_idx, cosp_idx], [qofv_model, cosphip_model]):
            for idx in idxs:
                DERController(net, idx, q_model=q_model, pqu_area=pqu_area, damping_coef=3,
                                p_profile=idx, profile_name=idx,
                                data_source=data_source)

        # --- trafo control
        DiscreteTapControl(net, net.trafo.index.difference([106]).values, 1.005, 1.055, side="lv")
        DiscreteTapControl(net, 106, 1.005, 1.055, side="hv")
        return {"sgen": qofv_idx.union(cosp_idx)}

    elif control == "QofV":  # DERController - only Q(Vm)
        q_model = QModelQV({
            "v_points_pu": (0, 0.93, 0.97, 1.03, 1.07),
            "q_points": (0.484, 0.484, 0, 0, -0.484)})  # QofV values
        pqv_area = PQVArea4120V2()  # PQVArea4120V1, PQVArea4120V3
        for idx in have_p_sgens:
            DERController(net, idx, q_model=q_model, pqv_area=pqv_area, damping_coef=3,
                             p_profile=idx, profile_name=idx,
                             data_source=data_source)
        return {"sgen": have_p_sgens}

    else:
        raise NotImplementedError(f"{control=} is unknown.")
