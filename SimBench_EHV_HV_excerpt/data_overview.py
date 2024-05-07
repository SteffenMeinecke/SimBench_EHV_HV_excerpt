from copy import deepcopy
import numbers
import math
import numpy as np
import pandas as pd
import pandapower as pp

try:
    from smeinecke.python.Distributed_OPF.sequential.eq_fct.vi.run_eq_fct import preparations
    preparations_imported = True
except ImportError:
    preparations_imported = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import SimBench_EHV_HV_excerpt as sbe


def digits(number:numbers.Number, add_minus_as_digit:bool=False) -> int:
    """
    Returns the number of digits of given number (base is 10).

    Example
    -------
    >>> digits(123.1)
    3
    >>> digits(123456)
    6
    """
    if np.isclose(number, 0):
        return 1
    to_add = int(number < 0 and add_minus_as_digit)
    return int(math.log10(abs(number))) + 1 + to_add

def voltlvls(neti, net):
    return ", ".join(["%g" %vn for vn in sorted(set(neti.bus.vn_kv), reverse=True)])


def ohl_length(neti, net):
    return neti.line.length_km.loc[neti.line.type.isin(["ohl", "ol"])].sum()


def cable_length(neti, net):
    return neti.line.length_km.loc[neti.line.type.isin(["cs", "cable"])].sum()


def p_sum(neti, profiles, et, extremum_fct):
    eq_elms = neti[et].index[neti[et].name.str.contains("eq") | neti[et].name.str.startswith("Ext")]
    idx = neti[et].index.intersection(neti[et].index).difference(eq_elms)
    return extremum_fct(profiles[f"{et}.p_mw"][idx].sum(axis=1))


def p_load_min(neti, profiles):
    return p_sum(neti, profiles, "load", np.min)


def p_load_max(neti, profiles):
    return p_sum(neti, profiles, "load", np.max)


def p_gen_min(neti, profiles):
    return p_sum(neti, profiles, "gen", np.min) + p_sum(neti, profiles, "sgen", np.min)


def p_gen_max(neti, profiles):
    return p_sum(neti, profiles, "gen", np.max) + p_sum(neti, profiles, "sgen", np.max)


def gen_types(neti, net):
    return "{" + ", ".join(sorted(set(neti.gen.type)|set(neti.sgen.type)-{"imp0", "imp1"})) + \
        "}" # gens representing the import from outside the grid is neglected


def add_thousand_sep_digit(digits:int) -> int:
    return digits + int((digits-1)/3)


def number_to_str(number:numbers.Number, max_digits:int, decimals=int) -> str:
    if isinstance(number, float):
        st = format(np.round(number, 1), ",f")
        st = st[:st.find(".")+1+decimals]
    elif isinstance(number, int):
        st = format(number, ",d")
    else:
        raise NotImplementedError(f"{type(number)=} is not, as expected, an int or a float.")
    if missing_digits := add_thousand_sep_digit(max_digits) - add_thousand_sep_digit(
            digits(number, add_minus_as_digit=True)):
        return "\\ph{%s}%s" % ("0"*missing_digits, st)
    else:
        return st


def add_ph(df:pd.DataFrame) -> None:
    for col in df.select_dtypes(include=[int, float]):
        max_digits = max([add_thousand_sep_digit(digits(val, add_minus_as_digit=True)) for val in
            df[col]])
        df[col] = df[col].apply(number_to_str, **{"max_digits": max_digits, "decimals": 1})


def element_number(net:pp.pandapowerNet, et:str, zone:int|str) -> int:
    if et == "bus" and zone != "Complete Grid":
        return net[et].zone.value_counts().at[zone]
    elif et in ["gen", "load"]:
        return sum(~(net[et].name.str.contains("eq") | net[et].name.str.startswith("Ext")))
    else:
        return len(net[et])


def SimBenchZoneName(zone):
    return f"DSO{zone}" if zone > 2 else f"TSO{zone}"


def electric_params(nets):
    """ Produces data summarized in Table A.4 of the Dissertation
    """
    # --- electric params table
    str_params = [
        "{Voltage levels in kV}",
        # "{Generation types}",
    ]
    str_param_fcts = [voltlvls] #, gen_types]
    num_params = [
        "{Total length of\\\\overhead lines in km}",
        "{Total length of\\\\cables in km}",
        "{Minimum active power\\\\consumption$^\\alpha$ in MW}",
        "{Maximum active power\\\\consumption$^\\alpha$ in MW}",
        "{Minimum active power\\\\generation$^\\alpha$ in MW}",
        "{Maximum active power\\\\generation$^\\alpha$ in MW}",
        ]
    num_param_fcts = [ohl_length, cable_length, p_load_min, p_load_max, p_gen_min, p_gen_max]

    df_num = pd.DataFrame({key if isinstance(key, str) else SimBenchZoneName(key): pd.Series({
        el_param: el_param_fct(neti, nets["Complete Grid"].profiles) for el_param, el_param_fct in
            zip(num_params, num_param_fcts)}) for key, neti in nets.items()})
    df_num_for_tex = deepcopy(df_num)
    add_ph(df_num_for_tex)

    df_str = pd.DataFrame({key if isinstance(key, str) else SimBenchZoneName(key): pd.Series({
        el_param: el_param_fct(neti, nets["Complete Grid"]) for el_param, el_param_fct in zip(
            str_params, str_param_fcts)}) for key, neti in nets.items()})
    el_par = pd.concat([df_str, df_num_for_tex])

    st = "\n" + "-"*20 + "electric params table" + "-"*20
    st += str(df_num)
    logger.info(st)

    return {
        "el_params": df_num,
        "el_params_for_tex": el_par.to_latex(),
        }


def element_numbers(nets):
    """ Produces data summarized in Table A.5 of the Dissertation
    """
    # --- Element number tables
    ets = ["bus", "line", "trafo", "load", "gen", "sgen"]
    et_names_for_doc = ["Bus", "Line", "Transformer","$PQ$ consumption", "$PV$ generator",
                        "$PQ$ static generator"]
    et_repl = dict(zip(ets, et_names_for_doc))

    et_count_complete = pp.count_elements(nets["Complete Grid"])

    et_count = pd.DataFrame({key if isinstance(key, str) else SimBenchZoneName(key): pd.Series({
        et: element_number(neti, et, key) for et in et_count_complete.keys()}) for key, neti in
        nets.items()})
    et_count.index = pd.Index(pd.Series(et_count.index).replace(et_repl))
    et_count = et_count.loc[et_names_for_doc]
    sum_diff_row = [row for row in et_count.index if et_count.loc[row].iloc[:-1].sum() != \
        et_count.loc[row].iat[-1]]
    et_count_for_tex = deepcopy(et_count)
    add_ph(et_count_for_tex)

    if len(sum_diff_row):
        logger.warning("Rows where 'Complete Grid' is not the sum of the individual SOs: "
                       f"\n{sum_diff_row}")

    st = "\n" + "-"*20 + "Element number tables" + "-"*20
    st += str(et_count)
    logger.info(st)

    ExtL = pd.Series({key: sum(neti.load.name.str.startswith("ExtL")) for key, neti in nets.items()})
    eq_gens = pd.Series({key: sum(neti.gen.name.str.startswith("equivalent gen")) for
                         key, neti in nets.items()})
    return {
        "et_count": et_count,
        "et_count_for_tex": et_count_for_tex.to_latex(),
        "equivalent gens": eq_gens.T,
        "equivalent loads": ExtL.T,
        }


def get_SimBench_nets_series():
    """ Splits the net into four parts according to the zones and with respect to the boundary
    definition used by the equivalent function method.
    """
    if not preparations_imported:
        raise ImportError("Function 'preparations()' from the equivalent function implementation is"
                          "expected in get_SimBench_nets_series() but not imported successfully.")
    net = sbe.SimBench_for_phd(time_steps=range(2*96))
    nets = preparations(net, 0.01, time_step=0, no_hv_zones_allowed=True,
        objective={1: 'P_LOSS', 2: 'profile_loadings', 3: 'P_LOSS', 4: 'profile_loadings'})[0]
    nets = pd.Series(nets).sort_index()
    nets.loc["Complete Grid"] = net
    return nets


if __name__ == "__main__":
    nets = get_SimBench_nets_series()

    el_pars = electric_params(nets)
    # print(elm_num["el_params_for_tex"])

    elm_num = element_numbers(nets)
    # print(elm_num["et_count_for_tex"])
