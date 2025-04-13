from copy import deepcopy
import pandas as pd


def downcast_profiles(profiles:dict[str, pd.DataFrame]|pd.io.pytables.HDFStore, **kwargs) -> None:
    for key in profiles.keys():
        if isinstance(profiles, dict):
            downcast_numerics(profiles[key])
        elif isinstance(profiles, pd.io.pytables.HDFStore):
            kwargs = deepcopy(kwargs)
            kwargs["format"] = kwargs.get("format", "table")
            df = profiles.get(key)
            downcast_numerics(df)
            profiles.put(key, df, **kwargs)
        else:
            raise NotImplementedError(f"{type(profiles)=}")


def downcast_numerics(df:pd.DataFrame) -> pd.DataFrame:
    """This function downcasts given numeric data to save storage.

    Parameters
    ----------
    df : pd.DataFrame
        data to downcast

    Returns
    -------
    pd.DataFrame
        downcasted data (Output can also be neglected since downcasting happens in place)

    See also
    --------
    For converting, for example from object dtype, and downcasting data, use
    simbench.to_numeric_ignored_errors(data). Here, only numerics are downcasted to save storage.
    """
    for typ in ["float", "integer"]:
        cols = df.select_dtypes(typ).columns
        df[cols] = df[cols].apply(pd.to_numeric, downcast=typ)
    return df
