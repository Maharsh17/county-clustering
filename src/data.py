"""Loading and cleaning the merged county file."""
import numpy as np
import pandas as pd

from src.utils import COLS, DATA


def load_counties() -> pd.DataFrame:
    """The merged dataset, cleaned the one way every model expects.

    SVI writes missing values as -999 rather than leaving cells blank, so those
    become nulls before anything else runs. This extract is clean and contains
    none, but a re-pull will not be so tidy.

    Rows missing any modelled column, the urban-rural code or population are
    dropped, which takes 3,132 rows down to 3,128. See data/README.md.
    """
    df = pd.read_csv(DATA, dtype={"FIPS": str}).replace(-999, np.nan)
    df = df.dropna(subset=COLS + ["CODE2023", "E_TOTPOP"]).reset_index(drop=True)
    df["FIPS"] = df["FIPS"].str.zfill(5)
    return df
