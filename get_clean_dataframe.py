import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    date_features  = [col for col in df.columns if ('date' in col.lower() or "year" in col.lower())]

    for col in date_features:
        df[col] = pd.to_datetime(df[col], errors = 'coerce')
        
    doc_prfx = ["passport_", "client_profile_", "account_form_"]

    fields = {}
    for col in df.columns:
        for prefix in doc_prfx:
            if col.startswith(prefix):
                base_field = col.replace(prefix, "", 1)
                if base_field not in fields:
                    fields[base_field] = {}
                fields[base_field][prefix] = col


    df.rename(columns = {"foldr_name": "client_id"}, inplace = True)

    for base, col_dict in fields.items():

        chosen_col = None
        for prfx in doc_prfx:
            if prfx in col_dict:
                chosen_col = col_dict[prfx]
                break
        if chosen_col is not None:
            df[base] = df[chosen_col]

    cols_to_drop = [col for col in df.columns if any(col.startswith(prefix) for prefix in doc_prfx)]
    df.drop(columns=cols_to_drop, inplace=True)

    return df