import pandas as pd
import numpy as np

df_train = pd.read_csv("train_csv.csv")
df_test = pd.read_csv('test_csv.csv')

print(df_train.dtypes)

#variables that need embedding: client_profile employment_history, client_profile_inheritance_details_profession, cleint_profile_real_estate_details, client_profile_preferred markets, client_description
categorical_features = [
    "passport_gender", "passport_country", "passport_country_code", "passport_nationality",
    "client_profile_country_of_domicile", "client_profile_nationality", "client_profile_gender",
    "client_profile_marital_status", "client_profile_inheritance_details_relationship",
    "client_profile_investment_risk_profile", "client_profile_investment_horizon",
    "client_profile_investment_experience", "client_profile_type_of_mandate",
    "client_profile_currency", "account_form_currency", "account_form_country_of_domicile"
]
date_features = ["passport_birth_date", "passport_passport_issue_date", "passport_passport_expiry_date", "client_profile_birth_date", "client_profile_passport_issue_date", 
                "client_profile_passport_expiry_date", "client_profile_inheritance_details_inheritance_year"]

date_features  = [col for col in df_train.columns if ('date' in col.lower() or "year" in col.lower())]
numerical_features = ["client_profile_aum_savings", "client_profile_aum_inheritance", "client_profile_aum_real_estate_value"]
df_train[categorical_features] = df_train[categorical_features].astype('category')

for col in date_features:
    df_train[col] = pd.to_datetime(df_train[col], errors = 'coerce')




doc_prfx = ["passport_", "client_profile_", "account_form_"]

fields = {}
for col in df_train.columns:
    for prefix in doc_prfx:
        if col.startswith(prefix):
            base_field = col.replace(prefix, "", 1)
            if base_field not in fields:
                fields[base_field] = {}
            fields[base_field][prefix] = col

inconsistencies = {}
for base, col_dict in fields.items():
    if len(col_dict) > 1:
        temp_df = df_train[["folder_name"] + list(col_dict.values())].copy()
        print(f"Processing base field: {base} for columns: {list(col_dict.values())}")
    
        def row_inconsistent(row):
            values = [row[col] for col in col_dict.values() if pd.notnull(row[col])]
            return len(set(values)) > 1 if values else False
    

        temp_df['inconsistent'] =  temp_df.apply(row_inconsistent, axis = 1) 

        bad_clients = temp_df.loc[temp_df['inconsistent'], 'folder_name'].tolist()
        if bad_clients:
            inconsistencies[base] = bad_clients       

for field, client_ids in inconsistencies.items():
    print(f"Inconsistency found in '{field}' for client_ids: {client_ids}")