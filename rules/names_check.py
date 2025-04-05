import json
import os
import datetime
from pathlib import Path
from typing import Tuple


def check_names(folder_dir: str) -> Tuple[bool, str]:
    """
    Validates the consistency of names across passport, account form, and client profile JSON files
    located in the specified folder directory.
    Args:
        folder_dir (str): The directory containing the JSON files: 'passport.json',
                          'account_form.json', and 'client_profile.json'.
    Returns:
        Tuple[bool, str]: A tuple where the first element is a boolean indicating whether
                          the validation passed, and the second element is a string message
                          describing the result or the specific issue encountered.
    Raises:
        Exception: If the specified folder does not exist or cannot be accessed.
    Validation Steps:
        1. Checks if the folder exists.
        2. Validates the presence of 'passport.json', 'account_form.json', and 'client_profile.json'.
        3. Ensures that the 'first_name', 'last_name', and optionally 'middle_name' fields are present
           in both 'passport.json' and 'account_form.json'.
        4. Reconstructs the full name from the first, middle, and last names in both files and compares them.
        5. Validates that the 'name' field in 'client_profile.json' matches the full name in 'account_form.json'.
    Possible Return Messages:
        - "No Passport": If 'passport.json' is missing.
        - "Passport missing first name": If the 'first_name' field is missing in 'passport.json'.
        - "Passport missing last name": If the 'last_name' field is missing in 'passport.json'.
        - "No Account Form": If 'account_form.json' is missing.
        - "Account form missing first name": If the 'first_name' field is missing in 'account_form.json'.
        - "Account form missing last name": If the 'last_name' field is missing in 'account_form.json'.
        - "Account form missing full name": If the 'name' field is missing in 'account_form.json'.
        - "Passport and Account Form names do not match": If the reconstructed names from 'passport.json'
          and 'account_form.json' do not match.
        - "Account Form and Full name do not match in account form": If the reconstructed name does not
          match the 'name' field in 'account_form.json'.
        - "No Client Profile": If 'client_profile.json' is missing.
        - "Client Profile missing name": If the 'name' field is missing in 'client_profile.json'.
        - "Client Profile and Account Form names do not match": If the 'name' field in 'client_profile.json'
          does not match the 'name' field in 'account_form.json'.
        - "Names good": If all validations pass successfully.
    """

    # Get current date
    current_date = datetime.datetime.now().date()

    # Get file and check existence
    if not os.path.isdir(folder_dir):
        print(f"Error: Directory '{folder_dir}' not found.")
        raise

    # ----------PASSPORT----------
    # Read the passport JSON file
    with open(passport_file, "r") as file:
        passport_data = json.load(file)

    passport_file = os.path.join(folder_dir, "passport.json")
    if not os.path.exists(passport_file):
        print(f"Passport found in '{folder_dir}'.")
        return False, "No Passport"

    passport_first_name = passport_data["first_name"]
    if not passport_first_name:
        return False, "Passport missing first name"

    passport_middle_name = passport_data["middle_name"]

    passport_last_name = passport_data["last_name"]
    if not passport_last_name:
        return False, "Passport missing last name"

    # reconstruct name from first,middle,last
    passport_full_name = " ".join(
        part.strip()
        for part in [passport_first_name, passport_middle_name, passport_last_name]
        if part and part.strip()
    )

    # ----------ACCOUNT----------
    account_file = os.path.join(folder_dir, "account_form.json")
    if not os.path.exists(account_file):
        print(f"Account form found in '{folder_dir}'.")
        return False, "No Account Form"

    # Read the account form JSON file
    with open(account_file, "r") as file:
        account_data = json.load(file)

    account_first_name = account_data["first_name"]
    if not account_first_name:
        return False, "Account form missing first name"

    account_middle_name = account_data["middle_name"]

    account_last_name = account_data["last_name"]
    if not account_last_name:
        return False, "Account form missing last name"

    account_full_name = account_data["name"]
    if not account_full_name:
        return False, "Account form missing full name"

    # reconstruct name from first,middle,last
    account_full_name_recon = " ".join(
        part.strip()
        for part in [account_first_name, account_middle_name, account_last_name]
        if part and part.strip()
    )

    # Check if names match
    if passport_full_name != account_full_name_recon:
        return False, "Passport and Account Form names do not match"

    if account_full_name != account_full_name_recon:
        return False, "Account Form and Full name do not match in account form"

    # ----------CLIENT PROFILE----------
    client_profile_file = os.path.join(folder_dir, "client_profile.json")
    if not os.path.exists(client_profile_file):
        print(f"Client profile found in '{folder_dir}'.")
        return False, "No Client Profile"

    # Read the client profile JSON file
    with open(client_profile_file, "r") as file:
        client_profile_data = json.load(file)

    client_profile_name = client_profile_data["name"]
    if not client_profile_name:
        return False, "Client Profile missing name"

    if client_profile_name != account_full_name:
        return False, "Client Profile and Account Form names do not match"

    return True, "Names good"
