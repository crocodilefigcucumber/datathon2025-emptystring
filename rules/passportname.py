import json
import os
import datetime
from pathlib import Path
from typing import Tuple


def check_passport_expiry(folder_dir: str) -> Tuple[bool, str]:
    """
    Checks JSON files in the specified directory for passport expiry dates
    and reports any that have already expired.

    Args:
        json_dir (str): Directory path containing JSON file to check

    Returns:
        bool: accepted or not, sting: explanation
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
    print(f"Passport full name: {passport_full_name}")

    # ----------ACCOUNT----------
    account_file = os.path.join(folder_dir, "account_form.json")
    if not os.path.exists(account_file):
        print(f"Account form found in '{folder_dir}'.")
        return False, "No Account Form"

    # Read the account form JSON file
    with open(account_file, "r") as file:
        account_data = json.load(file)

    # ----------CLIENT PROFILE----------
    client_profile_file = os.path.join(folder_dir, "client_profile.json")
    if not os.path.exists(client_profile_file):
        print(f"Client profile found in '{folder_dir}'.")
        return False, "No Client Profile"

    # Read the client profile JSON file
    with open(client_profile_file, "r") as file:
        client_profile_data = json.load(file)

    # Check if names match
