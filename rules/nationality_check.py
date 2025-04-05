import json
import os
from typing import Tuple


def check_nationality_match(folder_dir: str) -> Tuple[bool, str]:
    """
    Compares 'nationality' in 'passport.json' and 'client_profile.json'.

    Args:
        folder_dir (str): Path containing the JSON files.

    Returns:
        Tuple[bool, str]: (success, message)

    Possible messages:
        - "No Passport"
        - "Passport missing nationality"
        - "No Client Profile"
        - "Client profile missing nationality"
        - "Nationalities do not match"
        - "Nationalities match"
    """
    if not os.path.isdir(folder_dir):
        raise Exception(f"Directory '{folder_dir}' not found.")

    # ---------- PASSPORT ----------
    passport_path = os.path.join(folder_dir, "passport.json")
    if not os.path.exists(passport_path):
        return False, "No Passport"

    with open(passport_path, "r") as f:
        passport = json.load(f)

    passport_nationality = passport.get("nationality", "")
    if not passport_nationality:
        return False, "Passport missing nationality"

    # ---------- CLIENT PROFILE ----------
    client_path = os.path.join(folder_dir, "client_profile.json")
    if not os.path.exists(client_path):
        return False, "No Client Profile"

    with open(client_path, "r") as f:
        client = json.load(f)

    nationality_client = client.get("nationality", "")
    if not nationality_client:
        return False, "Client profile missing nationality"

    if passport_nationality != nationality_client:
        return False, "Nationalities do not match"

    return True, "Nationalities match"
