import json
import os
from typing import Tuple


def check_email_name(folder_dir: str) -> Tuple[bool, str]:
    """
    Compares 'email_address' fields in 'account_form.json' and 'client_profile.json'.

    Args:
        folder_dir (str): Path containing both JSON files.

    Returns:
        Tuple[bool, str]: (success, message)

    Possible messages:
        - "No Account Form"
        - "Account form missing email_address"
        - "No Client Profile"
        - "Client profile missing email_address"
        - "Email addresses do not match"
        - "Email addresses match"
    """
    if not os.path.isdir(folder_dir):
        raise Exception(f"Directory '{folder_dir}' not found.")

    # ---------- ACCOUNT FORM ----------
    account_path = os.path.join(folder_dir, "account_form.json")
    if not os.path.exists(account_path):
        return False, "No Account Form"

    with open(account_path, "r") as f:
        account = json.load(f)

    email_account = account.get("email_address", "")
    if not email_account:
        return False, "Account form missing email_address"

    # ---------- CLIENT PROFILE ----------
    client_path = os.path.join(folder_dir, "client_profile.json")
    if not os.path.exists(client_path):
        return False, "No Client Profile"

    with open(client_path, "r") as f:
        client = json.load(f)

    email_client = client.get("email_address", "")
    if not email_client:
        return False, "Client profile missing email_address"

    if email_account != email_client:
        return False, "Email addresses do not match"

    return True, "Email addresses match"
