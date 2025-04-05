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
    passport_file = os.path.join(folder_dir, "passport.json")
    if not os.path.exists(passport_file):
        print(f"Passport found in '{folder_dir}'.")
        return  False, "No Passport"
    
    # Read the JSON file
    with open(passport_file, 'r') as file:
        data = json.load(file)
    
    # Get the expiry date
    expiry_date_str = data["passport_expiry_date"]
    
    expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    
    # Check if expired
    if expiry_date < current_date:
        return False, "Passport expired"
    else:
        return True, "good"

