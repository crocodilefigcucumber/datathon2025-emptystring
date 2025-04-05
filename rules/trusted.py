import json
import os
import datetime
from pathlib import Path
from typing import Tuple

def RM_contact(folder_dir: str) -> Tuple[bool, str]:
    """
    
    """

    if not os.path.isdir(folder_dir):
            print(f"Error: Directory '{folder_dir}' not found.")
            raise
    description_file = os.path.join(folder_dir, "client_description.json")
    
    # Read the JSON file
    with open(description_file, 'r') as file:
        description = json.load(file)
    
    summary = description["Summary Note"]
    if " RM " in summary:
        return True, "Customer contacted by RM"
    return False, ""
    
