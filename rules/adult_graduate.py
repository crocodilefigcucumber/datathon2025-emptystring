import json
import os
import datetime
from typing import Tuple


def check_education_graduation(folder_dir: str) -> Tuple[bool, str]:
    """
    Checks the client_profile.json file in the specified directory for graduation ages from
    secondary school and higher education.

    Returns:
      - True if the client has graduated from either secondary school or any higher education
        institution at an age <= 17.
      - False otherwise.

    Args:
      folder_dir (str): Directory path containing the client_profile.json file.

    Returns:
      Tuple[bool, str]: A tuple where the boolean indicates if the criteria are met,
                        and the string provides an explanation.
    """
    # Check if the directory exists
    if not os.path.isdir(folder_dir):
        print(f"Error: Directory '{folder_dir}' not found.")
        raise Exception("Directory not found")

    profile_file = os.path.join(folder_dir, "client_profile.json")
    if not os.path.exists(profile_file):
        print(f"Error: File '{profile_file}' not found.")
        raise Exception("File not found")

    # Load the JSON file
    with open(profile_file, "r") as file:
        data = json.load(file)

    # Retrieve and parse the birth date
    birth_date_str = data.get("birth_date")
    if not birth_date_str:
        return False, "Birth date not found"

    try:
        birth_date = datetime.datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    except ValueError:
        return False, "Invalid birth date format"

    # Check secondary school graduation
    secondary_school = data.get("secondary_school")
    secondary_msg = ""
    valid_secondary = False
    if secondary_school:
        graduation_year = secondary_school.get("graduation_year")
        if graduation_year:
            try:
                # Assume graduation occurs on July 1st of the graduation year
                graduation_date = datetime.date(int(graduation_year), 7, 1)
                age_at_secondary = (
                    graduation_date.year
                    - birth_date.year
                    - (
                        (graduation_date.month, graduation_date.day)
                        < (birth_date.month, birth_date.day)
                    )
                )
                secondary_msg = f"Secondary school graduation age: {age_at_secondary}"
                if age_at_secondary <= 17:
                    valid_secondary = True
            except Exception:
                secondary_msg = "Invalid graduation year in secondary_school"
        else:
            secondary_msg = "No graduation year in secondary_school"
    else:
        secondary_msg = "No secondary_school record"

    # Check higher education graduation
    higher_education = data.get("higher_education", [])
    higher_msg = ""
    valid_higher = False
    if higher_education:
        for entry in higher_education:
            graduation_year = entry.get("graduation_year")
            if not graduation_year:
                continue  # Skip entries without a graduation year
            try:
                graduation_date = datetime.date(int(graduation_year), 7, 1)
                age_at_higher = (
                    graduation_date.year
                    - birth_date.year
                    - (
                        (graduation_date.month, graduation_date.day)
                        < (birth_date.month, birth_date.day)
                    )
                )
                higher_msg += f"Higher education graduation age: {age_at_higher}; "
                if age_at_higher <= 17:
                    valid_higher = True
                    break  # No need to check further if one record qualifies
            except Exception:
                higher_msg += "Invalid graduation year in a higher_education record; "
    else:
        higher_msg = "No higher education record"

    # Decide final result: if either secondary or higher education meets the criteria
    if valid_secondary or valid_higher:
        return True, f"Criteria met. {secondary_msg}. {higher_msg}"
    else:
        return False, f"Criteria not met. {secondary_msg}. {higher_msg}"
