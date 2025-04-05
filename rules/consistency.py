from typing import Tuple
import pathlib
import json

def check_inconsistency(client_folder: str) -> Tuple[bool, str]:
    # Get all json documents
    documents = list(pathlib.Path(client_folder).glob("*.json"))
    # write dict
    client_data = {}
    # dict for keys and files in which they appear
    # {key1: [file1, file2, ...], key2: ...}
    key_dict = {}
    for json_file in documents:
        try:
            # Read the JSON file
            with open(json_file, "r") as file:
                file_data = json.load(file)

                file_prefix = json_file.stem
                for key, value in file_data.items():
                    client_data[f"{file_prefix}_{key}"] = value
                    if key in key_dict:
                        key_dict[key].append(file_prefix)
                    else:
                        key_dict[key] = [file_prefix]

        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {json_file}")
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
    
    # check for inconsistent values
    for key in key_dict.keys():
        try:
            values = [client_data[f"{file_prefix}_{key}"] for file_prefix in key_dict[key]]
            if len(list(set(values))) > 1:
                return False, f"inconsistency in {key}" 
        except:
            pass
    return True, "good"
