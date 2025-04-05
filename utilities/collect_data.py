import pandas as pd
import pathlib
import os
import json
import numpy as np
import sys

def collect_enriched(mode: str, filename: str, rules: list) -> pd.DataFrame:
    n_rules = len(rules)

    mode = "train"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1].strip().lower() in ["--help", "-h"]:
            print("Usage: python main.py [mode]")
            print("mode: One of train, test, val, final")
            sys.exit(0)
        mode = sys.argv[1].strip().lower()

    if mode not in ["train", "test", "val", "final"]:
        raise ValueError("Invalid mode. Please enter one of: train, test, val, final.")

    if mode in ["train", "test", "val"]:
        dataset_path = "data/"
        split_path = "splits/" + mode + "_split.csv"
    else:
        raise NotImplementedError("Final mode not yet implemented.")

    clients = pd.read_csv(split_path)["file_path"].tolist()
    clients = sorted(clients)

    n_clients = len(clients)
    enriched = pd.DataFrame(data=np.zeros((n_clients, n_rules)))
    enriched.columns = [rule.__name__ for rule in rules]
    all_data = []

    i = 0
    for client in clients:
        client_folder = os.path.join(dataset_path, client)
        client_data = {"folder_name": client}  # Initialize with folder name

        # Get all json documents
        documents = list(pathlib.Path(client_folder).glob("*.json"))

        if not documents:
            print(f"No JSON files found in folder: {client}")
            all_data.append(client_data)
            continue
        
        # extract features
        for rule in rules:
            name = rule.__name__
            accept, comment = rule(client_folder)
            enriched.loc[i, name] = (accept)

        # write DataFrame
        for json_file in documents:
            try:
                # Read the JSON file
                with open(json_file, "r") as file:
                    file_data = json.load(file)

                    file_prefix = json_file.stem
                    for key, value in file_data.items():
                        # Flatten nested dictionaries (optional - comment out if not needed)
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                client_data[f"{file_prefix}_{key}_{nested_key}"] = (
                                    nested_value
                                )
                        else:
                            client_data[f"{file_prefix}_{key}"] = value

            except json.JSONDecodeError:
                print(f"Invalid JSON format in file: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")

        i += 1

        # Add this folder's data to the list
        all_data.append(client_data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)
    df_enriched = pd.concat([df, enriched], axis=1, ignore_index=True)
    df_enriched.to_csv(filename, index=False)
    return df_enriched


def collect_to_csv(clients: list, filename: str) -> pd.DataFrame:
    all_data = []

    for client in clients:
        client_folder = os.path.join("./data", client)
        client_data = {"folder_name": client}  # Initialize with folder name

        # Get all json documents
        documents = list(pathlib.Path(client_folder).glob("*.json"))

        if not documents:
            print(f"No JSON files found in folder: {client}")
            all_data.append(client_data)
            continue

        # Process each JSON file in the folder
        for json_file in documents:
            try:
                # Read the JSON file
                with open(json_file, "r") as file:
                    file_data = json.load(file)

                    file_prefix = json_file.stem
                    for key, value in file_data.items():
                        # Flatten nested dictionaries (optional - comment out if not needed)
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                client_data[f"{file_prefix}_{key}_{nested_key}"] = (
                                    nested_value
                                )
                        else:
                            client_data[f"{file_prefix}_{key}"] = value

            except json.JSONDecodeError:
                print(f"Invalid JSON format in file: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")

        # Add this folder's data to the list
        all_data.append(client_data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    return df
