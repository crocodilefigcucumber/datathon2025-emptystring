import numpy as np
import pandas as np
import os
import pathlib
import json
import sys
import pandas as pd

def collect_to_csv(clients: list, filename: str) -> pd.DataFrame:
    all_data = []

    for client in clients:
        client_folder = os.path.join(".", client)
        client_data = {"folder_name": client}  # Initialize with folder name
        
        # Get all json documents
        documents = list(pathlib.Path(client_folder).glob('*.json'))
        
        if not documents:
            print(f"No JSON files found in folder: {client}")
            all_data.append(client_data)
            continue
        
        # Process each JSON file in the folder
        for json_file in documents:
            try:
                # Read the JSON file
                with open(json_file, 'r') as file:
                    file_data = json.load(file)

                    file_prefix = json_file.stem
                    for key, value in file_data.items():
                        # Flatten nested dictionaries (optional - comment out if not needed)
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                client_data[f"{file_prefix}_{key}_{nested_key}"] = nested_value
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
    df.to_csv(filename)
    return df

if __name__=="__main__":
    mode = "test"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1].strip().lower() in ["--help", "-h"]:
            print("Usage: python main.py [mode]")
            print("mode: One of train, test, val, final")
            sys.exit(0)
        mode = sys.argv[1].strip().lower()
    if mode not in ["train", "test", "val", "final"]:
        raise ValueError("Invalid mode. Please enter one of: train, test, val, final.")

    if mode in ["train", "test", "val"]:
        # get list of clients
        split_path = "splits/" + mode + "_split.csv"

    clients = pd.read_csv(split_path)["file_path"].tolist()

    dataset_csv = mode + "_csv.csv"
    if not os.path.exists(dataset_csv):
        clients_dataframe = collect_to_csv(clients=clients, filename=dataset_csv)
    else:
        clients_dataframe = pd.read_csv(dataset_csv)
    
    print(clients_dataframe)