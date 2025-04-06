import pandas as pd
import pathlib
import os
import json
import numpy as np
import sys

from rules.openai_prompt import check_discrepancy_via_llm

def llm_enriched(clients: list) -> pd.DataFrame:
    n_clients = len(clients)
    enriched = pd.DataFrame(data=np.zeros((n_clients, 3)))
    enriched.columns = ["higher_education", "employment_history", "financial"]
    enriched.index = clients

    for client in clients:
        client_folder = os.path.join(dataset_path, client)
        client_data = {"folder_name": client}  # Initialize with folder name

        # Get all json documents
        documents = list(pathlib.Path(client_folder).glob("*.json"))

        if not documents:
            print(f"No JSON files found in folder: {client}")
            continue
        
        results = check_discrepancy_via_llm(client_folder)

        enriched.loc[client] = [int(result) for result in results]
    # Convert list of dictionaries to DataFrame
    return enriched

if __name__=="__main__":
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
    clients = sorted(clients)[:3]
    print(llm_enriched(clients=clients))
