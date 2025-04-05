import pandas as pd
import os
import numpy as np

from utilities.unzip_data import extract_all_archives
from passportdate import check_passport_expiry
import sys

if __name__ == "__main__":
    rules = [check_passport_expiry]

    # Read mode from flags
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
        # get list of clients
        split_path = "splits/" + mode + "_split.csv"

    clients = pd.read_csv(split_path)["file_path"].tolist()
    clients = sorted(clients)
    n_clients = len(clients)

    dataset_path = "data/"

    # set up results table
    results = pd.DataFrame(data=np.zeros((n_clients, 3)))
    results.columns = ["client_name", "accept", "comment"]
    results["client_name"] = clients
    results["accept"] = [""] * n_clients
    results["comment"] = [""] * n_clients

    if not clients:
        print(f"No folders found in {dataset_path}")

    i = 0
    # check for all clients
    for folder in clients:
        # folder_path = os.path.join(dataset_path, folder)
        print(f"Processing folder: {folder}")

        # apply all rules for rejection
        results.loc[i, "accept"] = "accept"
        for func in rules:
            accept, comment = func(folder)

            # update acceptation record
            if not accept:
                results.loc[i, "accept"] = "reject"
                # record reason for rejection
                results.loc[i, "comment"] = results["comment"][i] + ", " + comment
        i += 1

print(results)

print(sum(results["accept"] == "reject"))
