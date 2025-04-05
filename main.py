import pandas as pd
import os
import numpy as np

from utilities.unzip_data import extract_all_archives
from rules.passportdate import check_passport_expiry
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
        dataset_path = "data/"
        split_path = "splits/" + mode + "_split.csv"
    else:
        raise NotImplementedError("Final mode not yet implemented.")

    clients = pd.read_csv(split_path)["file_path"].tolist()
    clients = sorted(clients)

    results = pd.DataFrame(index=clients, columns=["accept", "comment"])
    results["accept"] = ""
    results["comment"] = ""

    if not clients:
        print("No folders found.")

    for client_name, _ in results.iterrows():
        client_path = os.path.join(dataset_path, client_name)
        print(f"Processing folder: {client_path}")

        results.at[client_name, "accept"] = "Accept"
        for func in rules:
            accept, comment = func(client_path)
            if not accept:
                results.at[client_name, "accept"] = "Reject"
                results.at[client_name, "comment"] += ", " + comment

    # output results in the correct format
    results_out = results[["accept"]].copy()
    results_out.index.name = None  # remove index name

    if mode in ["train", "test", "val"]:
        results_out.to_csv(f"{mode}_results.csv", sep=";", header=False)
    else:
        results_out.to_csv(".csv", sep=";", header=False)

    print(results)
    print(sum(results["accept"] == "Reject"))
