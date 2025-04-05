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
        print(f"Processing folder: {client_name}")

        results.at[client_name, "accept"] = "accept"
        for func in rules:
            accept, comment = func(client_name)
            if not accept:
                results.at[client_name, "accept"] = "reject"
                results.at[client_name, "comment"] += ", " + comment

    print(results)
    print(sum(results["accept"] == "reject"))
