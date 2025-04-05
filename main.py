import pandas as pd
import os
import numpy as np

from utilities.unzip_data import extract_all_archives
from utilities.evaluate import evaluate
from rules.passportdate import check_passport_expiry
from rules.names_check import check_names
import sys

if __name__ == "__main__":
    """
    Main script for processing client data and applying validation rules.
    This script reads a dataset split file, processes client folders, and applies a set of rules
    to determine whether each client folder is Accepted or Rejected. The results are saved to a CSV file.
    Modules:
        - pandas: For data manipulation and analysis.
        - os: For interacting with the operating system.
        - numpy: For numerical operations.
        - sys: For handling command-line arguments.
        - utilities.unzip_data.extract_all_archives: Custom module for extracting archives.
        - rules.passportdate.check_passport_expiry: Custom rule for checking passport expiry.
    Usage:
        python main.py [mode]
        mode: One of "train", "test", "val", "final".
        - "train", "test", "val": Process the respective dataset split.
        - "final": Not yet implemented.
    Functions:
        - check_passport_expiry(client_path): A rule function to validate client data.
    Command-line Arguments:
        - --help, -h: Display usage information.
    Raises:
        - ValueError: If an invalid mode is provided.
        - NotImplementedError: If the "final" mode is used (not implemented).
    Outputs:
        - A CSV file containing the results of the validation for each client folder.
        The file is named based on the mode (e.g., "train_results.csv").
    Processing Steps:
        1. Read the mode from command-line arguments or default to "train".
        2. Validate the mode and determine the dataset and split file paths.
        3. Load the list of client folders from the split file.
        4. Apply validation rules to each client folder.
        5. Save the results to a CSV file.
    """
    rules = [check_names]

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

    results = pd.DataFrame(index=clients, columns=["Accept", "comment"])
    results["Accept"] = ""
    results["comment"] = ""

    if not clients:
        print("No folders found.")

    for client_name, _ in results.iterrows():
        client_path = os.path.join(dataset_path, client_name)
        print(f"Processing folder: {client_path}")

        results.at[client_name, "Accept"] = "Accept"
        for func in rules:
            Accept, comment = func(client_path)
            if not Accept:
                results.at[client_name, "Accept"] = "Reject"
                results.at[client_name, "comment"] += ", " + comment

    # output results in the correct format
    results_out = results[["Accept"]].copy()
    results_out.index.name = None  # remove index name

    if mode in ["train", "test", "val"]:
        results_out.to_csv(f"{mode}_results.csv", sep=";", header=False)
    else:
        results_out.to_csv("emptystring.csv", sep=";", header=False)

    confus, false_negatives, false_positives, fp_rules = evaluate(results)
    print(confus)
    print("No. rejections:", sum(results["Accept"] == "Reject"))

    for reason, clients in zip(fp_rules, false_positives):
        print(f"False positive for {reason}: {clients}")
