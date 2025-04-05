import pandas as pd
import os
import numpy as np

from utils import extract_all_archives
from passportdate import check_passport_expiry
from test import toy

if __name__ == "__main__":
    rules = [toy, check_passport_expiry]

    dataset_path = "./data"
    if False:
        extract_all_archives(dataset_path)

    # get list of clients
    clients = [item for item in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, item))]
    clients = sorted(clients)
    n_clients = len(clients)

    # set up results table
    results = pd.DataFrame(data=np.zeros((n_clients, 3)))
    results.columns = ["client_name", "accept", "comment"]
    results["client_name"] = clients
    results["accept"] = [""]*n_clients
    results["comment"] = [""]*n_clients

    if not clients:
        print(f"No folders found in {dataset_path}")

    i = 0
    # check for all clients
    for folder in clients:
        folder_path = os.path.join(dataset_path, folder)
        print(f"Processing folder: {folder}")

        # apply all rules for rejection
        results.loc[i, "accept"] = "accept"       
        for func in rules:
            accept, comment = func(folder_path)
            
            # update acceptation record
            if not accept:
                results.loc[i, "accept"] = "reject"
                # record reason for rejection
                results.loc[i, "comment"] = results["comment"][i] + ", " + comment
        i += 1

print(results)

print(sum(results["accept"] == "reject"))