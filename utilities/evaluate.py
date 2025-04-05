import pandas as pd
import os
import numpy as np
import json
from typing import Tuple

def evaluate(results: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """
    in: results DataFrame:              Accept comment
                            client_0     Accept        
                            client_1     Reject "Because he's poor"

    out: pd.DataFrame: Confusion Matrix
            list: False Negatives: Client should be rejected but NO rule has triggered
            list: False Positives: Client should be accepted but was excluded by rule 
            list: Rules giving False Positives: the corresponding rules triggered on good case

    confusion_matrix, false_negatives, false_positives, false_positive_rules = evaluate(results)
    """
    clients = results.index

    confus = pd.DataFrame(data=np.zeros((2,2)))
    confus.columns = ["predicted: Accept", "predicted: Reject"]
    confus.index = ["Accept", "Reject"]

    false_positives = []
    fp_rules = []
    false_negatives = []

    for client in clients:

        # get client label
        client_folder = os.path.join("./data", client)
        label_file = os.path.join(client_folder, "label.json")
        with open(label_file, "r") as file:
            file_data = json.load(file)
            label = file_data["label"]
        

        # get confusion and record false assignments
        if label == "Accept":
            if results.loc[client, "Accept"] == label:
                confus.loc["Accept", "predicted: Accept"] += 1
            else:
                confus.loc["Accept", "predicted: Reject"] += 1
                false_positives.append(client)
                fp_rules.append(results.loc[client,"comment"].split(","))
        elif label == "Reject":
            if results.loc[client, "Accept"] == label:
                confus.loc["Reject", "predicted: Reject"] += 1
            else:
                confus.loc["Reject", "predicted: Accept"] += 1
                false_negatives.append(client)

    return confus, false_negatives, false_positives, fp_rules
        
