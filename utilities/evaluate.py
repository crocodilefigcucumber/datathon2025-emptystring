import pandas as pd
import os
import numpy as np
import json

def evaluate(results: pd.DataFrame) -> int:
    clients = results.index

    confus = pd.DataFrame(data=np.zeros((2,2)))
    confus.columns = ["predicted: Accept", "predicted: Reject"]
    confus.index = ["Accept", "Reject"]

    rule_list = []
    rule_index = pd.DataFrame(columns=["TP", "FP"])

    for client in clients:

        # get client label
        client_folder = os.path.join("./data", client)
        label_file = os.path.join(client_folder, "label.json")
        with open(label_file, "r") as file:
            file_data = json.load(file)
            label = file_data["label"]
        
        """ # add rules to 
        triggered_rules = client["comment"].split(",")
        for rule in triggered_rules:
            if not rule in rule_list:
                rule_list.append(rule)
                rule_index.append({"TP": 0, "FP": 0})
                rule_index.index = rule_list """

        # get confusion
        if label == "Accept":
            if results.loc[client, "Accept"] == label:
                confus.loc["Accept", "predicted: Accept"] += 1
            else:
                confus.loc["Accept", "predicted: Reject"] += 1
        elif label == "Reject":
            if results.loc[client, "Accept"] == label:
                confus.loc["Reject", "predicted: Reject"] += 1
            else:
                confus.loc["Reject", "predicted: Accept"] += 1

    return confus
        
