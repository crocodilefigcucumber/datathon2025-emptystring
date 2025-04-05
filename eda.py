import numpy as np
import pandas as np
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utilities.collect_csv import collect_to_csv

if __name__ == "__main__":
    mode = "train"  # Default mode

    if mode in ["train", "test", "val"]:
        # get list of clients
        split_path = "splits/" + mode + "_split.csv"

    clients = pd.read_csv(split_path)["file_path"].tolist()

    dataset_csv = mode + "_csv.csv"
    if not os.path.exists(dataset_csv):
        clients_dataframe = collect_to_csv(clients=clients, filename=dataset_csv)
    else:
        clients_dataframe = pd.read_csv(dataset_csv)

    # columns = clients_dataframe.columns

    # gaps = np.array([sum(np.isna(clients_dataframe[col])) for col in columns])
    # incomplete = gaps > 0

    # sns.set_theme()
    # pl1 = sns.relplot(y=gaps[incomplete], x=columns[incomplete])
    # pl1.set_xticklabels(rotation=90)
    # plt.show()

    # check for correlations between len
