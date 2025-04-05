import numpy as np
import pandas as pd
import os
import sys
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

    print(clients_dataframe.columns)

    # --- Correlation analysis: text length vs rejection ---
    # Convert rejection label to binary
    clients_dataframe["is_rejected"] = (
        clients_dataframe["label_label"] == "Reject"
    ).astype(int)

    # Create a new DataFrame with text lengths
    text_lengths = pd.DataFrame()
    for col in clients_dataframe.columns:
        if col in ["label_label", "is_rejected"]:
            continue
        lengths = clients_dataframe[col].apply(
            lambda x: len(str(x)) if pd.notnull(x) else 0
        )
        if lengths.nunique() > 1:  # Skip constant columns
            text_lengths[col + "_len"] = lengths

    # Add label column
    text_lengths["is_rejected"] = clients_dataframe["is_rejected"]

    # Compute correlation matrix
    correlation_matrix = text_lengths.corr()
    correlation_with_label = (
        correlation_matrix["is_rejected"]
        .drop("is_rejected")
        .sort_values(ascending=False)
    )

    print("Top correlations with Rejected label:")
    print(correlation_with_label)

    # Plot the correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_label.values, y=correlation_with_label.index)
    plt.title("Correlation between Text Field Lengths and 'Rejected' Label")
    plt.xlabel("Correlation with Rejected")
    plt.tight_layout()
    plt.show()
