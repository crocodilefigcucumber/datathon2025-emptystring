import os
import pandas as pd
import json


def print_sorted_false_negatives():
    # Read the false negatives from the CSV file
    false_negatives_file = "false_negatives.csv"
    if not os.path.exists(false_negatives_file):
        print(f"File {false_negatives_file} not found.")
        return

    false_negatives = pd.read_csv(false_negatives_file, header=None)[0].tolist()

    # Sort the false negatives alphabetically
    false_negatives.sort()

    for client_id in false_negatives:
        client_dir = os.path.join("data", client_id)
        if not os.path.exists(client_dir):
            print(f"Directory {client_dir} not found. Skipping...")
            continue

        # List all .json files in the client directory
        json_files = [f for f in os.listdir(client_dir) if f.endswith(".json")]
        if not json_files:
            print(f"No JSON files found in {client_dir}. Skipping...")
            continue

        print(f"Client: {client_id}")
        print("JSON files:")
        for json_file in json_files:
            print(f"  - {json_file}")
            json_path = os.path.join(client_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    json_content = json.load(f)
                    print(
                        json.dumps(json_content, indent=2)
                    )  # Pretty-print JSON content
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        input("Press Enter to continue to the next client...")


if __name__ == "__main__":
    print_sorted_false_negatives()
