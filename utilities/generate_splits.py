import os
import random
import csv

if __name__ == "__main__":
    """
    This script generates train, validation, and test splits from a dataset organized in client directories.
    It collects all file paths, shuffles them, and divides them into splits based on predefined ratios.
    The splits are saved as CSV files in the 'splits' directory.

    Usage:
    - Place the dataset in the 'data' directory, with each client's files in separate subdirectories.
    - Run the script to generate 'train_split.csv', 'val_split.csv', and 'test_split.csv' in the 'splits' directory.
    """

    # Seed the random number generator for reproducibility
    random.seed(42)

    # Define the directory containing client directories
    data_dir = "data"
    output_dir = "splits"
    os.makedirs(output_dir, exist_ok=True)

    # Define split ratios
    train_ratio = 0.9
    val_ratio = 0.05
    test_ratio = 0.05

    # Collect all file paths from client directories

    file_paths = []
    for client_dir in os.listdir(data_dir):
        client_path = os.path.join(data_dir, client_dir)
        if os.path.isdir(client_path):
            file_paths.append(client_path)

    # Shuffle file paths for random subsampling
    random.shuffle(file_paths)

    # Calculate split sizes
    total_files = len(file_paths)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # Create splits
    train_files = file_paths[:train_size]
    val_files = file_paths[train_size : train_size + val_size]
    test_files = file_paths[train_size + val_size :]

    # Write splits to CSV files
    def write_to_csv(file_list, output_file):
        with open(output_file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["file_path"])
            for file_path in file_list:
                writer.writerow([file_path])

    write_to_csv(train_files, os.path.join(output_dir, "train_split.csv"))
    write_to_csv(val_files, os.path.join(output_dir, "val_split.csv"))
    write_to_csv(test_files, os.path.join(output_dir, "test_split.csv"))

    print(f"Splits generated and saved in the {output_dir} directory.")
