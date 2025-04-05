import pandas as pd
import os
import numpy as np

from utilities.unzip_data import extract_all_archives
from utilities.evaluate import evaluate
from utilities.collect_data import collect_enriched

from rules.trusted import RM_contact
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
    rules = [check_passport_expiry, RM_contact]

    # Read mode from flags
    mode = "train"
    filename = "enriched_" + mode + ".csv"
    if not os.path.exists(filename):
        data = collect_enriched(mode=mode, filename=filename, rules=rules)
    else:
        data = pd.read_csv(filename)

    print(data.columns)