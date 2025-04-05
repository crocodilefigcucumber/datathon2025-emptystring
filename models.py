import pandas as pd
import os
import numpy as np

from utilities.evaluate import evaluate
from utilities.collect_data import collect_enriched
from get_clean_dataframe import clean_dataframe

from rules.trusted import RM_contact
from rules.passportdate import check_passport_expiry
from rules.names_check import check_names
import sys

if __name__ == "__main__":

    rules = [check_passport_expiry, RM_contact]

    # Read mode from flags
    mode = "train"
    filename = "enriched_" + mode + ".csv"
    if not os.path.exists(filename):
        data = collect_enriched(mode=mode, filename=filename, rules=rules)
    else:
        data = pd.read_csv(filename)

    print(clean_dataframe(data))