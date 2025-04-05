import tarfile
import os


def extract_all_tars(directory):
    for filename in os.listdir(directory):
        if filename.endswith((".tar", ".tar.gz", ".tgz")):
            file_path = os.path.join(directory, filename)
            try:
                with tarfile.open(file_path, "r:*") as tar:
                    tar.extractall(path=directory)
                    print(f"Extracted: {filename}")
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
