import os
import tarfile
import zipfile


def extract_all_archives(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith((".tar", ".tar.gz", ".tgz")):
            try:
                with tarfile.open(file_path, "r:*") as tar:
                    tar.extractall(path=directory)
                    print(f"Extracted tar: {filename}")
            except Exception as e:
                print(f"Error extracting tar {filename}: {e}")

        elif filename.endswith(".zip"):
            try:
                extract_dir = os.path.join(directory, filename[:-4])
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted zip: {filename} to {extract_dir}")
            except Exception as e:
                print(f"Error extracting zip {filename}: {e}")


if __name__ == "__main__":
    """
    This script provides functionality to extract compressed archive files
    from a specified directory. It supports `.tar`, `.tar.gz`, `.tgz`, and `.zip` formats.
    Functions:
        extract_all_archives(directory):
            Iterates through all files in the given directory and extracts
            supported archive files. Extracted `.tar`, `.tar.gz`, and `.tgz`
            files are unpacked into the same directory, while `.zip` files
            are extracted into a subdirectory named after the archive (excluding
            the `.zip` extension).
    Usage:
        Run the script directly to extract all supported archives in the
        "data/" directory:
            python unzip_data.py
    NOTE: expected directory structure:
        data/client_*
    """
    extract_all_archives("data/")
