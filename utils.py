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
    extract_all_archives("data/")
