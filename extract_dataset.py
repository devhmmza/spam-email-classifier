import zipfile
import os

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

if __name__ == "__main__":
    zip_path = "data/spam.zip"
    extract_to = "data/"
    extract_zip(zip_path, extract_to)

    