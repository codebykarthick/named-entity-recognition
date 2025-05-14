import os
import requests
import zipfile
from tqdm.notebook import tqdm


def download_glove_embeddings():
    """Function to download the GloVe embeddings from the official url and unzip to 
    src/data folder to be used for the actual NER tagging project.
    """
    if os.path.exists("data/glove.6B.100d.txt"):
        print("GloVe embeddings already exist. Skipping download.")
        return

    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    response = requests.get(glove_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    with open("glove.6B.zip", "wb") as f, tqdm(
        desc="Downloading GloVe",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("Download complete, unzipping")

    with zipfile.ZipFile("glove.6B.zip", 'r') as zip_ref:
        zip_ref.extractall("data")

    os.remove("glove.6B.zip")
    print("Extraction complete, zip file removed.")


def download_conll2003_dataset():
    """Downloads the CoNLL-2003 dataset and extracts it into the data folder."""
    dataset_url = "https://data.deepai.org/conll2003.zip"
    zip_path = "conll2003.zip"
    extract_path = "data/conll2003"
    target_file_check = os.path.join(extract_path, "train.txt")

    if os.path.exists(target_file_check):
        print("CoNLL-2003 dataset already exists. Skipping download.")
        return

    response = requests.get(dataset_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    with open(zip_path, "wb") as f, tqdm(
        desc="Downloading CoNLL-2003",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("Download complete, unzipping")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)
    print("Extraction complete, zip file removed.")
