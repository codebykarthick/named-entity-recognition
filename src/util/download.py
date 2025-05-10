import os
import requests
import zipfile
from tqdm.notebook import tqdm

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def download_glove_embeddings():
    """Function to download the GloVe embeddings from the official url and unzip to 
    src/data folder to be used for the actual NER tagging project.
    """
    if os.path.exists("data/glove.6B.100d.txt"):
        print("GloVe embeddings already exist. Skipping download.")
        return

    response = requests.get(GLOVE_URL, stream=True)
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
