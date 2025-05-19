import os
import re

import numpy as np


class DataItem:
    """A wrapper for the CoNLL data item to make it easier to manage types.
    """

    def __init__(self, tokens: list[str], tags: list[str]):
        self.tokens = tokens
        self.tags = tags

    def get_tokens(self) -> list[str]:
        """Get the tokens of a single sentence.

        Returns:
            list[str]: Tokens of the single sentence.
        """
        return self.tokens

    def get_tags(self) -> list[str]:
        """Get the tags of a single sentence.

        Returns:
            list[str]: Tags of the single sentence.
        """
        return self.tags

    def get(self) -> tuple[list[str], list[str]]:
        """Return both tokens and tags as a tuple.

        Returns:
            tuple[list[str], list[str]]: The tuple of both tokens and their tags for a sentence.
        """
        return (self.tokens, self.tags)


def load_glove_embeddings(glove_path: str = "data/glove.6B.100d.txt") -> tuple[dict[str, int], np.ndarray, int]:
    """Load the glove embeddings from the mentioned file along with the appropriate ID dictionary
    for easy access.

    Args:
        glove_path (str, optional): The path to the GloVe embedding file downloaded before. Defaults to "data/glove.6B.100d.txt".

    Returns:
        tuple[dict[str, int], np.ndarray, int]: Returns the index of each word in the vocabulary, their corresponding embedding list and the dimensions used.
    """
    pattern = r".*?(\d+)d"
    match = re.search(pattern, glove_path)
    if match:
        dimensions = int(match.group(1))
    else:
        raise ValueError("Could not extract dimension from filename")

    # For UNK reproducibility
    np.random.seed(42)

    word2idx = {}
    embeddings = []

    # Handle padding and unknown token
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    embeddings.append(np.zeros(dimensions))
    embeddings.append(np.random.uniform(-0.25, 0.25, dimensions))
    counter = 2

    print("Loading GloVe embeddings from file.")
    with open(glove_path, 'r') as f:
        for line in f:
            splits = line.split(" ")
            word, embedding = splits[0], list(map(float, splits[1:]))
            word2idx[word] = counter
            embeddings.append(embedding)
            counter += 1

    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Loaded {len(word2idx)} words into vocabulary from GloVe.")

    return (word2idx, embeddings, dimensions)


def load_conll_file(file_path: str) -> list[DataItem]:
    """Loads a single file from the CoNLL Dataset.

    Args:
        file_path (str): The path to the dataset file to load.
        desc (str): The description to be displayed in the progress bar of tqdm.

    Returns:
        list[list[str], list[str]]: Returns the list of word list and their corresponding token list for each sentence.
    """
    words = []
    tags = []
    data = []

    print(f"Loading CoNLL data from {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            # Skip the header
            if "-docstart-" in line.lower():
                continue

            # If line is blank finish aggregation for the current line
            # only if there's something in the aggregate
            if line == "\n" and len(words) != 0:
                data.append(DataItem(words, tags))
                words = []
                tags = []
            elif line != "\n":
                items = line.strip().lower().split(" ")
                words.append(items[0])
                tags.append(items[-1])

        if len(words) > 0:
            data.append(DataItem(words, tags))

    print(f"Dataset loaded.")
    return data


def load_conll_dataset(data_path: str = "data/conll2003/") -> tuple[list[DataItem], list[DataItem], list[DataItem]]:
    """Wrapper function to load train, test and validation datasets of the CoNLL dataset.

    Args:
        data_path (str, optional): The path to the downloaded dataset. Defaults to "data/conll2003/".

    Returns:
        tuple[list[list[str], list[str]], list[list[str], list[str]], list[list[str], list[str]]]: Returns the training, test and validation dataset.
    """
    train_path = os.path.join(data_path, "train.txt")
    test_path = os.path.join(data_path, "test.txt")
    valid_path = os.path.join(data_path, "valid.txt")

    train_set = load_conll_file(train_path)
    test_set = load_conll_file(test_path)
    valid_set = load_conll_file(valid_path)

    return (train_set, test_set, valid_set)
