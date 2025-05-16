import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    def __init__(self, conll_data: list[list[str], list[str]], word2idx: dict[str, int], tag2idx: dict[str, int], word_embeddings: np.ndarray):
        """Create the dataset instance from the provided CoNLL data loaded from the files.

        Args:
            conll_data (list[list[str], list[str]]): The data loaded from the CoNLL dataset file.
            word2idx (dict[str, int]): The word to id dictionary created as part of GloVe processing.
            tag2idx (dict[str, int]): The tag to id dictionary created as part of loading CoNLL dataset.
            word_embeddings (numpy.ndarray): The corresponding embedding for the index of the word returned by word2idx.
        """
        self.data = conll_data

        # Check if word2idx has <PAD> and <UNK>
        if "<PAD>" not in word2idx.keys() or "<UNK>" not in word2idx.keys():
            raise ValueError(
                "word2idx is missing crucial keys, check the value passed.")

        # Same check for tag2idx only for <PAD>
        if "<PAD>" not in tag2idx.keys():
            raise ValueError(
                "tag2idx is missing <PAD>, check the value passed.")

        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max(len(s) for s, _ in conll_data)
        self.embeddings = torch.tensor(word_embeddings, dtype=torch.float32)

        # The final dataset to pull items from
        self.samples = []
        # Pad them to be the same length and generate mask for the CRF part
        for tokens, tags in conll_data:
            tokens, tags = tokens[:], tags[:]
            while len(tokens) < self.max_len:
                tokens.append("<PAD>")
                tags.append("<PAD>")

            token_vectors = [self.embeddings[self.word2idx.get(
                token, self.word2idx["<UNK>"])] for token in tokens]
            tag_ids = [self.tag2idx.get(tag) for tag in tags]
            mask = [1 if token != "<PAD>" else 0 for token in tokens]

            self.samples.append(
                (torch.stack(token_vectors), torch.LongTensor(tag_ids), torch.BoolTensor(mask)))

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.LongTensor, torch.BoolTensor]:
        """Function that pytorch uses to create batch for an epoch through the dataloader

        Args:
            index (int): Index of the item to fetch from the dataset.

        Returns:
            tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]: Returns the token embeddings as tensors, tag ids as tensors and the mask for the CRF.
        """
        return self.samples[index]

    def __len__(self):
        """Returns length of the dataset
        """
        return len(self.data)


def create_data_loader(conll_data: list[list[str], list[str]], word2idx: dict[str, int],
                       tag2idx: dict[str, int], word_embeddings: np.ndarray, batch_size: int = 16,
                       num_workers: int = 4, is_train: bool = True) -> DataLoader:
    """Create the dataloader for the corresponding dataset

    Args:
        conll_data (list[list[str], list[str]]): The data loaded from the CoNLL dataset file.
        word2idx (dict[str, int]): The word to id dictionary created as part of GloVe processing.
        tag2idx (dict[str, int]): The tag to id dictionary created as part of loading CoNLL dataset.
        is_train (bool, optional): Do shuffling if it is a training set. Defaults to True.

    Returns:
        DataLoader: PyTorch dataloader instance to be used in the actual training.
    """
    dataset = NERDataset(conll_data, word2idx, tag2idx, word_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, num_workers=num_workers, pin_memory=True)

    return dataloader
