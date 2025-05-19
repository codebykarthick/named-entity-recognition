import torch
from torch.nn import LSTM, Dropout, Linear, Module
from torchcrf import CRF


class BiLSTMNER(Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 384, output_dim: int = 10):
        """Define the architecture of the neural network for the NER tagger.

        Args:
            input_dim (int): The input dimensions of the network. This is equal to the number of dimensions used for embedding.
            hidden_dim (int): 
            output_dim (int): The output dimensions of the network. This is equal to the number of labels that need to be classified.
        """
        super().__init__()
        self.blstml1 = LSTM(input_dim, hidden_dim, num_layers=2)
        self.dropout = Dropout(p=0.3)
        self.dense1 = Linear(hidden_dim, 512)
        self.dense2 = Linear(512, 1024)
        self.dense3 = Linear(1024, output_dim)
        self.crf1 = CRF(output_dim, True)

    def forward(self, x: torch.Tensor, mask: torch.ByteTensor, tags: torch.LongTensor | None = None):
        """The forward propagation step of the network. Goes through a bidirectional lstm network, followed by a
        dense network and then finally passing through the CRF layer to prevent malformed tags.

        Args:
            x (torch.Tensor): The batched input of token id to predict the tag id for.
            tags (torch.LongTensor, optional): The tag id for CRF field only during training. Defaults to None.
            mask (torch.BoolTensor, optional): The mask tensor for the CRF to mask <PAD> tokens. Defaults to None.

        Returns:
            _type_: _description_
        """
        emissions, _ = self.blstml1(x)
        emissions = self.dropout(emissions)
        emissions = self.dense1(emissions)
        emissions = self.dropout(emissions)
        emissions = self.dense2(emissions)
        emissions = self.dropout(emissions)
        emissions = self.dense3(emissions)

        if tags is not None:
            # negative log likelihood loss
            return -self.crf1(emissions, tags, mask=mask)
        else:
            # predicted tag sequence
            return self.crf1.decode(emissions, mask=mask)
