import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import GradScaler, autocast, nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from model.bilstm import BiLSTMNER


class Runner:
    """Bench for training, validating and testing performance of the NER Model.
    """

    def __init__(self, learning_rate: float, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, input_dim: int, output_dim: int, epochs: int,
                 weight_filename: str, label2tag: dict[int, str], patience: int = 2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.model = BiLSTMNER(input_dim=input_dim,
                               output_dim=output_dim).to(self.device)
        self.optimiser = Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.patience = patience
        self.weight_file = weight_filename
        self.label2tag = label2tag
        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def train(self):
        """Trains the model on the configured number of epochs with early stopping if needed. Saves models
        if performance improves.
        """
        counter = 0
        best_val_loss = float('inf')

        for epoch in tqdm(range(self.epochs), desc="Training progress"):
            train_loss = 0
            # Complete one training
            self.model.train()
            for tokens, tags, masks in self.train_loader:
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                masks = masks.to(self.device)

                self.optimiser.zero_grad()
                with autocast(device_type=self.device, enabled=torch.cuda.is_available()):
                    loss = self.model(x=tokens, tags=tags, mask=masks)
                train_loss += loss.item()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimiser.step()

            train_loss /= len(self.train_loader)

            # Complete one validation
            val_loss = self.validate()

            print(
                f"Epoch: {epoch + 1}/{self.epochs}, Training loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}")

            # Early stopping or update best loss and save the model.
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                self.save_model(loss=best_val_loss)
            else:
                counter += 1
                if counter <= self.patience:
                    break

    def validate(self) -> float:
        """Validates and returns the validation loss to decide whether to continue training or stop.

        Returns:
            float: Validation loss of the current model instance.
        """
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for tokens, tags, masks in self.val_loader:
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                masks = masks.to(self.device)

                with autocast(device_type=self.device, enabled=torch.cuda.is_available()):
                    loss += self.model(x=tokens, tags=tags, mask=masks).item()

        loss /= len(self.val_loader)

        return loss

    def test(self):
        """Measures the performance of the model (precision, recall, F1) on a separate test data.
        """
        self.load_model()
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for tokens, tags, masks in self.test_loader:
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                masks = masks.to(self.device)

                with autocast(device_type=self.device, enabled=torch.cuda.is_available()):
                    predictions = self.model(tokens, tags=None, mask=masks)

                for pred_seq, true_seq, mask in zip(predictions, tags, masks):
                    true_seq = true_seq[mask.bool()].tolist()
                    all_labels.extend(true_seq)
                    all_preds.extend(pred_seq[:len(true_seq)])

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("Classification Report:")
        label_names = [self.label2tag[idx] for idx in all_labels]
        pred_names = [self.label2tag[idx] for idx in all_preds]
        print(classification_report(label_names, pred_names))

    def predict(self, sentence: str, word2idx: dict[str, int], tag2label: dict[str, int], embeddings: np.ndarray) -> list[str]:
        words = sentence.strip().lower().split(" ")
        with torch.no_grad():
            # Unsqueeze is needed to account for the batch dimension.
            # Get the mask
            mask = torch.ByteTensor(
                [1 if word != "<PAD>" else 0 for word in words]).unsqueeze(0).to(self.device)
            # Get the tokens and then their embeddings
            tokens_vector = torch.stack(
                [torch.tensor(embeddings[word2idx.get(word, word2idx["<UNK>"])]) for word in words]).unsqueeze(0).to(self.device)
            with autocast(device_type=self.device, enabled=torch.cuda.is_available()):
                predictions = self.model(
                    tokens_vector, tags=None, mask=mask)[0]
        prediction_labels = [self.label2tag[label] for label in predictions]

        return prediction_labels

    def save_model(self, loss: float):
        """Save the model with the loss in the saved file name.

        Args:
            loss (float): The loss to add to the file name to save the weights.
        """
        if not os.path.exists("weights"):
            os.makedirs("weights")

        filename = f"bilstm_crf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_loss_{loss:.2f}.pth"

        filepath = os.path.join("weights", filename)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self):
        """Load the model from the configured file name in self.weight_file.
        """
        if not os.path.exists("weights"):
            print("No weights directory present, probably no training happened.")
            sys.exit(1)

        filepath = os.path.join("weights", self.weight_file)
        self.model.load_state_dict(torch.load(filepath))
