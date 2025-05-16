from model.bilstm import BiLSTMNER

from datetime import datetime
import os
import sys
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report


class Runner:
    def __init__(self, learning_rate: float, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, input_dim: int, output_dim: int, epochs: int,
                 weight_filename: str, label2tag: dict[int, str], patience: int = 2):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.model = BiLSTMNER(input_dim=input_dim, output_dim=output_dim)
        self.optimiser = Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.patience = patience
        self.weight_file = weight_filename
        self.label2tag = label2tag

    def train(self):
        counter = 0
        best_val_loss = float('inf')
        train_loss = 0
        for epoch in tqdm(range(self.epochs), desc="Training progress"):
            # Complete one training
            self.model.train()
            for tokens, tags, masks in self.train_loader:
                self.optimiser.zero_grad()
                loss = self.model(x=tokens, tags=tags, mask=masks)
                train_loss += loss.item()
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
                self.save_model()
            else:
                counter += 1
                if counter <= self.patience:
                    break

    def validate(self):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for tokens, tags, masks in self.val_loader:
                loss += self.model(x=tokens, tags=tags, mask=masks).item()

        loss /= len(self.val_loader)

        return loss

    def test(self):
        self.load_model()
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for tokens, tags, masks in self.test_loader:
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

    def save_model(self, loss):
        if not os.path.exists("weights"):
            os.makedirs("weights")

        filename = f"bilstm_crf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_loss_{loss}.pth"

        filepath = os.path.join("weights", filename)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self):
        if not os.path.exists("weights"):
            print("No weights directory present, probably no training happened.")
            sys.exit(1)

        filepath = os.path.join("weights", self.weight_file)
        self.model.load_state_dict(torch.load(filepath))
