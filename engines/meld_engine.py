import os
from typing import Dict, Tuple 

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter # type: ignore
from tqdm import tqdm

from datasets.meld_dataset import meld_dataloader
from models.meld_model import MultimodalSentimentModel


class Engine:
    """Engine class for training and evaluating models. Tailored for the MELD dataset."""

    def __init__(
        self,
        model: nn.Module = MultimodalSentimentModel(),
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter(log_dir="runs/meld_experiment")

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer= None,
        learning_rate: float = 1e-4,
        scheduler = None,
        num_epochs: int = 10,
        batch_size: int = 4,
        verbose: bool = True
    ) -> Dict:
        """Default training method for the MELD dataset.
        and using Adam optimizer with a learning rate of 1e-4."""


        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.results = {
            "train_loss": [],
            "train_emo_accuracy": [],
            "train_sent_accuracy": [],
            "test_loss": [],
            "test_emo_accuracy": [],
            "test_sent_accuracy": [],
        }
        best_loss = float("inf")
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_emo_accuracy, train_sent_accuracy = self._train_step(
                dataloader=train_dataloader,
                optimizer=optimizer,
                device_type=device_type,
                verbose=verbose
            )

            test_loss, test_emo_accuracy, test_sent_accuracy = self._test_step(
                dataloader=test_dataloader,
                verbose=verbose
            )

            self.results["train_loss"].append(train_loss)
            self.results["train_emo_accuracy"].append(train_emo_accuracy)
            self.results["train_sent_accuracy"].append(train_sent_accuracy)

            if test_loss < best_loss:
                best_loss = test_loss
                self._save_model()

            self.results["test_loss"].append(test_loss)
            self.results["test_emo_accuracy"].append(test_emo_accuracy)
            self.results["test_sent_accuracy"].append(test_sent_accuracy)

            self._log_results(self.results, epoch=epoch)

            if scheduler:
                scheduler.step()

        self.writer.close()

        return self.results

    def _log_results(self, results: dict, epoch: int) -> None:
        self.writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={
                "Train Loss": results["train_loss"][-1],
                "Test Loss": results["test_loss"][-1],
            },
            global_step=epoch,
        )
        self.writer.add_scalars(
            main_tag="Emotion Accuracy",
            tag_scalar_dict={
                "Train Emo Accuracy": results["train_emo_accuracy"][-1],
                "Test Emo Accuracy": results["test_emo_accuracy"][-1],
            },
            global_step=epoch,
        )
        self.writer.add_scalars(
            main_tag="Sentiment Accuracy",
            tag_scalar_dict={
                "Train Sent Accuracy": results["train_sent_accuracy"][-1],
                "Test Sent Accuracy": results["test_sent_accuracy"][-1],
            },
            global_step=epoch,
        )

    def test(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """Test the model on the provided dataloader and return average loss and accuracy (emo, sent)."""
        avg_loss, avg_emo_accuracy, avg_sent_accuracy = self._test_step(dataloader)
        print(f"Test Loss: {avg_loss:.4f}, Emo Accuracy: {avg_emo_accuracy:.4f}, Sent Accuracy: {avg_sent_accuracy:.4f}")
        return avg_loss, avg_emo_accuracy, avg_sent_accuracy

    def _test_step(
        self, dataloader: torch.utils.data.DataLoader, verbose: bool = True
    ) -> Tuple[float, float, float]:
        self.model.eval()
        running_loss: float = 0.0
        emo_correct: float = 0.0
        sent_correct: float = 0.0
        total_samples: int = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None:
                    if verbose: print("Skipping empty test batch")
                    continue
                batch_size = batch["text_inputs"]["input_ids"].shape[0]
                total_samples += batch_size

                text_enc = batch["text_inputs"]
                text_enc = {
                    "input_ids": text_enc["input_ids"].to(self.device),
                    "attention_mask": text_enc["attention_mask"].to(self.device),
                }
                # video
                video_frames = batch["video_frames"].to(
                    self.device
                )  # (batch_size, frames, channels, H, W)
                # audio
                audio_features = batch["audio_features"].to(self.device)
                audio_features = audio_features.transpose(1, 2)
                audio_lengths = torch.tensor(
                    [audio_features.shape[1]] * audio_features.shape[0],
                    device=self.device,
                )

                emo_labels = batch["emotion"].to(self.device)
                sent_labels = batch["sentiment"].to(self.device)

                emo_logits, sent_logits = self.model(
                    text_inputs=text_enc,
                    video_frames=video_frames,
                    audio_features=audio_features,
                    audio_lengths=audio_lengths,
                )

                emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_labels)
                sent_loss = nn.CrossEntropyLoss()(sent_logits, sent_labels)

                loss = emo_loss + sent_loss
                running_loss += loss.item() * batch_size
                emo_preds = emo_logits.argmax(dim=1)
                sent_preds = sent_logits.argmax(dim=1)
                emo_correct += (emo_preds == emo_labels).sum().item()
                sent_correct += (sent_preds == sent_labels).sum().item()

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        avg_emo_accuracy = emo_correct / total_samples if total_samples > 0 else 0.0
        avg_sent_accuracy = sent_correct / total_samples if total_samples > 0 else 0.0
        if verbose: print(
            f"\n\nTest Loss: {avg_loss:.4f}, Emo Accuracy: {avg_emo_accuracy:.4f}, Sent Accuracy: {avg_sent_accuracy:.4f}"
        )
        return avg_loss, avg_emo_accuracy, avg_sent_accuracy

    def _train_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device_type: str = "cuda",
        verbose: bool = True
    ) -> Tuple[float, float, float]:
        """a single training step for the model.

        Return:
        avg_loss, avg_emo_accuracy, avg_sent_accuracy"""
        self.model.train()
        running_loss: float = 0.0
        sent_correct: float = 0.0
        emo_correct: float = 0.0
        total_samples: int = 0

        # with torch.autocast(device_type=device_type, enabled=True):
        for batch in tqdm(dataloader):
            if batch is None:
                if verbose: print("Skipping empty train batch")
                continue
            batch_size = batch["text_inputs"]["input_ids"].shape[0]
            total_samples += batch_size

            # text
            text_enc = batch["text_inputs"]
            text_enc = {
                "input_ids": text_enc["input_ids"].to(self.device),
                "attention_mask": text_enc["attention_mask"].to(self.device),
            }
            # video
            video_frames = batch["video_frames"].to(
                self.device
            )  # (batch_size, frames, channels, H, W)
            # audio
            audio_features = batch["audio_features"].to(self.device)
            audio_features = audio_features.transpose(1, 2)
            audio_lengths = torch.tensor(
                [audio_features.shape[1]] * audio_features.shape[0],
                device=self.device,
            )

            emo_labels = batch["emotion"].to(self.device)
            sent_labels = batch["sentiment"].to(self.device)

            emo_logits, sent_logits = self.model(
                text_inputs=text_enc,
                video_frames=video_frames,
                audio_features=audio_features,
                audio_lengths=audio_lengths,
            )

            emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_labels)
            sent_loss = nn.CrossEntropyLoss()(sent_logits, sent_labels)
            loss = emo_loss + sent_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            emo_preds = emo_logits.argmax(dim=1)
            sent_preds = sent_logits.argmax(dim=1)
            emo_correct += (emo_preds == emo_labels).sum().item()
            sent_correct += (sent_preds == sent_labels).sum().item()

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        avg_emo_accuracy = emo_correct / total_samples if total_samples > 0 else 0.0
        avg_sent_accuracy = sent_correct / total_samples if total_samples > 0 else 0.0
        if verbose: print(
            f"\n\nTrain Loss: {avg_loss:.4f}, Emo Accuracy: {avg_emo_accuracy:.4f}, Sent Accuracy: {avg_sent_accuracy:.4f}"
        )
        return avg_loss, avg_emo_accuracy, avg_sent_accuracy

    def evaluate(self, y_true, y_pred):
        accuracy = (y_true == y_pred).float().mean().item()
        return accuracy

    def _save_model(self, model_save_path: str = "saved_models/model.pth") -> None:
        """Save the model to the specified path."""
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(device=device)
    batch_size = 8
    training_loader = meld_dataloader(
            csv_path="data/MELD/train/train_sent_emo.csv",
            video_dir="data/MELD/train/train_splits",
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    test_loader = meld_dataloader(
        csv_path="data/MELD/test/test_sent_emo.csv",
        video_dir="data/MELD/test/output_repeated_splits",
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    engine.train(
        train_dataloader=training_loader,
        test_dataloader=test_loader,
        optimizer=torch.optim.Adam(engine.model.parameters(), lr=1e-4),
        num_epochs=10,
        batch_size=8,
    )
 
