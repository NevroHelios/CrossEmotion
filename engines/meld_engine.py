import torch
from torch import nn
from typing import Optional
from datasets.meld_dataset import meld_dataloader
from models.meld_model import MultimodalSentimentModel
from tqdm import tqdm
import os

class Engine:
    """Engine class for training and evaluating models. Tailored for the MELD dataset."""
    def __init__(self,
                 model: nn.Module = MultimodalSentimentModel(),
                 device: torch.device = torch.device("cuda")) -> None:
        self.model = model.to(device)
        self.device = device
    
    def train(self,
              optimizer: Optional[torch.optim.Optimizer] = None,
              learning_rate: float = 1e-4,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              num_epochs: int = 10,
              batch_size: int = 4,
              ) -> None:
        training_loader = meld_dataloader(
            csv_path="data/meld/train/train_sent_emo.csv",
            video_dir="data/meld/train/train_splits",
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        test_loader = meld_dataloader(
            csv_path="data/meld/test/test_sent_emo.csv",
            video_dir="data/meld/test/output_repeated_splits",
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss: float = 0.0
            best_loss: float = 0.0
            running_accuracy_emo: float = 0.0
            running_accuracy_sent: float = 0.0
            with torch.autocast(device_type=self.device.type, enabled=True):
                for batch in tqdm(training_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                    # text
                    text_enc = batch["text_inputs"]
                    text_enc = {
                        "input_ids": text_enc["input_ids"].to(self.device),
                        "attention_mask": text_enc["attention_mask"].to(self.device),
                    }
                    #video
                    video_frames = batch["video_frames"].to(self.device)  # (batch_size, frames, channels, H, W)
                    # audio
                    audio_features = batch["audio_features"].to(self.device)
                    audio_features = audio_features.transpose(1, 2)
                    audio_lengths = torch.tensor(
                        [audio_features.shape[1]] * audio_features.shape[0], device=self.device
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

                    running_loss += loss.item()
                    emo_preds = emo_logits.argmax(dim=1)
                    sent_preds = sent_logits.argmax(dim=1)
                    running_accuracy_emo += self.evaluate(emo_labels, emo_preds)
                    running_accuracy_sent += self.evaluate(sent_labels, sent_preds)
            if scheduler is not None:
                scheduler.step()

            avg_loss = running_loss / len(training_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Emo Accuracy: {running_accuracy_emo / len(training_loader):.4f}, Sent Accuracy: {running_accuracy_sent / len(training_loader):.4f}")
            
            if best_loss < avg_loss and epoch > 0:
                best_loss = avg_loss
                self._save_model(path=f"models/model_epoch_{epoch + 1}_loss_{best_loss:.4f}.pth")
                print(f"Model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

    def evaluate(self, y_true, y_pred):
        accuracy = (y_true == y_pred).float().mean().item()
        return accuracy

    def _save_model(self,
                    path: str = "saved_models/model.pth") -> None:
        """Save the model to the specified path."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(device=device)
    engine.train(
        optimizer=torch.optim.Adam(engine.model.parameters(), lr=1e-4),
        num_epochs=10,
        batch_size=8,
    )