"""iffed engine module for training IFeedModel on IFeed dataset."""

import torch
from torch import nn
from tqdm import tqdm

from models.ifeed_model import IFeedModel


class IFeedEngine:
    def __init__(self, device: str | torch.device) -> None:
        self.model = IFeedModel()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int,
    ):

        with torch.autocast(device_type=self.device.type, enabled=True):
            self.model.train()
            for epoch in range(num_epochs):
                running_loss, running_acc = 0, 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                    image_data, labels = batch
                    image_data, labels = image_data.to(self.device), labels.to(
                        self.device
                    )

                    optimizer.zero_grad()
                    outputs = self.model(image_data)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * image_data.size(0)
                    _, preds = torch.max(torch.softmax(outputs, dim=1), 1)
                    running_acc += self.calculate_acc(labels, preds) * image_data.size(
                        0
                    )
                epoch_loss = running_loss / len(train_loader.dataset)  # type: ignore
                epoch_acc = running_acc / len(train_loader.dataset)  # type: ignore
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
                )

    def calculate_acc(self, y_true, y_pred):
        return torch.eq(y_true, y_pred).sum().item() / len(y_true)


if __name__ == "__main__":
    from datasets.ifeed_dataset import ifeed_dataloader

    train_loader = ifeed_dataloader(
        data_dir="data/ifeed/IFEED_Base/Training", batch_size=32, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = IFeedEngine(device=device)

    optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    engine.train(
        optimizer=optimizer, loss_fn=loss_fn, train_loader=train_loader, num_epochs=10
    )
