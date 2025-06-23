import torch

from engines.meld_engine import Engine
from datasets.meld_dataset import meld_dataloader

EPOCHS = 10
BATCH_SIZE = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(device=device)
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_loader = meld_dataloader(
        csv_path="data/MELD/train/train_sent_emo.csv",
        video_dir="data/MELD/train/train_splits",
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = meld_dataloader(
        csv_path="data/MELD/test/test_sent_emo.csv",
        video_dir="data/MELD/test/output_repeated_splits",
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    results = engine.train(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        scheduler=scheduler
    )

    print("Training completed.")
    print("Results:", results)

if __name__ == "__main__":
    main()