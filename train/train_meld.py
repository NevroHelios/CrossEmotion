import torch
import os
import argparse
import sys
import torchaudio

from engines.meld_engine import Engine
from datasets.meld_dataset import meld_dataloader
from setup.install_ffmpeg import install_ffmpeg

EPOCHS = 10
BATCH_SIZE = 10


#  aws sagemaker configs
SM_MODEL_DIR = "/opt/ml/model"
SM_CHANNEL_TRAINING = os.environ.get(
    "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
)
SM_CHANNEL_VALIDATION = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
)
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MELD model")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=SM_CHANNEL_TRAINING,
        help="Directory for training data",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=SM_CHANNEL_VALIDATION,
        help="Directory for validation data",
    )
    parser.add_argument(
        "--test-dir", type=str, default=SM_CHANNEL_TEST, help="Directory for test data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=SM_MODEL_DIR,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu)",
    )
    return parser.parse_args()


def main():

    install_ffmpeg()  # Ensure FFmpeg is installed

    audio_backends = torchaudio.get_audio_backend()
    if audio_backends and "ffmpeg" not in audio_backends:
        print("FFmpeg is not available. Please install FFmpeg to process audio files.")
        sys.exit(69)

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # in mb
        print(f"Memory allocated on CUDA device: {memory_allocated:.2f} MB")

    engine = Engine(device=device)
    optimizer = torch.optim.Adam(engine.model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_loader = meld_dataloader(
        csv_path=os.path.join(args.train_dir, "train_sent_emo.csv"),
        video_dir=os.path.join(args.train_dir, "train_splits"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = meld_dataloader(
        csv_path=os.path.join(args.validation_dir, "dev_sent_emo.csv"),
        video_dir=os.path.join(args.validation_dir, "dev_splits_complete"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = meld_dataloader(
        csv_path=os.path.join(args.test_dir, "test_sent_emo.csv"),
        video_dir=os.path.join(args.test_dir, "test_splits_complete"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    results = engine.train(
        train_dataloader=train_loader,
        test_dataloader=dev_loader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        scheduler=scheduler,
    )

    print("Training completed.")
    print("Train Results:", results)

    print("Beginning evaluation on test set...")
    test_results = engine.test(test_loader)

    print("Test Results:", test_results)

if __name__ == "__main__":
    main()
