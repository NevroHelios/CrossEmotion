from typing import Dict
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

VIDEO_BACKEND = "ffmpeg"  # "cuda"


class MeldDataset(Dataset):
    def __init__(self, csv_path: str, video_dir: str) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path).sample(frac=0.1)
        self.video_dir = Path(video_dir)
        self.transformer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # neutral, joy, sadness, anger, surprise, fear, disgust
        emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }
        # positive, neutral, negative
        sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}

        self.df["Emotion"] = self.df["Emotion"].map(emotion_map)
        self.df["Sentiment"] = self.df["Sentiment"].map(sentiment_map)

    def _read_video_frames(self, video_path: str):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        try:
            while (
                len(frames) < 60
            ):  # read 60 frames -> approx 2 sec at 30fps (might increase if needed)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))  # cause cuda out of memory
                frames.append(
                    torch.tensor(frame).float() / 255.0
                )  # normalize to [0, 1]
        except Exception as e:
            print(f"Error reading video frames from {video_path}: {e}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video file {video_path}")
        if len(frames) < 60:
            frames += [torch.zeros_like(frames[0])] * (
                60 - len(frames)
            )  # pad with zeros if less than 60 frames

        return torch.stack(frames).permute(0, 3, 1, 2)  # -> (60, 3, H, W) format

    def _extract_audio_features(self, video_path: str):
        wave, sr = torchaudio.load(video_path, backend="ffmpeg")
        if sr != 16000:
            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wave)
        wave = wave[0]  # using the first channel only
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, n_mels=64, hop_length=512
        )
        mel_spectrogram = mel(wave)
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / (
            mel_spectrogram.std() + 1e-5
        )  # Normalize

        if mel_spectrogram.shape[1] < 300:
            padding = 300 - mel_spectrogram.size(1)
            mel_spectrogram = torch.nn.functional.pad(
                mel_spectrogram, (0, padding), "constant", 0
            )
        else:
            mel_spectrogram = mel_spectrogram[:, :300]
        return mel_spectrogram.unsqueeze(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int | torch.Tensor) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(index, torch.Tensor):
                index = int(index.item())
            if index < 0 or index >= len(self.df):
                raise IndexError(
                    f"Index {index} out of range for dataset of size {len(self.df)}"
                )

            dia_id = self.df.iloc[index]["Dialogue_ID"]
            utt_id = self.df.iloc[index]["Utterance_ID"]
            video_path = os.path.join(self.video_dir, f"dia{dia_id}_utt{utt_id}.mp4")

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_path} does not exist.")

            input_tokens = self.transformer(
                self.df.iloc[index]["Utterance"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            audio_features = self._extract_audio_features(video_path)
            video_frames = self._read_video_frames(video_path)

            # return input_tokens, audio_features, video_frames, self.df.iloc[index]["Emotion"], self.df.iloc[index]["Sentiment"]
            return {
                "text_inputs": {
                    "input_ids": input_tokens["input_ids"].squeeze(0),
                    "attention_mask": input_tokens["attention_mask"].squeeze(0),
                },
                "audio_features": audio_features.squeeze(
                    0
                ),  # (1, 64, 300) -> (64, 300)
                "video_frames": video_frames,  # (60, 3, H, W)
                "emotion": self.df.iloc[index]["Emotion"],
                "sentiment": self.df.iloc[index]["Sentiment"],
            }
        except Exception as e:
            # print(f"Error processing index {index}: {e}")
            return None


def collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def meld_dataloader(
    csv_path: str,
    video_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    collate_fn=collate_fn,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = MeldDataset(csv_path=csv_path, video_dir=video_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # train_dataset = MeldDataset(csv_path="../dataset/train/train_sent_emo.csv",video_dir="../dataset/train/train_splits")
    # print(train_dataset.__len__())
    # sample = train_dataset[0]
    # print(sample)
    train_loader = meld_dataloader(
        csv_path="data/MELD/train/train_sent_emo.csv",
        video_dir="data/MELD/train/train_splits",
        batch_size=10,
        shuffle=True,
    )
    for batch in train_loader:
        batch["text_inputs"]["input_ids"].to("cuda")
        print(
            "text_inputs:", batch["text_inputs"]["input_ids"].shape
        )  # (batch_size, 128)
        print("audio_features:", batch["audio_features"].shape)  # (batch_size, 64, 300)
        print("video_frames:", batch["video_frames"].shape)  # (batch_size, 60, 3, H, W)
        print("emotion:", batch["emotion"])  # (batch_size,)
        print("sentiment:", batch["sentiment"])  # (batch_size,)
        break
