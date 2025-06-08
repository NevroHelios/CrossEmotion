import torch
import torch.nn as nn
from torchvision import models as visoin_models
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # x = x.pooler_output
        # x = self.fc(x)
        return self.fc(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = visoin_models.video.r3d_18(weights=visoin_models.video.R3D_18_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential( # type: ignore
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        ) 
        
    def forward(self, x):
        # x = x.transpose(1, 2)  # (batch_size, frames, channels, H, W) -> (batch_size, channels, frames, H, W)
        return self.resnet(x.transpose(1, 2)) 


if __name__ == "__main__":
    from dataset_loader import meld_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = meld_dataloader(
        csv_path="../dataset/train/train_sent_emo.csv",
        video_dir="../dataset/train/train_splits",
        batch_size=4,
        shuffle=True,
    )
    data = next(iter(loader))
    text_enc = data["text_inputs"]
    print(
        text_enc["input_ids"].shape, text_enc["attention_mask"].shape
    )  # (batch_size, seq_len)

    with torch.no_grad():
        text_model = TextEncoder().to(device)
        y = text_model(
            input_ids=text_enc["input_ids"].to(device),
            attention_mask=text_enc["attention_mask"].to(device),
        )
        print(y.shape)  # (batch_size, 128)

        video_model = VideoEncoder().to(device)
        video_frames = data["video_frames"].to(device)  # (batch_size, frames, channels, H, W)
        video_features = video_model(video_frames)
        print(video_features.shape)  