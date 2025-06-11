import torch
import torch.nn as nn
from torchvision import models as visoin_models
from torchaudio import models as audio_models
from transformers import BertModel
import tqdm


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 128)

        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT params

    def forward(self, input_ids, attention_mask):
        # x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # x = x.pooler_output
        # x = self.fc(x)
        return self.fc(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        )


class VideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = visoin_models.video.r3d_18(
            weights=visoin_models.video.R3D_18_Weights.DEFAULT
        )

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Sequential(  # type: ignore
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # x = x.transpose(1, 2)  # (batch_size, frames, channels, H, W) -> (batch_size, channels, frames, H, W)
        return self.resnet(x.transpose(1, 2))


class AudioEncoder(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.conformer = audio_models.Conformer(
            input_dim=64,
            num_heads=8,
            ffn_dim=1024,
            num_layers=12,
            depthwise_conv_kernel_size=31,
        )
        for param in self.conformer.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        inputs,  # input: (batch_size, num_frames, feature_dim)
        lengths,  # lengths: (batch_size, )
    ):
        x, _ = self.conformer(inputs, lengths)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
        # return self.fc(self.conformer(inputs, lengths)[0].mean(dim=2))


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.emo_clf = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7),  # 7 emotions
            nn.Softmax(dim=1),
        )

        self.sent_clf = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3 sentiments
            nn.Softmax(dim=1),
        )

    def forward(self, text_inputs, video_frames, audio_features, audio_lengths):
        text_features = self.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features, audio_lengths)

        combined_features = torch.cat(
            (text_features, video_features, audio_features), dim=1
        )
        combined_features = self.fusion_layer(combined_features)

        emo_logits = self.emo_clf(combined_features)
        sent_logits = self.sent_clf(combined_features)

        return emo_logits, sent_logits


if __name__ == "__main__":
    from datasets.meld_dataset import meld_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = meld_dataloader(
        csv_path="../dataset/train/train_sent_emo.csv",
        video_dir="../dataset/train/train_splits",
        batch_size=10,
        shuffle=True,
    )
    data = next(iter(loader))
    for data in tqdm.tqdm(loader):
        text_enc = data["text_inputs"]
        # print(
        #     text_enc["input_ids"].shape, text_enc["attention_mask"].shape
        # )  # (batch_size, seq_len)

        with torch.autocast("cuda"):
            # text_model = TextEncoder().to(device)
            # y = text_model(
            #     input_ids=text_enc["input_ids"].to(device),
            #     attention_mask=text_enc["attention_mask"].to(device),
            # )
            # print(y.shape)  # (batch_size, 128)

            # video_model = VideoEncoder().to(device)
            # video_frames = data["video_frames"].to(
            #     device
            # )  # (batch_size, frames, channels, H, W)
            # video_features = video_model(video_frames)
            # print(video_features.shape)

            # audio_model = AudioEncoder().to(device)
            audio_features = data["audio_features"].to(device)
            audio_features = audio_features.transpose(1, 2)
            audio_lengths = torch.tensor(
                [audio_features.shape[1]] * audio_features.shape[0], device=device
            )
            # audio_features = audio_model(audio_features, audio_lengths)
            # print(audio_features.shape)
            multimodal_model = MultimodalSentimentModel().to(device)
            text_enc = {
                "input_ids": text_enc["input_ids"].to(device),
                "attention_mask": text_enc["attention_mask"].to(device),
            }
            emo_logits, sent_logits = multimodal_model(
                text_inputs=text_enc,
                video_frames=data["video_frames"].to(device),
                audio_features=audio_features,
                audio_lengths=audio_lengths,
            )
            # print(emo_logits.shape)
            # print(sent_logits.shape)
