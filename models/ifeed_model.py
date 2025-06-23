from torchvision import models as vision_models
import torch.nn as nn


class IFeedModel(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.backbone = vision_models.resnet50(weights=vision_models.ResNet50_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Sequential( # type: ignore
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        # x = self.backbone(x)
        # x = self.fc(x)
        return self.backbone(x)
    


if __name__ == "__main__":
    from datasets.ifeed_dataset import ifeed_dataloader
    loader = ifeed_dataloader(data_dir="data/ifeed/IFEED_Base/Training", 
                              batch_size=32, 
                              shuffle=True)
    batch = next(iter(loader))
    image_data, _ = batch
    print(image_data.shape)
    model = IFeedModel()
    # print(model)

    y = model(image_data)
    print(y.argmax(dim=1))