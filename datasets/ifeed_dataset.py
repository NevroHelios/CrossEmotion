from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
import torchvision
import torch
from pathlib import Path

DATA_FOLDER = "data/ifeed/IFEED_Base"


class IFeedDataset(Dataset):
    """
    emotions in `meld` dataset: neutral, joy, sadness, anger, surprise, fear, disgust
    emotions in `ifeed` dataset: neutral, happy, sad, angry, surprise, fear, disgust

    Consistent Mapping with MELD:
    - angry -> anger (0)
    - disgust -> disgust (1)
    - fear, fea -> fear (2)
    - happy -> joy (3)
    - neutral -> neutral (4)
    - sad -> sadness (5)
    - surprise -> surprise (6)
    """

    def __init__(self, data_dir: str = DATA_FOLDER):
        self.files = list(Path(data_dir).glob("*/*"))
        self.emotion_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "fea": 2,
            "happy": 3,
            "neutral": 4,
            "sad": 5,
            "surprise": 6,
        }

    def _get_image_data(self, image_path: str | Path):
        image_data = torchvision.io.decode_image(str(image_path))
        image_data = image_data.float() / 255.0
        return image_data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            image = self.files[idx]
            image_tensor = self._get_image_data(image)
            emotion = image.name.split("_")[-1].split(".")[0]
            return image_tensor, self.emotion_map[emotion]
        except Exception as e:
            print(f"Error processing file {self.files[idx]}: {e}")
            return None, None


def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def ifeed_dataloader(
    data_dir: str = DATA_FOLDER,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = IFeedDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    loader = ifeed_dataloader(
        data_dir="data/ifeed/IFEED_Base/Training", batch_size=2, num_workers=0
    )
    for batch in loader:
        print(batch)
        break
