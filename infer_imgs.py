import torch
from model import UNET
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from dataset import CarvanaDatasetInfer
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    inference_prediction
)

# params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 450 # model req
IMAGE_WIDTH = 720 # model req
INFER_IMG_DIR = "inference_imgs/input/"


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    img_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    img_ds = CarvanaDatasetInfer(
        image_dir=INFER_IMG_DIR,
        transform=img_transforms,
    )

    img_loader = DataLoader(
        img_ds,
        shuffle=False
    )

    inference_prediction(
            img_loader, model, folder="inference_imgs/output/", device=DEVICE
        )
    

if __name__ == "__main__":
    main()