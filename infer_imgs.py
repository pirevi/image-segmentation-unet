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
IMAGE_HEIGHT = 160 # model req
IMAGE_WIDTH = 240 # model req
INFER_IMG_DIR = "inference_imgs/input/"
MODEL_SAVE_NAME = "model_160x240.pth.tar"


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(MODEL_SAVE_NAME), model)

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