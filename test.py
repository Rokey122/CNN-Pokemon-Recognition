import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from pathlib import Path
import PIL
from PIL import Image

from model import *

model_name = Path("model.pt")
if not model_name.exists():
    model_name = input("Couldn't load model.pt. Input the model you want to load > ")
model = torch.load(Path(model_name), weights_only=False).to('cpu')

while True:
    image_name = input("Input the image file you want to pass to the model to get its prediction. Input q to quit. > ")
    if image_name == "q":
        break
    img = Image.open(image_name)

    print("The model predicted that the Pokemon on the image is: " + model(img))
