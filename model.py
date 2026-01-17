import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import PIL
from PIL import Image

from hyperparams import *




class PokemonRecognition(nn.Module):

  class CNN_Block(nn.Module):
    def __init__(self, in_channels, hidden_units):
      super().__init__()
      self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.MaxPool2d(kernel_size=2)
      )

    def forward(self, x):
      return self.conv_block(x)




  image_transform_train = v2.Compose([
      v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
      v2.TrivialAugmentWide(num_magnitude_bins=TRIVIAL_AUGMENT_BINS),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True)
  ])

  image_transform_test = v2.Compose([
      v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True)
  ])




  def __init__(self, in_channels=3, cnn_hidden_units=50, classification_hidden_units=50,
               classes=None):
    super().__init__()
    self.classes = classes
    self.cnn = nn.Sequential(
        self.CNN_Block(in_channels=in_channels, hidden_units=cnn_hidden_units),
        self.CNN_Block(in_channels=cnn_hidden_units,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**2),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**2,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**3),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**3,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**4)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=int(cnn_hidden_units*(HIDDEN_UNITS_MULTIPLICATION**(CONV_LAYERS-1))*((IMAGE_SIZE/(2**(CONV_LAYERS)))**2)),
                  out_features=classification_hidden_units),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=classification_hidden_units),
        nn.Dropout(p=DROPOUT_LINEAR),
        nn.Linear(in_features=classification_hidden_units, out_features=len(classes))
    )




  def _forward(self, x):
    x = self.cnn(x)
    x = self.classifier(x)
    return x
  
  def predict(self, x: torch.Tensor):
    preds = self._forward(x)
    return self.classes[preds.argmax()]
  
  def forward(self, x, classes=None):
    if isinstance(x, torch.Tensor):
      if x.size(dim=2) != 224 or x.size(dim=3) != 224:
        raise ValueError("Input tensor should be of 224x224 pixels")
      return self._forward(x)
    elif isinstance(x, Image.Image):
      x = x.convert("RGB")
      x = self.image_transform_test(x).unsqueeze(dim=0)
      return self.predict(x)
    else:
      raise("Incompatible input. You have to input either a torch.Tensor or a PIL Image.Image")